import Combine
import Foundation
import UIKit

@MainActor
final class ShoppingViewModel: ObservableObject {
    let camera = GlassesCamera()
    let audio = AudioController()
    let conversation = ConversationClient()
    @Published var voiceConsented = false
    @Published private(set) var automaticCaptureEnabled = true
    @Published private(set) var automaticCaptureStatus = "Automatic capture starts with conversation"
    @Published var serverAddress = UserDefaults.standard.string(forKey: "backendOrigin") ?? ""
    @Published var dashboardAddress = UserDefaults.standard.string(forKey: "dashboardOrigin") ?? ""
    // Development launches can supply the existing LAN token in the process
    // environment, avoiding repeated manual entry during hardware fixes. It is
    // still memory-only; normal launches and release builds receive no token.
    #if DEBUG
    @Published var bearerToken = ProcessInfo.processInfo.environment["SHELF_LAN_API_TOKEN"] ?? ""
    #else
    @Published var bearerToken = ""
    #endif
    @Published var mode: CaptureMode = GlassesCamera.sdkIncluded ? .liveGlasses : .livePhone
    @Published private(set) var analysisMode: AnalysisMode = .product
    @Published var goalsText = "Avoid sugar"
    @Published var avoidIngredientsText = ""
    @Published var consented = false
    @Published var question = "Is this a good option for me?"
    @Published var recipient = "self"
    @Published var portionText = ""
    @Published var showPhoneCamera = false
    @Published var labelChecked = false
    @Published var attachSessionID = "" {
        didSet { pendingAttach = nil; attachReviewed = false }
    }
    @Published var attachReviewed = false
    @Published private(set) var pendingAttach: ShoppingSession?
    @Published private(set) var session: ShoppingSession?
    @Published private(set) var lastResponse: TurnResponse?
    @Published private(set) var conversationSnapshot: ConversationSnapshot?
    @Published private(set) var pendingPhoto: CapturedPhoto?
    @Published private(set) var lastPhoto: CapturedPhoto?
    @Published private(set) var lastProductBoxes: [DetectionBox] = []
    @Published private(set) var busy = false
    @Published private(set) var stage = "Ready to connect"
    @Published var error: String?
    @Published private(set) var health = "Backend not checked"

    private var api: APIClient?
    private var operation: Task<Void, Never>?
    private var generation = UUID()
    private var cancellationBarrier: Task<Void, Never>?
    private var conversationUpdates: AnyCancellable?
    private var automaticCaptureTask: Task<Void, Never>?
    private var automaticCaptureContext = UUID()
    private var automaticCaptureRetryAt: TimeInterval = 0
    private var latestConversationCaptureReference: (id: UUID, pixels: [UInt8], uptime: TimeInterval)?
    private var cameraRecoveryTask: Task<Void, Never>?
    private var cameraRecoveryBarrier: Task<Void, Never>?
    private var cameraRecoveryContext = UUID()
    private var failedStillRecoveryCount = 0
    private var cameraRecoveryNeedsRestart = false

    init() {
        audio.onInterrupted = { [weak self] in self?.stop() }
        conversation.onCameraCancelled = { [weak self] in
            self?.camera.stop()
            self?.scheduleCameraRecovery()
        }
        conversation.onEnded = { [weak self] in
            guard let self else { return }
            self.stopAutomaticCapture()
            self.cancelCameraRecovery()
            self.camera.stop()
            self.stage = "Conversation ended · microphone and automatic capture off"
        }
        conversationUpdates = conversation.objectWillChange.sink { [weak self] in self?.objectWillChange.send() }
        conversation.onEvent = { [weak self] event in
            guard let self, self.conversation.isRunning else { return }
            if event["type"] as? String == "scene.accepted" {
                self.automaticCaptureStatus = "Steady view accepted · taking a fresh photo"
            } else if event["type"] as? String == "scene.ignored" {
                let reason = event["reason"] as? String ?? "busy"
                self.automaticCaptureStatus = "Waiting until the conversation is ready"
                self.automaticCaptureRetryAt = ProcessInfo.processInfo.systemUptime + (reason == "cooldown" ? 8 : 2)
                self.automaticCaptureContext = UUID()
            }
            if event["type"] as? String == "snapshot",
               let snapshot = event["snapshot"] as? [String: Any],
               let data = try? JSONSerialization.data(withJSONObject: snapshot),
               let decoded = try? JSONDecoder().decode(ConversationSnapshot.self, from: data) {
                self.recipient = decoded.recipient_id
                self.conversationSnapshot = decoded
            }
        }
    }

    func startConversation() {
        guard !busy, !conversation.isRunning else { return }
        guard voiceConsented, let current = session, current.profile.consented else {
            error = "Review the profile and allow voice processing before starting conversation."
            return
        }
        perform { [self] token in try await beginConversation(current, token: token) }
    }

    private func beginConversation(_ current: ShoppingSession, token: UUID) async throws {
        try check(token)
        guard voiceConsented, current.profile.consented, session?.id == current.id else {
            throw AppError.message("Review this session and allow voice processing before starting conversation.")
        }
        audio.stop()
        lastResponse = nil
        lastProductBoxes = []
        conversationSnapshot = nil
        // An explicit conversation start permits one fresh recovery attempt.
        // Merely restarting the camera must not reset this budget, or repeated
        // still timeouts could create an endless automatic capture/restart loop.
        failedStillRecoveryCount = 0
        cameraRecoveryNeedsRestart = false
        // Finish DAT startup before opening voice: camera requests have a
        // bounded deadline for the established eight-second still path.
        if mode == .liveGlasses, !camera.isStreaming { try await camera.start() }
        try check(token)
        try await conversation.start(address: serverAddress, token: bearerToken,
            sessionID: current.id, recipientID: recipient, requireGlasses: mode == .liveGlasses,
            automaticCaptureEnabled: automaticCaptureEnabled,
            frameProvider: { [weak self] in
                guard let self, self.session?.id == current.id else { throw CancellationError() }
                try self.check(token)
                self.lastProductBoxes = []
                self.conversationSnapshot = nil
                if self.mode == .liveGlasses {
                    guard self.camera.isStreaming else {
                        self.recoverStoppedCameraAfterCaptureFailure(token: token, sessionID: current.id)
                        throw AppError.message(self.cameraRecoveryTask == nil
                            ? "The glasses camera disconnected. End and restart conversation."
                            : "The glasses camera is reconnecting after an interruption. Wait a moment, then ask again.")
                    }
                    // A conversation image request uses the proven fresh
                    // JPEG capture, never a cached preview frame.
                    if let frame = self.camera.freshFrame(), let pixels = SceneStabilityGate.sample(frame.image) {
                        self.latestConversationCaptureReference = (UUID(), pixels, ProcessInfo.processInfo.systemUptime)
                    }
                    let photo: CapturedPhoto
                    do {
                        photo = try await self.camera.capture()
                    } catch {
                        // The eight-second still timeout intentionally stops
                        // DAT to reject unidentified late photo callbacks. It
                        // is a failure, not a ConversationClient cancellation,
                        // so schedule recovery here as well as in onCancelled.
                        if !(error is CancellationError), !Task.isCancelled {
                            self.recoverStoppedCameraAfterCaptureFailure(token: token, sessionID: current.id)
                        }
                        throw error
                    }
                    try self.check(token)
                    self.failedStillRecoveryCount = 0
                    self.cameraRecoveryNeedsRestart = false
                    self.lastPhoto = photo
                    return photo
                }
                guard let photo = self.pendingPhoto,
                      Date().timeIntervalSince(photo.capturedAt) < 30 else { return nil }
                self.pendingPhoto = nil
                return photo
            })
        try check(token)
        startAutomaticCapture(token: token)
        stage = mode == .liveGlasses && automaticCaptureEnabled
            ? "Hold one or two products steady; I’ll compare them automatically. You can ask a follow-up."
            : "Conversation active · ask for a product comparison or a follow-up"
    }

    func endConversation() {
        stopAutomaticCapture()
        cancelCameraRecovery()
        conversation.stop()
        stage = "Conversation stopped · microphone off"
    }

    func setAutomaticCapture(_ enabled: Bool) {
        automaticCaptureEnabled = enabled
        automaticCaptureContext = UUID()
        automaticCaptureRetryAt = 0
        automaticCaptureStatus = enabled ? "Hold the product fronts steady" : "Automatic capture off · ask for a photo when needed"
        conversation.configureAutomaticCapture(enabled && mode == .liveGlasses)
    }

    private func startAutomaticCapture(token: UUID) {
        stopAutomaticCapture()
        automaticCaptureContext = UUID()
        automaticCaptureRetryAt = 0
        latestConversationCaptureReference = nil
        automaticCaptureTask = Task { [weak self] in
            var gate = SceneStabilityGate()
            var lastFrameTimestamp: Date?
            var context: UUID?
            var captureReferenceID: UUID?
            while !Task.isCancelled {
                do { try await Task.sleep(nanoseconds: 300_000_000) } catch { return }
                guard let self, self.generation == token, self.conversation.isRunning else { return }
                if context != self.automaticCaptureContext {
                    context = self.automaticCaptureContext
                    gate = SceneStabilityGate()
                    lastFrameTimestamp = nil
                    captureReferenceID = self.latestConversationCaptureReference?.id
                }
                if let reference = self.latestConversationCaptureReference, reference.id != captureReferenceID {
                    gate.recordCapture(reference.pixels, at: reference.uptime)
                    captureReferenceID = reference.id
                }
                guard self.automaticCaptureEnabled, self.mode == .liveGlasses else {
                    gate.breakStability()
                    self.automaticCaptureStatus = self.mode == .liveGlasses ? "Automatic capture off" : "Automatic capture requires live glasses"
                    continue
                }
                guard !self.cameraRecoveryNeedsRestart else {
                    gate.breakStability()
                    self.automaticCaptureStatus = "Camera recovery paused · end and start conversation to retry"
                    continue
                }
                guard self.cameraRecoveryTask == nil else {
                    gate.breakStability()
                    self.automaticCaptureStatus = "Reconnecting glasses camera after interruption"
                    continue
                }
                guard self.voiceConsented, self.conversation.isConnected,
                      let frame = self.camera.freshFrame(), frame.capturedAt != lastFrameTimestamp,
                      let pixels = SceneStabilityGate.sample(frame.image) else {
                    gate.breakStability()
                    self.automaticCaptureStatus = "Waiting for fresh glasses preview frames"
                    continue
                }
                lastFrameTimestamp = frame.capturedAt
                let now = ProcessInfo.processInfo.systemUptime
                let eligible = self.conversation.canRequestAutomaticCapture && now >= self.automaticCaptureRetryAt
                if gate.consume(pixels, at: now, eligible: eligible), self.conversation.requestAutomaticCapture() {
                    self.automaticCaptureStatus = "Steady view found · requesting a fresh photo"
                } else if !gate.hasUsableView {
                    self.automaticCaptureStatus = "Show well-lit package fronts in the glasses view"
                } else if !eligible {
                    self.automaticCaptureStatus = "Waiting for speech and the current answer to finish"
                } else if !gate.isArmed {
                    self.automaticCaptureStatus = "View already captured · move to new products to capture again"
                } else {
                    self.automaticCaptureStatus = "Hold the product fronts steady"
                }
            }
        }
    }

    private func stopAutomaticCapture() {
        automaticCaptureTask?.cancel()
        automaticCaptureTask = nil
        automaticCaptureContext = UUID()
        automaticCaptureStatus = "Automatic capture stopped"
    }

    private func recoverStoppedCameraAfterCaptureFailure(token: UUID, sessionID: String) {
        guard generation == token, session?.id == sessionID, conversation.isRunning,
              mode == .liveGlasses, !camera.isStreaming, cameraRecoveryTask == nil else { return }
        guard failedStillRecoveryCount < 1 else {
            cameraRecoveryNeedsRestart = true
            automaticCaptureStatus = "Camera recovery paused · end and start conversation to retry"
            error = "The glasses camera stopped again after recovery. End and start conversation to reconnect."
            return
        }
        failedStillRecoveryCount += 1
        scheduleCameraRecovery()
    }

    private func scheduleCameraRecovery() {
        guard conversation.isRunning, mode == .liveGlasses, let current = session,
              cameraRecoveryTask == nil, !cameraRecoveryNeedsRestart else { return }
        let token = generation
        let context = cameraRecoveryContext
        let previousRecovery = cameraRecoveryBarrier
        automaticCaptureStatus = "Reconnecting glasses camera after interruption"
        cameraRecoveryTask = Task { [weak self] in
            guard let self else { return }
            await previousRecovery?.value
            defer { if self.cameraRecoveryContext == context { self.cameraRecoveryTask = nil } }
            while !Task.isCancelled {
                guard self.generation == token, self.cameraRecoveryContext == context,
                      self.session?.id == current.id, self.conversation.isRunning else { return }
                // A restart can take longer than a still request's deadline.
                // Wait for the conversation to settle, then recover separately.
                if self.conversation.isIdleForCameraRecovery { break }
                do { try await Task.sleep(nanoseconds: 300_000_000) } catch { return }
            }
            guard !Task.isCancelled, self.generation == token, self.cameraRecoveryContext == context,
                  self.session?.id == current.id, self.conversation.isRunning else { return }
            do {
                try await self.camera.start()
                try Task.checkCancellation()
                guard self.generation == token, self.cameraRecoveryContext == context,
                      self.session?.id == current.id, self.conversation.isRunning else { return }
                self.automaticCaptureContext = UUID()
                self.automaticCaptureStatus = "Camera ready · hold the product fronts steady"
            } catch is CancellationError {
                // Stop/session/context changes own cancellation and stale work.
            } catch {
                guard self.generation == token, self.cameraRecoveryContext == context,
                      self.session?.id == current.id, self.conversation.isRunning else { return }
                if self.failedStillRecoveryCount > 0 { self.cameraRecoveryNeedsRestart = true }
                self.error = "The glasses camera could not reconnect: \(error.localizedDescription)"
                self.automaticCaptureStatus = "Camera reconnect needs attention"
            }
        }
    }

    private func cancelCameraRecovery() {
        cameraRecoveryContext = UUID()
        let previousBarrier = cameraRecoveryBarrier
        let pending = cameraRecoveryTask
        pending?.cancel()
        cameraRecoveryTask = nil
        // Await the old start() cleanup before any later camera operation. This
        // keeps a canceled startup's defer from stopping a replacement stream.
        cameraRecoveryBarrier = Task {
            await previousBarrier?.value
            await pending?.value
        }
    }

    private func makeAPI() throws -> APIClient {
        let client = try APIClient(address: serverAddress, bearerToken: bearerToken)
        UserDefaults.standard.set(serverAddress, forKey: "backendOrigin")
        UserDefaults.standard.set(dashboardAddress, forKey: "dashboardOrigin")
        return client
    }

    private func perform(_ body: @escaping (UUID) async throws -> Void) {
        operation?.cancel()
        let token = UUID()
        generation = token
        busy = true
        error = nil
        operation = Task { [weak self] in
            guard let self else { return }
            await cancellationBarrier?.value
            await cameraRecoveryBarrier?.value
            do {
                try Task.checkCancellation()
                try await body(token)
            } catch is CancellationError {
                // Stop/context generation owns cancellation; it is not a provider failure.
            } catch {
                if generation == token {
                    self.error = error.localizedDescription
                    stage = "Needs attention"
                }
            }
            if generation == token { busy = false; operation = nil }
        }
    }

    private func check(_ token: UUID) throws {
        try Task.checkCancellation()
        guard generation == token else { throw CancellationError() }
    }

    func checkHealth() {
        health = "Checking connection and authorization…"
        perform { [self] token in
            let client = try makeAPI()
            let result: HealthResponse = try await client.request("health")
            try check(token)
            try await client.checkAuthorization()
            try check(token)
            let configured = result.providers.filter(\.value).map(\.key).sorted().joined(separator: ", ")
            health = "\(result.status) · authorized · \(result.version) · configured: \(configured.isEmpty ? "none" : configured)"
            stage = "Backend reached and authorized; providers still need a real request"
        }
    }

    func startSession() {
        createSession(startVoice: false)
    }

    func startConversationFromSetup() {
        guard voiceConsented else { error = "Allow microphone transcription before starting conversation."; return }
        createSession(startVoice: true)
    }

    private func createSession(startVoice: Bool) {
        guard !busy, session == nil, !conversation.isRunning else { return }
        guard consented else { error = "Confirm the profile and image-processing consent before starting."; return }
        let profile = ShopperProfile(display_name: "Avoid sugar",
                                     goals: split(goalsText), avoid_ingredients: split(avoidIngredientsText),
                                     consented: consented)
        perform { [self] token in
            var startedSuccessfully = false
            defer { if !startedSuccessfully, generation == token { camera.stop() } }
            let client = try makeAPI()
            try await client.checkAuthorization()
            stage = mode == .liveGlasses ? "Starting glasses; approve Meta AI if prompted" : "Starting phone-camera fallback"
            if mode == .liveGlasses { try await camera.start() }
            try check(token)
            let created: ShoppingSession = try await client.post("sessions", body: CreateSessionRequest(mode: mode.rawValue, profile: profile))
            try check(token)
            api = client
            conversation.resetSessionEvidence()
            session = created
            conversationSnapshot = nil
            recipient = created.recipient_id
            portionText = ""
            lastResponse = nil
            pendingPhoto = nil
            lastPhoto = nil
            lastProductBoxes = []
            stage = "Session started · capture only when you ask"
            startedSuccessfully = true
            if startVoice { try await beginConversation(created, token: token) }
        }
    }

    /// Read an existing dashboard-created session without copying or replacing its profile.
    func loadSessionToAttach() {
        guard session == nil else { error = "Disconnect from the current session before switching."; return }
        guard let identifier = UUID(uuidString: attachSessionID.trimmingCharacters(in: .whitespacesAndNewlines)) else {
            error = "Paste the session ID from the dashboard, not the whole dashboard URL."
            return
        }
        let id = identifier.uuidString.lowercased()
        pendingAttach = nil
        attachReviewed = false
        perform { [self] token in
            let client = try makeAPI()
            let loaded: ShoppingSession = try await client.request("sessions/\(id)")
            try check(token)
            mode = try supportedAttachMode(loaded)
            pendingAttach = loaded
            stage = "Review the dashboard profile and source before attaching"
        }
    }

    private func supportedAttachMode(_ value: ShoppingSession) throws -> CaptureMode {
        guard let source = CaptureMode(rawValue: value.mode) else {
            throw AppError.message("This is a \(value.mode) session. Create a Live glasses or Live phone session in the dashboard; the iPhone cannot relabel synthetic or recorded data as live.")
        }
        guard source != .liveGlasses || GlassesCamera.sdkIncluded else {
            throw AppError.message("This phone-only build cannot attach a Live glasses session. Use the DAT build or create a Live phone session.")
        }
        guard value.profile.consented else {
            throw AppError.message("This dashboard profile has not been consented. Review and confirm it in the dashboard before attaching.")
        }
        return source
    }

    func attachLoadedSession() {
        guard session == nil, let reviewed = pendingAttach, attachReviewed else {
            error = "Load a session and confirm its displayed profile before attaching."
            return
        }
        perform { [self] token in
            var attachedSuccessfully = false
            defer { if !attachedSuccessfully { camera.stop() } }
            let client = try makeAPI()
            let refreshed: ShoppingSession = try await client.request("sessions/\(reviewed.id)")
            try check(token)
            let attachedMode = try supportedAttachMode(refreshed)
            guard refreshed.profile == reviewed.profile, refreshed.mode == reviewed.mode else {
                pendingAttach = nil
                attachReviewed = false
                throw AppError.message("The session profile or source changed. Load and review it again before attaching.")
            }
            stage = attachedMode == .liveGlasses ? "Connecting glasses to the reviewed dashboard session" : "Attaching the reviewed phone session"
            if attachedMode == .liveGlasses { try await camera.start() }
            try check(token)
            // Clear any previous wearer's pending action and invalidate in-flight replies.
            // Reviewed product facts, profile metadata, confirmed cart, and context remain server-owned.
            let joined = try await client.cancel(refreshed.id)
            try check(token)
            guard joined.profile == reviewed.profile, joined.mode == reviewed.mode else {
                camera.stop()
                pendingAttach = nil
                attachReviewed = false
                throw AppError.message("The profile changed during connection. Load and review the session again.")
            }
            api = client
            mode = attachedMode
            conversation.resetSessionEvidence()
            session = joined
            conversationSnapshot = nil
            voiceConsented = false
            recipient = joined.recipient_id
            portionText = joined.portion_grams.map { String(format: "%g", $0) } ?? ""
            lastResponse = nil
            pendingPhoto = nil
            lastPhoto = nil
            lastProductBoxes = []
            labelChecked = false
            pendingAttach = nil
            attachReviewed = false
            question = "Is this a good option for me?"
            stage = "Attached to \(joined.profile.display_name) · previous pending actions cleared"
            attachedSuccessfully = true
        }
    }

    /// Leave a shared session locally; preserve its reviewed facts/profile on the server.
    func disconnectSession() {
        stop()
        conversation.resetSessionEvidence()
        audio.deactivate()
        session = nil
        conversationSnapshot = nil
        voiceConsented = false
        api = nil
        lastResponse = nil
        pendingPhoto = nil
        lastPhoto = nil
        lastProductBoxes = []
        pendingAttach = nil
        attachSessionID = ""
        attachReviewed = false
        labelChecked = false
        consented = false
        recipient = "self"
        portionText = ""
        question = "Is this a good option for me?"
        stage = "Disconnected locally · server session preserved"
    }

    private func split(_ text: String) -> [String] {
        text.split(separator: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
    }

    func capturePhonePhoto(_ image: UIImage) {
        showPhoneCamera = false
        guard mode == .livePhone, session != nil else { return }
        let photo = CapturedPhoto(image: image, capturedAt: Date(), mode: .livePhone)
        pendingPhoto = photo
        lastPhoto = photo
        lastProductBoxes = []
        stage = "Phone photo ready · tap Ask to send"
    }

    func openPhoneCamera() {
        guard UIImagePickerController.isSourceTypeAvailable(.camera) else {
            error = "The phone camera is unavailable. Use a physical iPhone; no synthetic image has been substituted."
            return
        }
        audio.stop()
        if conversation.isRunning { endConversation() }
        showPhoneCamera = true
    }

    func ask(captureNewGlassesPhoto: Bool = false) {
        if conversation.isRunning { conversation.sendText(question); return }
        guard let client = api, let current = session else { return }
        let text = question.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { error = "Speak or type a question first."; return }
        let portion: Double?
        if analysisMode == .product || portionText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { portion = nil }
        else if let value = Double(portionText), value.isFinite, value > 0, value <= 5000 { portion = value }
        else { error = "Enter a portion in grams greater than 0 and no more than 5,000, or leave it blank so Shelf Aware can ask."; return }
        audio.stop()
        let requestedAnalysis = analysisMode
        perform { [self] token in
            var photo = pendingPhoto
            if captureNewGlassesPhoto {
                guard mode == .liveGlasses else { throw AppError.message("Select live glasses before requesting a glasses photo.") }
                lastProductBoxes = []
                stage = requestedAnalysis == .product ? "Taking a new glasses product photo" : "Taking a new glasses label photo"
                if !camera.isStreaming { try await camera.start() }
                photo = try await camera.capture()
            }
            try check(token)
            if let photo {
                guard photo.mode == mode else { throw AppError.message("This image belongs to another capture mode. Capture again.") }
                lastPhoto = photo
                lastProductBoxes = []
            }
            stage = photo == nil ? "Asking about the current product" : requestedAnalysis == .product ? "Identifying held products and looking up sources" : "Sending the captured label"
            let barcode: String?
            if requestedAnalysis == .label { barcode = await LocalBarcode.detect(in: photo?.image) }
            else { barcode = nil }
            try check(token)
            let request = TurnRequest(text: text, client_turn_id: UUID().uuidString, recipient_id: recipient, portion_grams: portion, image_data_url: try photo?.dataURL(), captured_at: photo.map { ISO8601DateFormatter().string(from: $0.capturedAt) }, barcode: barcode, analysis_mode: requestedAnalysis.rawValue)
            let response: TurnResponse = try await client.post("sessions/\(current.id)/turn", body: request)
            try check(token)
            if let photo, requestedAnalysis == .product, response.status != "cancelled" {
                lastProductBoxes = (response.events ?? [])
                    .filter { $0.type == "product.detected" }
                    .compactMap(\.box)
                    .filter { $0.rect(in: photo.image.size, image: photo.image.size) != nil }
            }
            pendingPhoto = nil
            await apply(response, token: token)
        }
    }

    func changeAnalysisMode(_ newMode: AnalysisMode) {
        guard analysisMode != newMode, !busy, !conversation.isRunning else { return }
        conversation.resetSessionEvidence()
        conversationSnapshot = nil
        analysisMode = newMode
        lastResponse = nil
        lastProductBoxes = []
        labelChecked = false
        stage = newMode == .product ? "Show one or two products with front names and branding visible" : "Show a readable nutrition panel with its serving-size line"
    }

    private func apply(_ response: TurnResponse, token: UUID) async {
        guard generation == token, !Task.isCancelled else { return }
        session = response.session
        conversationSnapshot = nil
        lastResponse = response
        recipient = response.session.recipient_id
        portionText = response.session.portion_grams.map { String(format: "%g", $0) } ?? ""
        labelChecked = false
        stage = response.status.replacingOccurrences(of: "_", with: " ")
        guard response.status != "cancelled" else { lastProductBoxes = []; return }
        do { try await audio.speak(response.reply, requireGlasses: mode == .liveGlasses) }
        catch {
            if generation == token, !Task.isCancelled {
                self.error = "The answer is on screen. Audio: \(error.localizedDescription)"
            }
        }
    }

    func changeRecipient(_ newRecipient: String) {
        guard newRecipient != recipient else { return }
        cancelCameraRecovery()
        recipient = newRecipient
        portionText = ""
        labelChecked = false
        lastResponse = nil
        lastProductBoxes = []
        conversationSnapshot = nil
        automaticCaptureContext = UUID()
        if conversation.isRunning {
            conversation.setRecipient(newRecipient)
            if !camera.isStreaming { scheduleCameraRecovery() }
            return
        }
        conversation.resetSessionEvidence()
        guard let client = api, let current = session else { return }
        audio.stop()
        perform { [self] token in
            let updated: ShoppingSession = try await client.post("sessions/\(current.id)/context", body: ContextRequest(recipient_id: newRecipient, portion_grams: nil))
            try check(token)
            session = updated
            lastResponse = nil
            stage = newRecipient == "self" ? "Shopping for me · choose a portion" : "Shopping for household · personal goals are not applied"
        }
    }

    func reviewLabel(confirmed: Bool) {
        guard !conversation.isRunning else { return }
        guard let client = api, let current = session, let product = current.currentProduct else { return }
        guard !confirmed || labelChecked else { error = "Check the physical package against every displayed field before confirming."; return }
        audio.stop()
        perform { [self] token in
            let response: TurnResponse = try await client.post("sessions/\(current.id)/review", body: ReviewRequest(product_id: product.id, confirmed: confirmed))
            try check(token)
            await apply(response, token: token)
        }
    }

    func confirmCart() {
        guard !conversation.isRunning else { return }
        guard let client = api, let current = session, let action = current.pending_action else { return }
        guard current.recipient_id == recipient, action.recipient_id == recipient else {
            error = "The recipient change has not been confirmed by the backend. Ask again for the intended recipient before adding."
            return
        }
        audio.stop()
        perform { [self] token in
            let response: TurnResponse = try await client.post("sessions/\(current.id)/confirm", body: ConfirmRequest(action_id: action.id))
            try check(token)
            await apply(response, token: token)
        }
    }

    func toggleListening() {
        if conversation.isRunning { conversation.interrupt(); return }
        if audio.isListening { audio.stopListening(); return }
        perform { [self] token in
            try await audio.startListening(requireGlasses: mode == .liveGlasses) { [weak self] text in self?.question = text }
            try check(token)
            stage = "Listening for up to 12 seconds · tap Finish, review, then Ask"
        }
    }

    /// Cancels the network task and invalidates backend generations before another turn.
    func stop() {
        stopAutomaticCapture()
        cancelCameraRecovery()
        conversation.stop()
        generation = UUID()
        let stopGeneration = generation
        operation?.cancel()
        operation = nil
        busy = false
        lastResponse = nil
        lastProductBoxes = []
        audio.stop()
        camera.stop()
        stage = "Stopped · no listening"
        guard let client = api, let current = session else { return }
        let previousBarrier = cancellationBarrier
        cancellationBarrier = Task { [weak self] in
            await previousBarrier?.value
            do {
                let updated = try await client.cancel(current.id)
                guard let self, self.generation == stopGeneration, self.session?.id == current.id else { return }
                self.session = updated
                self.lastResponse = nil
            } catch {
                guard let self, self.generation == stopGeneration else { return }
                self.error = "Local audio/request stopped. Backend cancellation could not be confirmed: \(error.localizedDescription)"
            }
        }
    }

    func endSession() {
        stop()
        camera.stop()
        audio.deactivate()
        guard let client = api, let current = session else { return }
        perform { [self] token in
            let deleted: DeleteResponse = try await client.request("sessions/\(current.id)", method: "DELETE")
            try check(token)
            guard deleted.deleted else { throw AppError.message("The backend did not confirm deletion. Retry End and delete.") }
            conversation.resetSessionEvidence()
            session = nil
            conversationSnapshot = nil
            voiceConsented = false
            lastResponse = nil
            pendingPhoto = nil
            lastPhoto = nil
            lastProductBoxes = []
            api = nil
            question = "Is this a good option for me?"
            portionText = ""
            stage = "Session deleted · camera and audio stopped"
        }
    }

    func backgrounded() {
        // The registration/permission redirect happens before an application session exists.
        guard session != nil || camera.isStreaming else { return }
        stop()
        camera.stop()
        audio.deactivate()
        stage = "Paused in background · return and start conversation or request a new photo"
    }

    var dashboardURL: URL? {
        guard let session else { return nil }
        let origin: String
        if dashboardAddress.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            guard var components = URLComponents(string: serverAddress) else { return nil }
            components.port = 5173
            components.path = ""
            origin = components.string ?? ""
        } else { origin = dashboardAddress }
        guard var components = URLComponents(string: origin), components.host != nil else { return nil }
        components.queryItems = [URLQueryItem(name: "session", value: session.id)]
        return components.url
    }
}
