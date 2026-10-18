import Combine
import Foundation
import UIKit

/// Foreground conversation controller. Provider credentials remain on the backend.
/// One writer drains a bounded queue; one reader processes ordered server events.
@MainActor
final class ConversationClient: ObservableObject {
    @Published private(set) var isRunning = false
    @Published private(set) var isConnected = false
    @Published private(set) var state = "Stopped"
    @Published private(set) var transcript = ""
    @Published private(set) var reply = ""
    @Published private(set) var error: String?
    @Published private(set) var route = "Audio stopped"
    @Published private(set) var microphoneLevel: Float = 0
    @Published private(set) var microphoneEnabled = false
    @Published private(set) var audioInputAvailable = false
    @Published private(set) var providerDescription = "Speech providers not connected"
    @Published private(set) var voiceName = ""
    @Published private(set) var reconnectAttempt = 0
    @Published private(set) var droppedAudioChunks = 0
    var onEvent: (([String: Any]) -> Void)?
    var onCameraCancelled: (() -> Void)?
    var onEnded: (() -> Void)?

    private struct Outgoing {
        let json: String
        let bytes: Int
        let microphone: Bool
    }
    private let audio = ConversationAudio()
    private let nativeSpeech = ConversationNativeSpeech()
    private var useNativeSpeech = false
    private var transcriptionProvider = "unavailable"
    private let transport: URLSession
    private var socket: URLSessionWebSocketTask?
    private var microphoneTask: Task<Void, Never>?
    private var readerTask: Task<Void, Never>?
    private var writerTask: Task<Void, Never>?
    private var watchdogTask: Task<Void, Never>?
    private var reconnectTask: Task<Void, Never>?
    private var cameraTask: Task<Void, Never>?
    private var cameraRequestGeneration: UUID?
    private var cameraRequestIsAutomatic = false
    private var automaticCaptureEnabled = false
    private var automaticRequestPending = false
    private var speechInProgress = false
    private var serverListeningEnabled = false
    private var queue: [Outgoing] = []
    private var queueBytes = 0
    private var writerWake: CheckedContinuation<Void, Never>?
    private var generation = UUID()
    private var connectionGeneration = UUID()
    private var clientEpoch = 0
    private var request: URLRequest?
    private var sessionID = ""
    private var recipientID = "self"
    private var frameProvider: (@MainActor () async throws -> CapturedPhoto?)?
    private var connectedAt: TimeInterval = .greatestFiniteMagnitude
    private var connectionStartedAt: TimeInterval = 0
    private var lastReceivedAt: TimeInterval = 0
    private var sendingSince: TimeInterval?
    private var activeContext: String?
    private var endAfterPlaybackContext: String?
    private var cancelledContexts: [String] = []
    // A bounded 4 MB JPEG expands to about 5.34 MB in a camera JSON message.
    // Mic backlog remains independently capped to eight 40 ms packets.
    private let maximumQueueBytes = 6 * 1_024 * 1_024
    private let maximumIncomingMessageBytes = 4 * 1_024 * 1_024
    private let maximumMicrophoneChunks = 8 // At most 320 ms; older audio is dropped.

    init() {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.urlCache = nil
        configuration.httpCookieStorage = nil
        configuration.timeoutIntervalForRequest = 12
        transport = URLSession(configuration: configuration)
        audio.onFailure = { [weak self] message in
            self?.stop()
            self?.error = message
            self?.state = "Audio stopped"
        }
        audio.onRoute = { [weak self] value in self?.route = value }
        audio.onPlayback = { [weak self] type, contextID, playedMilliseconds in
            guard let self, self.isConnected else { return }
            if type == "playback.finished", self.endAfterPlaybackContext == contextID {
                self.finishConversation()
                return
            }
            if type != "playback.started", self.activeContext == contextID { self.activeContext = nil }
            self.enqueue(["type": type, "context_id": contextID, "played_ms": playedMilliseconds])
            self.applyMicrophoneGate()
        }
    }

    func start(
        address: String, token: String, sessionID: String,
        recipientID: String = "self", requireGlasses: Bool, automaticCaptureEnabled: Bool,
        frameProvider: @escaping @MainActor () async throws -> CapturedPhoto?
    ) async throws {
        stop()
        guard UUID(uuidString: sessionID) != nil else { throw AppError.message("Join or create a shopping session before starting conversation.") }
        let api = try APIClient(address: address, bearerToken: token)
        guard var components = URLComponents(url: api.baseURL, resolvingAgainstBaseURL: false) else {
            throw AppError.message("The backend address could not be parsed.")
        }
        components.scheme = api.baseURL.scheme == "https" ? "wss" : "ws"
        components.path = "/api/conversation/ws"
        components.queryItems = [URLQueryItem(name: "session_id", value: sessionID)]
        guard let url = components.url else { throw AppError.message("The conversation address could not be created.") }
        var request = URLRequest(url: url)
        if !api.bearerToken.isEmpty { request.setValue("Bearer \(api.bearerToken)", forHTTPHeaderField: "Authorization") }
        self.request = request
        self.sessionID = sessionID
        self.recipientID = recipientID
        self.automaticCaptureEnabled = automaticCaptureEnabled && requireGlasses
        self.frameProvider = frameProvider
        clearContextEvidence()
        droppedAudioChunks = 0
        reconnectAttempt = 0
        state = requireGlasses ? "Connecting glasses audio · waiting for Bluetooth" : "Starting microphone"
        let epoch = generation
        let chunks: AsyncStream<ConversationMicrophoneChunk>
        do { chunks = try await audio.start(requireGlasses: requireGlasses) }
        catch {
            if generation == epoch { audio.stop(); state = "Audio unavailable" }
            throw error
        }
        guard generation == epoch else { throw CancellationError() }
        if Task.isCancelled { audio.stop(); throw CancellationError() }
        isRunning = true
        microphoneTask = Task { [weak self] in
            for await chunk in chunks {
                guard let self, self.generation == epoch, !Task.isCancelled else { return }
                // Frames captured before readiness/reconnect are never replayed.
                guard self.isConnected, self.audioInputAvailable, self.microphoneEnabled,
                      chunk.uptime >= self.connectedAt else { continue }
                self.microphoneLevel = min(1, chunk.rms * 8)
                self.enqueue(["type": "audio.append", "audio": chunk.pcm.base64EncodedString()], microphone: true)
            }
            guard let self, self.generation == epoch, self.isRunning, !Task.isCancelled else { return }
            self.stop()
            self.error = "Microphone capture stopped. Start conversation again."
        }
        connect()
    }

    private func connect() {
        guard isRunning, let request else { return }
        closeConnection()
        let connection = UUID()
        connectionGeneration = connection
        let task = transport.webSocketTask(with: request)
        task.maximumMessageSize = maximumIncomingMessageBytes
        socket = task
        connectionStartedAt = ProcessInfo.processInfo.systemUptime
        lastReceivedAt = connectionStartedAt
        state = reconnectAttempt == 0 ? "Connecting voice" : "Reconnecting voice (\(reconnectAttempt)/3)"
        task.resume()
        writerTask = Task { [weak self] in await self?.writeLoop(socket: task, connection: connection) }
        readerTask = Task { [weak self] in await self?.readLoop(socket: task, connection: connection) }
        // The caller displays and obtains the separate microphone/cloud-audio consent.
        enqueue(["type": "session.start", "session_id": sessionID, "recipient_id": recipientID,
                 "voice_consented": true, "automatic_capture_enabled": automaticCaptureEnabled])
        watchdogTask = Task { [weak self] in
            while !Task.isCancelled {
                do { try await Task.sleep(nanoseconds: 3_000_000_000) } catch { return }
                guard let self, self.connectionGeneration == connection, self.isRunning else { return }
                let now = ProcessInfo.processInfo.systemUptime
                if (!self.isConnected && now - self.connectionStartedAt > 12) || now - self.lastReceivedAt > 18 || self.sendingSince.map({ now - $0 > 5 }) == true {
                    self.disconnected("Voice connection timed out.", connection: connection)
                    return
                }
                if self.isConnected { self.enqueue(["type": "ping"]) }
            }
        }
    }

    private func enqueue(_ event: [String: Any], microphone: Bool = false) {
        guard socket != nil, isRunning else { return }
        if microphone && (!isConnected || !microphoneEnabled) { return }
        var payload = event
        payload["client_epoch"] = clientEpoch
        guard JSONSerialization.isValidJSONObject(payload), let data = try? JSONSerialization.data(withJSONObject: payload),
              let json = String(data: data, encoding: .utf8), data.count <= maximumQueueBytes else {
            error = "A conversation message exceeded the supported size."
            return
        }
        if microphone && queue.filter(\.microphone).count >= maximumMicrophoneChunks,
           let oldest = queue.firstIndex(where: \.microphone) {
            queueBytes -= queue.remove(at: oldest).bytes
            droppedAudioChunks += 1
        }
        guard queue.count < 80, queueBytes + data.count <= maximumQueueBytes else {
            if microphone { droppedAudioChunks += 1; return }
            disconnected("Conversation upload queue became full.", connection: connectionGeneration)
            return
        }
        queue.append(Outgoing(json: json, bytes: data.count, microphone: microphone))
        queueBytes += data.count
        writerWake?.resume()
        writerWake = nil
    }

    private func writeLoop(socket: URLSessionWebSocketTask, connection: UUID) async {
        do {
            while !Task.isCancelled && connectionGeneration == connection {
                if queue.isEmpty {
                    await withCheckedContinuation { writerWake = $0 }
                    continue
                }
                let next = queue.removeFirst()
                queueBytes -= next.bytes
                // A queued chunk may have been produced before VAD completed.
                // Never send it after the listening gate closes.
                if next.microphone && !microphoneEnabled { continue }
                sendingSince = ProcessInfo.processInfo.systemUptime
                try await socket.send(.string(next.json))
                guard connectionGeneration == connection else { return }
                sendingSince = nil
            }
        } catch {
            guard !Task.isCancelled else { return }
            disconnected("Voice connection was lost while sending.", connection: connection)
        }
    }

    private func readLoop(socket: URLSessionWebSocketTask, connection: UUID) async {
        do {
            while !Task.isCancelled && connectionGeneration == connection {
                let message = try await socket.receive()
                guard connectionGeneration == connection, !Task.isCancelled else { return }
                let data: Data
                switch message {
                case .string(let text): data = Data(text.utf8)
                case .data(let bytes): data = bytes
                @unknown default: continue
                }
                guard data.count <= maximumIncomingMessageBytes,
                      let event = try JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let type = event["type"] as? String else {
                    throw AppError.message("The voice server returned an invalid event.")
                }
                lastReceivedAt = ProcessInfo.processInfo.systemUptime
                try handle(event, type: type, connection: connection)
            }
        } catch {
            guard !Task.isCancelled else { return }
            disconnected("Voice connection ended. Unsent audio was discarded.", connection: connection)
        }
    }

    private func handle(_ event: [String: Any], type: String, connection: UUID) throws {
        // Session/context resets invalidate both the connection and event epoch.
        // A late transcript or snapshot must not repopulate cleared evidence.
        guard isRunning, connectionGeneration == connection,
              let epoch = event["client_epoch"] as? Int, epoch == clientEpoch else { return }
        switch type {
        case "session.ready":
            useNativeSpeech = event["speech_provider"] as? String == "native" || event["audio_mode"] as? String == "native"
            audioInputAvailable = event["audio_input_available"] as? Bool ?? false
            let transcription = event["transcription_provider"] as? String ?? "unavailable"
            let speech = event["speech_provider"] as? String ?? "unavailable"
            transcriptionProvider = transcription
            providerDescription = "Transcription: \(providerLabel(transcription)) · Speech: \(useNativeSpeech ? "native iPhone voice" : providerLabel(speech))"
            if !useNativeSpeech { voiceName = "" }
            isConnected = true
            connectedAt = ProcessInfo.processInfo.systemUptime
            state = audioInputAvailable ? "Listening" : "Text input only · microphone transcription unavailable"
            error = audioInputAvailable ? nil : transcriptionUnavailableMessage
            serverListeningEnabled = event["listening_enabled"] as? Bool ?? false
            applyMicrophoneGate()
            if let enabled = event["automatic_capture_enabled"] as? Bool, enabled != automaticCaptureEnabled {
                enqueue(["type": "scene.configure", "enabled": automaticCaptureEnabled])
            }
        case "state":
            let receivedState = event["state"] as? String ?? state
            if receivedState.lowercased() != "listening" { speechInProgress = false }
            if ["transcribing", "thinking", "capturing", "speaking"].contains(receivedState.lowercased()) {
                pauseMicrophone()
            }
            state = !audioInputAvailable && receivedState.lowercased() == "listening" ? "Text input only · microphone transcription unavailable" : receivedState
        case "input.enabled":
            guard let enabled = event["enabled"] as? Bool else { return }
            serverListeningEnabled = enabled
            applyMicrophoneGate()
        case "transcript.partial", "transcript.final":
            transcript = event["text"] as? String ?? transcript
            if type == "transcript.final" { speechInProgress = false }
        case "snapshot":
            // The epoch check above rejects snapshots from before a local
            // recipient change. Adopt spoken/server changes without sending a
            // new context update or interrupting the response being delivered.
            if let snapshot = event["snapshot"] as? [String: Any],
               let recipient = snapshot["recipient_id"] as? String,
               recipient == "self" || recipient == "household" {
                recipientID = recipient
            }
        case "response.text":
            pauseMicrophone()
            if let text = event["text"] as? String { reply = text }
            if let context = event["context_id"] as? String, !cancelledContexts.contains(context) {
                activeContext = context
                endAfterPlaybackContext = (event["end_conversation"] as? Bool == true) ? context : nil
                audio.beginResponse(contextID: context)
                if useNativeSpeech, let text = event["text"] as? String {
                    nativeSpeech.speak(text, contextID: context, audio: audio) { [weak self] message in
                        guard let self, self.connectionGeneration == connection, self.activeContext == context else { return }
                        if self.endAfterPlaybackContext == context {
                            self.finishConversation()
                            self.error = message
                            return
                        }
                        self.activeContext = nil
                        self.enqueue(["type": "playback.interrupted", "context_id": context, "played_ms": 0])
                        self.error = message
                        self.applyMicrophoneGate()
                    }
                    voiceName = nativeSpeech.voiceDescription
                }
            }
        case "audio.chunk":
            guard !useNativeSpeech else { return }
            pauseMicrophone()
            guard let context = event["context_id"] as? String, !cancelledContexts.contains(context),
                  let sequence = event["sequence"] as? Int,
                  let encoded = event["audio"] as? String, encoded.utf8.count <= 64_000,
                  let bytes = Data(base64Encoded: encoded), let sampleRate = event["sample_rate"] as? Int else { return }
            if activeContext == nil { activeContext = context; audio.beginResponse(contextID: context) }
            guard activeContext == context else { return }
            try audio.append(bytes, contextID: context, sequence: sequence, sampleRate: sampleRate)
        case "audio.done":
            guard !useNativeSpeech else { return }
            if let context = event["context_id"] as? String { audio.finishResponse(contextID: context) }
        case "speech.started":
            // No ambient sound or stale VAD event may cancel a protected reply.
            guard microphoneEnabled, activeContext == nil else { return }
            speechInProgress = true
        case "turn.progress":
            if event["stage"] as? String == "photo_received" {
                state = "Photo received · checking products"
            }
        case "scene.accepted":
            pauseMicrophone()
            automaticRequestPending = false
            state = "Capturing a steady view"
        case "scene.ignored":
            automaticRequestPending = false
        case "scene.configured":
            if let enabled = event["enabled"] as? Bool {
                automaticCaptureEnabled = enabled
                if !enabled, cameraRequestIsAutomatic { cancelCameraRequest() }
            }
        case "response.cancelled":
            if let context = event["context_id"] as? String {
                if endAfterPlaybackContext == context {
                    finishConversation()
                    return
                }
                rememberCancelled(context)
                if activeContext == context { activeContext = nil; nativeSpeech.stop(); audio.flushPlayback() }
            } else { cancelCurrentPlayback() }
        case "camera.request":
            pauseMicrophone()
            guard let requestID = event["request_id"] as? String else { break }
            let automatic = event["trigger"] as? String == "scene.stable"
            guard !automatic || automaticCaptureEnabled else {
                enqueue(["type": "camera.response", "request_id": requestID, "error": "automatic_capture_disabled"])
                break
            }
            cancelCameraRequest()
            let captureGeneration = UUID()
            cameraRequestGeneration = captureGeneration
            cameraRequestIsAutomatic = automatic
            cameraTask = Task { [weak self] in
                guard let self else { return }
                defer {
                    if self.cameraRequestGeneration == captureGeneration {
                        self.cameraRequestGeneration = nil
                        self.cameraRequestIsAutomatic = false
                        self.cameraTask = nil
                    }
                }
                guard self.connectionGeneration == connection, !Task.isCancelled,
                      let provider = self.frameProvider else { return }
                do {
                    guard let photo = try await provider() else { throw AppError.message("No fresh glasses frame is available.") }
                    guard self.connectionGeneration == connection, !Task.isCancelled else { return }
                    let dataURL = try photo.dataURL()
                    // Report the encoded JPEG dimensions, including renderer rounding.
                    guard let encoded = dataURL.split(separator: ",", maxSplits: 1).last,
                          let jpeg = Data(base64Encoded: String(encoded)),
                          let pixels = UIImage(data: jpeg)?.cgImage else {
                        throw AppError.message("The conversation image could not be decoded.")
                    }
                    let formatter = ISO8601DateFormatter()
                    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
                    self.enqueue(["type": "camera.response", "request_id": requestID,
                                  "image_data_url": dataURL,
                                  "width": pixels.width,
                                  "height": pixels.height,
                                  "captured_at": formatter.string(from: photo.capturedAt)])
                } catch {
                    guard self.connectionGeneration == connection, !Task.isCancelled else { return }
                    self.error = "A fresh photo was unavailable: \(error.localizedDescription)"
                    self.enqueue(["type": "camera.response", "request_id": requestID,
                                  "error": "capture_unavailable"])
                }
            }
        case "error":
            error = event["message"] as? String ?? "The voice server could not complete the request."
            if event["code"] as? String == "speech_unavailable" { cancelCurrentPlayback() }
            if event["code"] as? String == "transcription_unavailable" {
                audioInputAvailable = false
                pauseMicrophone()
                state = "Text input only · microphone transcription unavailable"
                error = transcriptionUnavailableMessage
            }
        case "ping": enqueue(["type": "pong"])
        default: break // Includes pong and server-specific snapshot/context events.
        }
        onEvent?(event)
    }

    private func providerLabel(_ provider: String) -> String {
        switch provider {
        case "local": return "local Whisper on backend"
        case "elevenlabs": return "ElevenLabs cloud"
        case "openai": return "OpenAI cloud"
        case "native": return "native iPhone voice"
        case "disabled": return "disabled on backend"
        default: return provider
        }
    }

    private var transcriptionUnavailableMessage: String {
        let action: String
        switch transcriptionProvider {
        case "local":
            action = "Preload the configured local Whisper model on the backend."
        case "elevenlabs", "openai":
            action = "Check the configured \(providerLabel(transcriptionProvider)) API key, model, permissions, quota, and network on the backend."
        case "disabled":
            action = "Enable a transcription provider on the backend to use the microphone."
        default:
            action = "Check the configured speech API key and model, or preload the model if local transcription is selected."
        }
        return "Microphone transcription is unavailable. \(action) Then restart conversation; typed questions still work."
    }

    func sendText(_ text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard isConnected, !clean.isEmpty, clean.count <= 2_000 else { return }
        clientEpoch += 1
        endAfterPlaybackContext = nil
        pauseMicrophone()
        automaticRequestPending = false
        speechInProgress = false
        discardQueuedMicrophone()
        cancelCameraRequest()
        cancelCurrentPlayback()
        enqueue(["type": "input.text", "text": clean])
    }

    func interrupt() {
        clientEpoch += 1
        endAfterPlaybackContext = nil
        pauseMicrophone()
        automaticRequestPending = false
        speechInProgress = false
        discardQueuedMicrophone()
        cancelCurrentPlayback()
        cancelCameraRequest()
        enqueue(["type": "interrupt"])
    }

    func setRecipient(_ id: String) {
        guard recipientID != id else { return }
        recipientID = id
        interrupt()
        clearContextEvidence()
        enqueue(["type": "context.update", "recipient_id": id])
    }

    var canRequestAutomaticCapture: Bool {
        automaticCaptureEnabled && !automaticRequestPending && isIdleForCameraRecovery
    }

    var isIdleForCameraRecovery: Bool {
        isRunning && isConnected && microphoneEnabled && !speechInProgress && state.lowercased() == "listening" &&
            cameraTask == nil && activeContext == nil && microphoneLevel < 0.25
    }

    func configureAutomaticCapture(_ enabled: Bool) {
        automaticCaptureEnabled = enabled
        if !enabled {
            automaticRequestPending = false
            if cameraRequestIsAutomatic { cancelCameraRequest() }
        }
        if isConnected { enqueue(["type": "scene.configure", "enabled": enabled]) }
    }

    @discardableResult
    func requestAutomaticCapture() -> Bool {
        guard canRequestAutomaticCapture else { return false }
        automaticRequestPending = true
        enqueue(["type": "scene.stable"])
        return true
    }

    /// Clear evidence only when leaving its session/context. Merely ending a
    /// conversation keeps the same session's last answer available for review.
    func resetSessionEvidence() {
        stop()
        clearContextEvidence()
    }

    private func clearContextEvidence() {
        transcript = ""
        reply = ""
        error = nil
    }

    private func cancelCameraRequest() {
        let wasCapturing = cameraRequestGeneration != nil
        cameraRequestGeneration = nil
        cameraRequestIsAutomatic = false
        cameraTask?.cancel()
        cameraTask = nil
        // DAT still callbacks have no request ID. The owner invalidates its
        // camera stream synchronously before another request can start.
        if wasCapturing { onCameraCancelled?() }
    }

    private func rememberCancelled(_ context: String) {
        guard !cancelledContexts.contains(context) else { return }
        cancelledContexts.append(context)
        if cancelledContexts.count > 32 { cancelledContexts.removeFirst() }
    }

    private func discardQueuedMicrophone() {
        queue.removeAll { entry in
            if entry.microphone { queueBytes -= entry.bytes; return true }
            return false
        }
        connectedAt = ProcessInfo.processInfo.systemUptime
    }

    private func pauseMicrophone() {
        serverListeningEnabled = false
        applyMicrophoneGate()
    }

    private func applyMicrophoneGate() {
        let enabled = isRunning && isConnected && audioInputAvailable && serverListeningEnabled && activeContext == nil
        guard microphoneEnabled != enabled else {
            if !enabled { discardQueuedMicrophone(); microphoneLevel = 0 }
            return
        }
        microphoneEnabled = enabled
        // Reset the forwarding timestamp and drop both the outgoing backlog and
        // the worker's partial packet/converter history across every boundary.
        discardQueuedMicrophone()
        microphoneLevel = 0
        audio.setMicrophoneEnabled(enabled)
    }

    private func cancelCurrentPlayback() {
        endAfterPlaybackContext = nil
        if let activeContext { rememberCancelled(activeContext) }
        activeContext = nil
        nativeSpeech.stop()
        audio.flushPlayback()
    }

    private func disconnected(_ message: String, connection: UUID) {
        guard isRunning, connectionGeneration == connection else { return }
        if endAfterPlaybackContext != nil {
            // A lost connection must not undo the wearer's spoken goodbye.
            finishConversation()
            error = message
            return
        }
        closeConnection()
        error = message
        guard reconnectAttempt < 3 else {
            stop()
            state = "Disconnected"
            error = "Voice disconnected after three reconnect attempts. Start conversation to retry."
            return
        }
        reconnectAttempt += 1
        state = "Reconnecting voice (\(reconnectAttempt)/3) · audio discarded"
        let delay = UInt64(1 << (reconnectAttempt - 1)) * 1_000_000_000
        let epoch = generation
        reconnectTask = Task { [weak self] in
            do { try await Task.sleep(nanoseconds: delay) } catch { return }
            guard let self, self.generation == epoch, self.isRunning else { return }
            self.connect()
        }
    }

    private func closeConnection() {
        clientEpoch += 1
        connectionGeneration = UUID()
        isConnected = false
        pauseMicrophone()
        automaticRequestPending = false
        speechInProgress = false
        audioInputAvailable = false
        connectedAt = .greatestFiniteMagnitude
        cancelCameraRequest()
        readerTask?.cancel(); readerTask = nil
        writerTask?.cancel(); writerTask = nil
        watchdogTask?.cancel(); watchdogTask = nil
        writerWake?.resume(); writerWake = nil
        queue.removeAll(keepingCapacity: false)
        queueBytes = 0
        sendingSince = nil
        socket?.cancel(with: .goingAway, reason: nil)
        socket = nil
        cancelCurrentPlayback()
    }

    func stop() {
        // WebSocket close itself terminates the backend session immediately; do
        // not wait behind queued microphone packets to send session.stop.
        generation = UUID()
        isRunning = false
        endAfterPlaybackContext = nil
        reconnectTask?.cancel(); reconnectTask = nil
        microphoneTask?.cancel(); microphoneTask = nil
        closeConnection()
        audio.stop()
        request = nil
        frameProvider = nil
        cancelledContexts.removeAll()
        microphoneLevel = 0
        route = "Audio stopped"
        providerDescription = "Speech providers not connected"
        transcriptionProvider = "unavailable"
        voiceName = ""
        state = "Stopped"
    }

    private func finishConversation() {
        // A spoken goodbye ends capture and prevents the reconnect loop only
        // after its matching reply has drained (or speech explicitly failed).
        // Closing the WebSocket terminates the server connection immediately.
        stop()
        state = "Conversation ended"
        onEnded?()
    }
}
