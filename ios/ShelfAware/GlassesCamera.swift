import Combine
import Foundation
import UIKit
#if HAS_META_DAT
@preconcurrency import MWDATCore
@preconcurrency import MWDATCamera
#endif

struct CapturedPhoto {
    let image: UIImage
    let capturedAt: Date
    let mode: CaptureMode

    /// Bound upload size; preserve the source aspect ratio and physical label pixels.
    func dataURL() throws -> String {
        let maximumBytes = 4_000_000
        // UIImage.size is in points; use pixels so a scaled image is not needlessly
        // reduced. Drawing also normalizes the image's orientation before upload.
        let pixelSize = CGSize(width: image.size.width * image.scale, height: image.size.height * image.scale)
        guard pixelSize.width.isFinite, pixelSize.height.isFinite,
              pixelSize.width > 0, pixelSize.height > 0 else {
            throw AppError.message("The captured photo has invalid dimensions.")
        }
        var scale = min(1, 2560 / max(pixelSize.width, pixelSize.height))
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        // Most label photos fit on the first pass. Bound both retries and bytes;
        // never send an oversized payload or substitute a lower-quality frame.
        for _ in 0..<5 {
            let size = CGSize(width: max(1, floor(pixelSize.width * scale)),
                              height: max(1, floor(pixelSize.height * scale)))
            let rendered = UIGraphicsImageRenderer(size: size, format: format).image { _ in
                image.draw(in: CGRect(origin: .zero, size: size))
            }
            for quality: CGFloat in [0.94, 0.88, 0.82] {
                guard let data = rendered.jpegData(compressionQuality: quality) else { continue }
                if data.count <= maximumBytes {
                    return "data:image/jpeg;base64," + data.base64EncodedString()
                }
            }
            scale *= 0.8
        }
        throw AppError.message("The captured photo could not be encoded within the upload limit. Try another capture.")
    }
}

/// An immutable SDK-produced image transferred to the main actor. Only this wrapper
/// crosses executors; the UIImage is never mutated after decoding.
private struct ReceivedFrame: @unchecked Sendable {
    let image: UIImage
    let receivedAt: Date
    let uptime: TimeInterval
}

/// DAT is isolated here. No experimental microphone, wake word, or display APIs.
/// API lifecycle based on Meta's 1.0.1 CameraAccess sample; see THIRD_PARTY_NOTICES.md.
@MainActor
final class GlassesCamera: ObservableObject {
    @Published private(set) var state = "Not connected"
    @Published private(set) var registration = "Not registered"
    @Published private(set) var isStreaming = false
    @Published private(set) var preview: UIImage?
    @Published private(set) var frameReceivedAt: Date?
    @Published private(set) var error: String?
    @Published private(set) var isOpeningGlassesUpdate = false
    private var photoContinuation: CheckedContinuation<CapturedPhoto, Error>?
    private var photoTimeout: Task<Void, Never>?
    private var epoch = UUID()
    private var frameUptime: TimeInterval?
    static let maximumFrameAge: TimeInterval = 0.75

    /// Reuse just one decoded frame. Never identify a product from a stale buffer.
    func freshFrame() -> CapturedPhoto? {
        guard isStreaming, let image = preview, let receivedAt = frameReceivedAt, let frameUptime else { return nil }
        let age = ProcessInfo.processInfo.systemUptime - frameUptime
        guard age >= 0, age <= Self.maximumFrameAge else { return nil }
        return CapturedPhoto(image: image, capturedAt: receivedAt, mode: .liveGlasses)
    }

    #if HAS_META_DAT
    static let sdkIncluded = true
    private var wearables: WearablesInterface?
    private var selector: AutoDeviceSelector?
    private var deviceSession: DeviceSession?
    private var camera: MWDATCamera.Camera?
    private var tokens: [AnyListenerToken] = []
    private var registrationTask: Task<Void, Never>?
    private var frameConsumer: Task<Void, Never>?
    private var frameContinuation: AsyncStream<ReceivedFrame>.Continuation?
    private var streamPaused = false

    init() {
        do {
            try Wearables.configure()
            let sdk = Wearables.shared
            wearables = sdk
            selector = AutoDeviceSelector(wearables: sdk)
            registration = String(describing: sdk.registrationState)
            registrationTask = Task { [weak self] in
                for await value in sdk.registrationStateStream() {
                    guard !Task.isCancelled else { return }
                    self?.registration = String(describing: value)
                }
            }
        } catch {
            self.error = error.localizedDescription
            state = "SDK configuration failed"
        }
    }

    func register() async {
        guard let wearables else { return }
        do { try await wearables.startRegistration() }
        catch { self.error = error.localizedDescription }
    }

    func openGlassesAppUpdate() async {
        guard !isOpeningGlassesUpdate, let wearables else { return }
        isOpeningGlassesUpdate = true
        defer { isOpeningGlassesUpdate = false }
        // Navigation only: opening Meta AI does not prove that an update completed.
        do { try await wearables.openDATGlassesAppUpdate() }
        catch { self.error = error.localizedDescription }
    }

    func handleURL(_ url: URL) async {
        let configuration = Bundle.main.object(forInfoDictionaryKey: "MWDAT") as? [String: Any]
        let callback = configuration?["AppLinkURLScheme"] as? String ?? "shelfaware://"
        guard let expectedScheme = URL(string: callback)?.scheme,
              url.scheme == expectedScheme,
              URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems?.contains(where: { $0.name == "metaWearablesAction" }) == true,
              let wearables else { return }
        do { try await wearables.handleUrl(url) }
        catch { self.error = error.localizedDescription }
    }

    func start() async throws {
        if isStreaming { return }
        guard let wearables, let selector else { throw AppError.message(error ?? "The Meta SDK is unavailable.") }
        // DAT explicitly says to retain a paused session, not restart it while paused.
        if let existing = deviceSession, existing.state == .paused || streamPaused {
            state = "Glasses paused · waiting for resume"
            let resumeDeadline = ProcessInfo.processInfo.systemUptime + 6
            while existing.state == .paused || streamPaused {
                try Task.checkCancellation()
                guard ProcessInfo.processInfo.systemUptime < resumeDeadline else {
                    throw AppError.message("The glasses are paused. Resume them and tap Look & Ask again, or press Stop before reconnecting.")
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            if isStreaming { return }
        }
        error = nil
        // Request before creating the camera session: the Meta AI redirect backgrounds this app.
        state = "Checking camera permission"
        if try await wearables.checkPermissionStatus(.camera) != .granted {
            state = "Approve camera access in Meta AI"
            guard try await wearables.requestPermission(.camera) == .granted else {
                throw AppError.message("Meta glasses camera permission was denied.")
            }
        }
        try Task.checkCancellation()
        stop()
        var startedSuccessfully = false
        defer { if !startedSuccessfully { stop() } }
        let requestEpoch = epoch
        let session = try wearables.createSession(deviceSelector: selector)
        deviceSession = session
        tokens.append(session.statePublisher.listen { [weak self] value in
            Task { @MainActor in
                guard let self, self.epoch == requestEpoch else { return }
                self.state = value.description
                if value == .stopped {
                    self.isStreaming = false
                    self.streamPaused = false
                    self.clearPreview()
                    self.finishPhoto(.failure(AppError.message("The glasses session stopped. Reconnect and try again.")))
                }
            }
        })
        tokens.append(session.errorPublisher.listen { [weak self] value in
            let message = value.localizedDescription
            Task { @MainActor in
                guard let self, self.epoch == requestEpoch else { return }
                self.error = message
                self.finishPhoto(.failure(AppError.message(message)))
            }
        })
        state = "Connecting glasses"
        try session.start()
        let deadline = Date().addingTimeInterval(15)
        while session.state != .started {
            try Task.checkCancellation()
            guard epoch == requestEpoch else { throw CancellationError() }
            if let error { throw AppError.message(error) }
            guard Date() < deadline else {
                stop()
                throw AppError.message("The glasses did not connect within 15 seconds. Check pairing, wear the glasses, and retry.")
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        let config = StreamConfiguration(videoCodec: .raw, resolution: .high, frameRate: 7)
        guard let newCamera = try session.addCamera(config: config) else {
            stop()
            throw AppError.message("The SDK could not add the glasses camera.")
        }
        camera = newCamera
        let stream = newCamera.stream
        tokens.append(stream.statePublisher.listen { [weak self] value in
            Task { @MainActor in
                guard let self, self.epoch == requestEpoch else { return }
                self.isStreaming = value == .streaming
                self.streamPaused = value == .paused
                self.state = String(describing: value)
                if value == .stopped || value == .paused {
                    self.clearPreview()
                    self.finishPhoto(.failure(AppError.message("Camera streaming stopped or paused. Resume the glasses connection.")))
                }
            }
        })
        // Buffer the newest frame only. A busy main thread must not accumulate old
        // frame tasks and then give those frames artificially fresh timestamps.
        let (frames, continuation) = AsyncStream<ReceivedFrame>.makeStream(bufferingPolicy: .bufferingNewest(1))
        frameContinuation = continuation
        frameConsumer = Task { [weak self] in
            for await frame in frames {
                guard let self, self.epoch == requestEpoch, !Task.isCancelled else { return }
                guard self.isStreaming else { continue }
                self.preview = frame.image
                self.frameReceivedAt = frame.receivedAt
                self.frameUptime = frame.uptime
            }
        }
        tokens.append(stream.videoFramePublisher.listen { frame in
            let receivedAt = Date()
            let uptime = ProcessInfo.processInfo.systemUptime
            guard let image = frame.makeUIImage() else { return }
            continuation.yield(ReceivedFrame(image: image, receivedAt: receivedAt, uptime: uptime))
        })
        tokens.append(stream.photoDataPublisher.listen { [weak self] photo in
            let bytes = photo.data
            Task { @MainActor in
                guard let self, self.epoch == requestEpoch else { return }
                guard let image = UIImage(data: bytes) else {
                    self.finishPhoto(.failure(AppError.message("The glasses photo could not be decoded. Try another capture.")))
                    return
                }
                self.finishPhoto(.success(CapturedPhoto(image: image, capturedAt: Date(), mode: .liveGlasses)))
            }
        })
        tokens.append(stream.errorPublisher.listen { [weak self] value in
            let message = value.localizedDescription
            Task { @MainActor in
                guard let self, self.epoch == requestEpoch else { return }
                self.error = message
                self.finishPhoto(.failure(AppError.message(message)))
            }
        })
        stream.start()
        let streamDeadline = Date().addingTimeInterval(15)
        while !isStreaming {
            try Task.checkCancellation()
            guard epoch == requestEpoch else { throw CancellationError() }
            if let error { throw AppError.message(error) }
            guard Date() < streamDeadline else {
                stop()
                throw AppError.message("No live glasses stream arrived within 15 seconds. Check the glasses connection.")
            }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        startedSuccessfully = true
    }

    func capture() async throws -> CapturedPhoto {
        guard isStreaming, let stream = camera?.stream else { throw AppError.message("Start the live glasses camera first.") }
        guard photoContinuation == nil else { throw AppError.message("A photo is already being captured.") }
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { continuation in
                photoContinuation = continuation
                guard stream.capturePhoto(format: .jpeg) else {
                    finishPhoto(.failure(AppError.message("The glasses rejected this capture. Wait a moment and retry.")))
                    return
                }
                photoTimeout = Task { [weak self] in
                    do { try await Task.sleep(nanoseconds: 8_000_000_000) } catch { return }
                    self?.finishPhoto(.failure(AppError.message("The glasses photo timed out. Please capture again.")))
                    // A late SDK photo has no request ID. Recreate the session so it
                    // cannot accidentally satisfy a subsequent capture continuation.
                    self?.stop()
                }
            }
        }, onCancel: { [weak self] in
            Task { @MainActor in self?.finishPhoto(.failure(CancellationError())) }
        })
    }

    func stop() {
        epoch = UUID()
        finishPhoto(.failure(CancellationError()))
        camera?.stop()
        deviceSession?.stop()
        camera = nil
        deviceSession = nil
        tokens.removeAll()
        frameContinuation?.finish()
        frameContinuation = nil
        frameConsumer?.cancel()
        frameConsumer = nil
        streamPaused = false
        isStreaming = false
        clearPreview()
        state = "Stopped"
    }
    #else
    static let sdkIncluded = false
    init() { state = "Phone-only build — glasses SDK excluded" }
    func register() async { error = "Install the ShelfAware target to test glasses. This build is phone-only." }
    func openGlassesAppUpdate() async { error = "Glasses updates are unavailable in this phone-only build." }
    func handleURL(_ url: URL) async {}
    func start() async throws { throw AppError.message("This is the explicitly labeled phone-only fallback build.") }
    func capture() async throws -> CapturedPhoto { throw AppError.message("Glasses capture is unavailable in this build.") }
    func stop() { isStreaming = false; clearPreview() }
    #endif

    private func clearPreview() {
        preview = nil
        frameReceivedAt = nil
        frameUptime = nil
    }

    private func finishPhoto(_ result: Result<CapturedPhoto, Error>) {
        photoTimeout?.cancel()
        photoTimeout = nil
        let pending = photoContinuation
        photoContinuation = nil
        pending?.resume(with: result)
    }
}
