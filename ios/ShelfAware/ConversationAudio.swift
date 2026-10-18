@preconcurrency import AVFoundation
import Foundation

struct ConversationMicrophoneChunk: Sendable {
    let pcm: Data
    let rms: Float
    let uptime: TimeInterval
}

/// Tap-owned converter. No main-actor object or AVAudioPCMBuffer crosses executors.
/// The lock protects stop versus the audio callback; all conversion stays in that callback.
private final class ConversationCaptureWorker: @unchecked Sendable {
    private let lock = NSLock()
    private let converter: AVAudioConverter
    private let outputFormat: AVAudioFormat
    private let continuation: AsyncStream<ConversationMicrophoneChunk>.Continuation
    private var pending = Data()
    private var active = true
    private var inputEnabled = false
    private let chunkBytes = 1_920 // 40 ms at 24 kHz, mono signed16.

    init(inputFormat: AVAudioFormat, continuation: AsyncStream<ConversationMicrophoneChunk>.Continuation) throws {
        guard let output = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inputFormat, to: output) else {
            throw AppError.message("The microphone format could not be converted to 24 kHz mono audio.")
        }
        outputFormat = output
        self.converter = converter
        self.continuation = continuation
    }

    func consume(_ input: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard active, inputEnabled, input.frameLength > 0 else { return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * 24_000 / input.format.sampleRate)) + 64
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { return }
        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, state in
            if supplied { state.pointee = .noDataNow; return nil }
            supplied = true
            state.pointee = .haveData
            return input
        }
        guard status != .error, conversionError == nil, let samples = output.floatChannelData?[0] else {
            active = false
            continuation.finish()
            return
        }
        var sum: Float = 0
        for index in 0..<Int(output.frameLength) {
            let sample = samples[index].isFinite ? max(-1, min(1, samples[index])) : 0
            sum += sample * sample
            var integer = Int16(sample * 32_767).littleEndian
            withUnsafeBytes(of: &integer) { pending.append(contentsOf: $0) }
        }
        let rms = output.frameLength > 0 ? sqrt(sum / Float(output.frameLength)) : 0
        while pending.count >= chunkBytes {
            let bytes = Data(pending.prefix(chunkBytes))
            pending.removeFirst(chunkBytes)
            continuation.yield(ConversationMicrophoneChunk(pcm: bytes, rms: rms, uptime: ProcessInfo.processInfo.systemUptime))
        }
    }

    func setInputEnabled(_ enabled: Bool) {
        lock.lock()
        defer { lock.unlock() }
        guard active, inputEnabled != enabled else { return }
        inputEnabled = enabled
        pending.removeAll(keepingCapacity: false)
        converter.reset()
    }

    func stop() {
        lock.lock()
        active = false
        inputEnabled = false
        pending.removeAll(keepingCapacity: false)
        continuation.finish()
        lock.unlock()
    }
}

/// NotificationCenter permits removal on any thread; keeps deinit out of MainActor.
private final class ConversationAudioObservers: @unchecked Sendable {
    var tokens: [NSObjectProtocol] = [] // Appended only by the owning main-actor initializer.
    deinit { for token in tokens { NotificationCenter.default.removeObserver(token) } }
}

/// One audio engine owns both microphone voice processing and streaming playback.
/// Actual route and echo-cancellation behavior require a test on the paired iPhone.
@MainActor
final class ConversationAudio {
    var onFailure: ((String) -> Void)?
    var onRoute: ((String) -> Void)?
    var onPlayback: ((_ type: String, _ contextID: String, _ playedMilliseconds: Int) -> Void)?

    private let session = AVAudioSession.sharedInstance()
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var worker: ConversationCaptureWorker?
    private var tapInstalled = false
    private let observers = ConversationAudioObservers()
    private var running = false
    private var starting = false
    private var ownsAudioSession = false
    private var startupFailure: String?
    private var captureEpoch = UUID()
    private var requireGlasses = false
    private var playbackFormat: AVAudioFormat?
    private var contextID: String?
    private var playbackEpoch = UUID()
    private var queuedFrames: Int = 0
    private var playedFrames: Int = 0
    private var receivedFrames: Int = 0
    private var providerDone = false
    private var nextSequence = 0
    private let maximumResponseFrames = 24_000 * 30
    private let maximumQueuedFrames = 24_000 * 2 // Server paces PCM; cap device backlog to two seconds.
    var queuedPlaybackFrames: Int { queuedFrames }

    init() {
        observers.tokens.append(NotificationCenter.default.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] notification in
            let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            guard raw == AVAudioSession.InterruptionType.began.rawValue else { return }
            // These observers deliver on .main. Capture ownership at receipt,
            // before the task can cross a stop followed by a replacement start.
            let epoch = MainActor.assumeIsolated { () -> UUID? in
                guard let self, self.starting || self.running else { return nil }
                return self.captureEpoch
            }
            guard let epoch else { return }
            Task { @MainActor in
                guard let self, self.captureEpoch == epoch else { return }
                if self.starting {
                    self.startupFailure = "Audio was interrupted while connecting. Try again when the interruption ends."
                } else {
                    self.fail("Audio was interrupted. Start conversation again when ready.")
                }
            }
        })
        observers.tokens.append(NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] _ in
            let epoch = MainActor.assumeIsolated { () -> UUID? in
                guard let self, self.starting || self.running else { return nil }
                return self.captureEpoch
            }
            guard let epoch else { return }
            Task { @MainActor in
                guard let self, self.captureEpoch == epoch else { return }
                // A preferred HFP input and voice-processing startup legitimately
                // change the route. start() verifies that they settle before any
                // microphone forwarding or network conversation can begin.
                if self.starting {
                    self.onRoute?("Connecting glasses audio · \(self.routeSummary)")
                    return
                }
                guard self.running else { return }
                self.publishRoute()
                if self.requireGlasses && !self.hasHFPRoute {
                    self.fail("Glasses microphone or audio output disconnected. Conversation stopped without switching to the phone speaker.")
                }
            }
        })
        observers.tokens.append(NotificationCenter.default.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            let epoch = MainActor.assumeIsolated { () -> UUID? in
                guard let self, self.starting || self.running else { return nil }
                return self.captureEpoch
            }
            guard let epoch else { return }
            Task { @MainActor in
                guard let self, self.captureEpoch == epoch, !self.starting, self.running, !self.engine.isRunning else { return }
                self.fail("The audio device format changed. Start conversation again to rebuild the audio route.")
            }
        })
    }

    private var hasHFPRoute: Bool {
        session.currentRoute.inputs.contains { $0.portType == .bluetoothHFP } &&
        session.currentRoute.outputs.contains { $0.portType == .bluetoothHFP }
    }

    private var routeSummary: String {
        let inputs = session.currentRoute.inputs.map(\.portName).joined(separator: ", ")
        let outputs = session.currentRoute.outputs.map(\.portName).joined(separator: ", ")
        return "In: \(inputs.isEmpty ? "none" : inputs) · Out: \(outputs.isEmpty ? "none" : outputs)"
    }

    private func publishRoute() {
        onRoute?("\(routeSummary) · Voice processing on")
    }

    private func checkStartup(_ epoch: UUID) throws {
        try Task.checkCancellation()
        guard captureEpoch == epoch else { throw CancellationError() }
        if let startupFailure { throw AppError.message(startupFailure) }
    }

    private var routeFormatSignature: String? {
        let format = engine.inputNode.outputFormat(forBus: 0)
        guard format.sampleRate.isFinite, format.sampleRate > 0, format.channelCount > 0,
              !session.currentRoute.inputs.isEmpty, !session.currentRoute.outputs.isEmpty,
              !requireGlasses || hasHFPRoute else { return nil }
        let ports = (session.currentRoute.inputs + session.currentRoute.outputs)
            .map { "\($0.portType.rawValue):\($0.uid)" }.joined(separator: "|")
        return "\(ports)|\(format.sampleRate)|\(format.channelCount)|\(format.commonFormat.rawValue)|\(format.isInterleaved)"
    }

    private func waitForStableRoute(epoch: UUID, deadline: TimeInterval) async throws -> String {
        var previous: String?
        var stableSince = ProcessInfo.processInfo.systemUptime
        while ProcessInfo.processInfo.systemUptime < deadline {
            try checkStartup(epoch)
            let signature = routeFormatSignature
            let now = ProcessInfo.processInfo.systemUptime
            if signature == nil || signature != previous { stableSince = now }
            previous = signature
            if let signature, now - stableSince >= 0.35 { return signature }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        throw AppError.message("The glasses audio route did not settle. Check the Bluetooth connection and try again. \(routeSummary)")
    }

    private func clearAudioGraph() {
        // Keep the AVAudioSession active across a bounded startup rebuild, so
        // rebuilding an old-format tap does not initiate another HFP switch.
        engine.stop()
        if tapInstalled { engine.inputNode.removeTap(onBus: 0); tapInstalled = false }
        worker?.stop()
        worker = nil
        if player.engine != nil { engine.disconnectNodeOutput(player) }
        playbackFormat = nil
    }

    func start(requireGlasses: Bool) async throws -> AsyncStream<ConversationMicrophoneChunk> {
        stop()
        let epoch = captureEpoch
        var startedSuccessfully = false
        defer {
            if !startedSuccessfully, captureEpoch == epoch { stop() }
        }
        let allowed = await withCheckedContinuation { continuation in
            session.requestRecordPermission { continuation.resume(returning: $0) }
        }
        guard allowed else { throw AppError.message("Allow Microphone access in iPhone Settings to use conversation.") }
        guard captureEpoch == epoch else { throw CancellationError() }
        try Task.checkCancellation()
        self.requireGlasses = requireGlasses
        starting = true
        startupFailure = nil
        ownsAudioSession = true
        onRoute?(requireGlasses ? "Connecting glasses microphone and audio" : "Connecting phone microphone and audio")
        let startupDeadline = ProcessInfo.processInfo.systemUptime + 6
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetooth])
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true)
        if requireGlasses {
            var microphone = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP })
            while microphone == nil, ProcessInfo.processInfo.systemUptime < startupDeadline {
                try checkStartup(epoch)
                try await Task.sleep(nanoseconds: 100_000_000)
                microphone = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP })
            }
            guard let microphone else { throw AppError.message("No Bluetooth HFP microphone is available. Pair the glasses for audio first. \(routeSummary)") }
            try session.setPreferredInput(microphone)
        } else {
            if let microphone = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
                try session.setPreferredInput(microphone)
            }
            try session.overrideOutputAudioPort(.speaker)
        }
        try checkStartup(epoch)
        let input = engine.inputNode
        try input.setVoiceProcessingEnabled(true)
        guard input.isVoiceProcessingEnabled else {
            throw AppError.message("Voice processing could not be enabled on this audio route.")
        }
        // Keep HFP and the playback engine active, but capture no input before
        // the server explicitly enables the next listening turn.
        input.isVoiceProcessingInputMuted = true
        for attempt in 0..<2 {
            try checkStartup(epoch)
            clearAudioGraph()
            let settledRoute = try await waitForStableRoute(epoch: epoch, deadline: startupDeadline)
            let format = input.outputFormat(forBus: 0)
            guard let playbackFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false) else {
                throw AppError.message("The selected audio route has no usable microphone format.")
            }
            self.playbackFormat = playbackFormat
            if player.engine == nil { engine.attach(player) }
            engine.connect(player, to: engine.mainMixerNode, format: playbackFormat)
            let (stream, continuation) = AsyncStream<ConversationMicrophoneChunk>.makeStream(bufferingPolicy: .bufferingNewest(8))
            let worker = try ConversationCaptureWorker(inputFormat: format, continuation: continuation)
            self.worker = worker
            // No input leaves this tap until the backend enables listening.
            input.installTap(onBus: 0, bufferSize: 960, format: format) { @Sendable buffer, _ in worker.consume(buffer) }
            tapInstalled = true
            do {
                engine.prepare()
                try engine.start()
                // Let queued route/configuration notifications run while startup
                // owns the transition. Apple stops the engine on hardware format
                // changes; one rebuild uses the newly settled format.
                for _ in 0..<5 {
                    try await Task.sleep(nanoseconds: 100_000_000)
                    try checkStartup(epoch)
                    guard ProcessInfo.processInfo.systemUptime < startupDeadline,
                          engine.isRunning, routeFormatSignature == settledRoute else {
                        throw AppError.message("The audio input changed while connecting the glasses.")
                    }
                }
                starting = false
                running = true
                startedSuccessfully = true
                publishRoute()
                return stream
            } catch {
                try checkStartup(epoch)
                guard attempt == 0, ProcessInfo.processInfo.systemUptime < startupDeadline else { throw error }
                onRoute?("Audio route changed · settling glasses connection")
            }
        }
        throw AppError.message("The glasses audio route could not be started.")
    }

    func setMicrophoneEnabled(_ enabled: Bool) {
        guard running else { return }
        // The worker gate precedes PCM conversion/queueing. Muting the voice
        // processing input does not pause the shared output engine or HFP route.
        worker?.setInputEnabled(enabled)
        engine.inputNode.isVoiceProcessingInputMuted = !enabled
    }

    func beginResponse(contextID: String) {
        guard self.contextID != contextID else { return }
        flushPlayback()
        self.contextID = contextID
    }

    func append(_ pcm: Data, contextID: String, sequence: Int, sampleRate: Int) throws {
        guard running, engine.isRunning else { throw AppError.message("The conversation audio engine is stopped.") }
        guard sampleRate == 24_000, !pcm.isEmpty, pcm.count % 2 == 0, pcm.count <= 48_000 else {
            throw AppError.message("Conversation returned invalid PCM audio.")
        }
        if self.contextID == nil { beginResponse(contextID: contextID) }
        guard self.contextID == contextID else { return }
        guard sequence == nextSequence else { throw AppError.message("Conversation audio arrived out of order. Please ask again.") }
        let frameCount = pcm.count / 2
        guard receivedFrames + frameCount <= maximumResponseFrames, queuedFrames + frameCount <= maximumQueuedFrames,
              let format = playbackFormat,
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)),
              let destination = buffer.floatChannelData?[0] else {
            throw AppError.message("Conversation audio exceeded its bounded playback limit.")
        }
        buffer.frameLength = AVAudioFrameCount(frameCount)
        pcm.withUnsafeBytes { (bytes: UnsafeRawBufferPointer) in
            for index in 0..<frameCount {
                let low = UInt16(bytes[index * 2])
                let high = UInt16(bytes[index * 2 + 1]) << 8
                destination[index] = Float(Int16(bitPattern: low | high)) / 32_768
            }
        }
        nextSequence += 1
        receivedFrames += frameCount
        queuedFrames += frameCount
        let epoch = playbackEpoch
        player.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
            Task { @MainActor in
                guard let self, self.playbackEpoch == epoch else { return }
                self.queuedFrames = max(0, self.queuedFrames - frameCount)
                self.playedFrames += frameCount
                self.finishIfDrained()
            }
        }
        if !player.isPlaying {
            player.play()
            // This reports the engine's playback start, not proof that a person
            // heard it. Device output latency precedes audible sound.
            onPlayback?("playback.started", contextID, playedFrames * 1_000 / 24_000)
        }
    }

    func finishResponse(contextID: String) {
        guard self.contextID == contextID else { return }
        providerDone = true
        finishIfDrained()
    }

    private func finishIfDrained() {
        guard providerDone, queuedFrames == 0, let contextID else { return }
        let milliseconds = playedFrames * 1_000 / 24_000
        resetPlayback()
        onPlayback?("playback.finished", contextID, milliseconds)
    }

    func flushPlayback() {
        let interruptedContext = receivedFrames > 0 ? contextID : nil
        let milliseconds = playedFrames * 1_000 / 24_000
        // Reset before calling outward: a full network queue may synchronously
        // stop the connection and reenter flushPlayback from the callback.
        resetPlayback()
        if let interruptedContext { onPlayback?("playback.interrupted", interruptedContext, milliseconds) }
    }

    private func resetPlayback() {
        playbackEpoch = UUID()
        player.stop()
        contextID = nil
        queuedFrames = 0
        playedFrames = 0
        receivedFrames = 0
        providerDone = false
        nextSequence = 0
    }

    private func fail(_ message: String) {
        guard running else { return }
        stop()
        onFailure?(message)
    }

    func stop() {
        let shouldDeactivate = ownsAudioSession
        ownsAudioSession = false
        captureEpoch = UUID()
        starting = false
        startupFailure = nil
        running = false
        flushPlayback()
        clearAudioGraph()
        if shouldDeactivate { try? session.setActive(false, options: .notifyOthersOnDeactivation) }
    }
}
