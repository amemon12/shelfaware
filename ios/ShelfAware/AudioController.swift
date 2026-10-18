import AVFoundation
import Combine
import Speech

/// One owner for the iOS audio session. Push-to-talk is deliberately half duplex.
/// DAT supplies images; Bluetooth HFP supplies optional voice input, independently.
@MainActor
final class AudioController: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var isListening = false
    @Published private(set) var isSpeaking = false
    @Published private(set) var route = "Audio idle"
    @Published private(set) var transcript = ""
    @Published private(set) var error: String?
    var onInterrupted: (() -> Void)?

    private let session = AVAudioSession.sharedInstance()
    private let synthesizer = AVSpeechSynthesizer()
    private let engine = AVAudioEngine()
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?
    private var listeningTimeout: Task<Void, Never>?
    private var observers: [NSObjectProtocol] = []
    private var inputTapInstalled = false
    private var requiresGlassesRoute = false
    private var listenGeneration = UUID()
    private var transcriptHandler: ((String) -> Void)?
    private var currentUtterance: AVSpeechUtterance?
    private var preparingRoute = false
    private var audioOwnershipEpoch = UUID()

    override init() {
        super.init()
        synthesizer.delegate = self
        observers.append(NotificationCenter.default.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] notification in
            let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            guard raw == AVAudioSession.InterruptionType.began.rawValue else { return }
            let epoch = MainActor.assumeIsolated { () -> UUID? in
                guard let self, self.isListening || self.isSpeaking || self.preparingRoute else { return nil }
                return self.audioOwnershipEpoch
            }
            guard let epoch else { return }
            Task { @MainActor in
                guard let self, self.audioOwnershipEpoch == epoch,
                      self.isListening || self.isSpeaking || self.preparingRoute else { return }
                self.stop()
                self.error = "Audio was interrupted. Tap Ask or Speak when ready."
                self.onInterrupted?()
            }
        })
        observers.append(NotificationCenter.default.addObserver(forName: AVAudioSession.routeChangeNotification, object: nil, queue: .main) { [weak self] _ in
            let epoch = MainActor.assumeIsolated { () -> UUID? in
                guard let self, self.isListening || self.isSpeaking || self.preparingRoute else { return nil }
                return self.audioOwnershipEpoch
            }
            guard let epoch else { return }
            Task { @MainActor in
                guard let self, self.audioOwnershipEpoch == epoch else { return }
                self.updateRoute()
                if self.requiresGlassesRoute && ((self.isListening && !self.hasHFPInput) || ((self.isListening || self.isSpeaking) && !self.hasBluetoothOutput)) {
                    self.stop()
                    self.error = "The glasses audio route disconnected. Speech stopped; no automatic phone-speaker fallback."
                    self.onInterrupted?()
                }
            }
        })
    }

    private var hasBluetoothOutput: Bool {
        session.currentRoute.outputs.contains { [.bluetoothHFP, .bluetoothA2DP, .bluetoothLE].contains($0.portType) }
    }

    private var hasHFPInput: Bool {
        session.currentRoute.inputs.contains { $0.portType == .bluetoothHFP }
    }

    private func updateRoute() {
        let input = session.currentRoute.inputs.map(\.portName).joined(separator: ", ")
        let output = session.currentRoute.outputs.map(\.portName).joined(separator: ", ")
        route = "In: \(input.isEmpty ? "none" : input) · Out: \(output.isEmpty ? "none" : output)"
    }

    private func prepareRoute(requireGlasses: Bool) async throws {
        preparingRoute = true
        defer { preparingRoute = false }
        requiresGlassesRoute = requireGlasses
        // allowBluetooth is the iOS 17-compatible spelling for HFP input/output.
        try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.allowBluetooth])
        try session.setActive(true)
        if requireGlasses {
            guard let input = session.availableInputs?.first(where: { $0.portType == .bluetoothHFP }) else {
                throw AppError.message("No Bluetooth HFP input is available. Pair the glasses for audio, then retry. Typed questions remain available.")
            }
            try session.setPreferredInput(input)
            for _ in 0..<10 {
                if hasBluetoothOutput && session.currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }) { break }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            guard hasBluetoothOutput, session.currentRoute.inputs.contains(where: { $0.portType == .bluetoothHFP }) else {
                throw AppError.message("iOS has not routed audio through Bluetooth HFP. No phone-speaker fallback was used.")
            }
        } else {
            if let input = session.availableInputs?.first(where: { $0.portType == .builtInMic }) {
                try session.setPreferredInput(input)
            }
            try session.overrideOutputAudioPort(.speaker)
        }
        updateRoute()
    }

    func speak(_ text: String, requireGlasses: Bool) async throws {
        stop()
        error = nil
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        try await prepareRoute(requireGlasses: requireGlasses)
        try Task.checkCancellation()
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: Locale.current.identifier)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 0.93
        utterance.volume = 0.72
        currentUtterance = utterance
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    func startListening(requireGlasses: Bool, onTranscript: @escaping (String) -> Void) async throws {
        stop()
        error = nil
        let authorized = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0 == .authorized) }
        }
        guard authorized else { throw AppError.message("Speech recognition is not authorized. Type your question, or allow Speech Recognition in Settings.") }
        let microphoneAllowed = await withCheckedContinuation { continuation in
            session.requestRecordPermission { continuation.resume(returning: $0) }
        }
        guard microphoneAllowed else { throw AppError.message("Microphone access is not authorized. Typed questions still work.") }
        guard let recognizer = SFSpeechRecognizer(locale: Locale.current), recognizer.isAvailable else {
            throw AppError.message("Speech recognition is currently unavailable. Type your question.")
        }
        try Task.checkCancellation()
        try await prepareRoute(requireGlasses: requireGlasses)
        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        if recognizer.supportsOnDeviceRecognition { request.requiresOnDeviceRecognition = true }
        recognitionRequest = request
        transcript = ""
        transcriptHandler = onTranscript
        let generation = UUID()
        listenGeneration = generation
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw AppError.message("The selected microphone did not provide an audio format.") }
        input.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in request.append(buffer) }
        inputTapInstalled = true
        recognitionTask = recognizer.recognitionTask(with: request) { [weak self] result, failure in
            let text = result?.bestTranscription.formattedString
            let final = result?.isFinal ?? false
            let message = failure?.localizedDescription
            Task { @MainActor in
                guard let self, self.listenGeneration == generation else { return }
                if let text {
                    self.transcript = text
                    self.transcriptHandler?(text)
                }
                if final || message != nil {
                    self.stopListening()
                    if self.transcript.isEmpty, let message { self.error = message }
                }
            }
        }
        do {
            engine.prepare()
            try engine.start()
            isListening = true
            listeningTimeout = Task { [weak self] in
                do { try await Task.sleep(nanoseconds: 12_000_000_000) } catch { return }
                self?.stopListening()
            }
        } catch {
            stopListening()
            throw error
        }
    }

    func stopListening() {
        listenGeneration = UUID()
        listeningTimeout?.cancel()
        listeningTimeout = nil
        engine.stop()
        if inputTapInstalled { engine.inputNode.removeTap(onBus: 0); inputTapInstalled = false }
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()
        recognitionTask = nil
        recognitionRequest = nil
        transcriptHandler = nil
        isListening = false
    }

    func stop() {
        audioOwnershipEpoch = UUID()
        preparingRoute = false
        stopListening()
        currentUtterance = nil
        synthesizer.stopSpeaking(at: .immediate)
        isSpeaking = false
    }

    func deactivate() {
        stop()
        try? session.setActive(false, options: .notifyOthersOnDeactivation)
        route = "Audio idle"
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        finishSpeech(ObjectIdentifier(utterance))
    }
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        finishSpeech(ObjectIdentifier(utterance))
    }

    private nonisolated func finishSpeech(_ identifier: ObjectIdentifier) {
        Task { @MainActor [weak self] in
            guard let self, self.currentUtterance.map(ObjectIdentifier.init) == identifier else { return }
            self.currentUtterance = nil
            self.isSpeaking = false
        }
    }
}
