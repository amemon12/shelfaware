@preconcurrency import AVFoundation
import Foundation

/// Retain only for identity comparison on MainActor; no utterance properties are
/// read or mutated on the delegate callback thread.
private struct NativeUtteranceIdentity: @unchecked Sendable {
    let utterance: AVSpeechUtterance
}

/// Converts each callback-owned synthesis buffer synchronously. Only owned Data
/// enters the bounded stream; AVAudioPCMBuffer never crosses an executor.
private final class NativeSpeechPCMProducer: @unchecked Sendable {
    private let lock = NSLock()
    private let continuation: AsyncThrowingStream<Data, Error>.Continuation
    private var converter: AVAudioConverter?
    private var outputFormat: AVAudioFormat?
    private var pending = Data()
    private var totalBytes = 0
    private var active = true
    private let maximumBytes = 24_000 * 2 * 30
    private let chunkBytes = 1_920

    init(continuation: AsyncThrowingStream<Data, Error>.Continuation) { self.continuation = continuation }

    func consume(_ raw: AVAudioBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard active else { return }
        guard let input = raw as? AVAudioPCMBuffer else { fail("Native speech did not return PCM audio."); return }
        if input.frameLength == 0 { finishLocked(); return }
        guard input.format.sampleRate > 0, input.format.channelCount > 0 else { fail("Native speech returned an invalid format."); return }
        if converter == nil {
            outputFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 24_000, channels: 1, interleaved: false)
            if let outputFormat { converter = AVAudioConverter(from: input.format, to: outputFormat) }
        }
        guard let converter, let outputFormat else { fail("Native speech could not be converted."); return }
        let capacity = AVAudioFrameCount(ceil(Double(input.frameLength) * 24_000 / input.format.sampleRate)) + 64
        guard capacity <= AVAudioFrameCount(maximumBytes / 2 + 64) else { fail("Native speech exceeded its conversion buffer limit."); return }
        guard let output = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: capacity) else { fail("Native speech buffer allocation failed."); return }
        var supplied = false
        var conversionError: NSError?
        let status = converter.convert(to: output, error: &conversionError) { _, state in
            if supplied { state.pointee = .noDataNow; return nil }
            supplied = true
            state.pointee = .haveData
            return input
        }
        guard status != .error, conversionError == nil else {
            fail("Native speech conversion failed."); return
        }
        appendConverted(output)
    }

    private func appendConverted(_ output: AVAudioPCMBuffer) {
        guard active, output.frameLength > 0 else { return }
        guard let samples = output.floatChannelData?[0] else { fail("Native speech conversion returned no samples."); return }
        totalBytes += Int(output.frameLength) * 2
        guard totalBytes <= maximumBytes else { fail("Native speech exceeded its 30-second limit."); return }
        for index in 0..<Int(output.frameLength) {
            let sample = samples[index].isFinite ? max(-1, min(1, samples[index])) : 0
            var value = Int16(sample * 32_767).littleEndian
            withUnsafeBytes(of: &value) { pending.append(contentsOf: $0) }
        }
        while pending.count >= chunkBytes && active {
            let data = Data(pending.prefix(chunkBytes))
            pending.removeFirst(chunkBytes)
            emit(data)
        }
    }

    private func emit(_ data: Data) {
        switch continuation.yield(data) {
        case .enqueued: break
        case .dropped: fail("Native speech queue exceeded its bounded capacity.")
        case .terminated: active = false
        @unknown default: fail("Native speech queue could not accept audio.")
        }
    }

    private func fail(_ message: String) {
        active = false
        pending.removeAll(keepingCapacity: false)
        continuation.finish(throwing: AppError.message(message))
    }

    private func finishLocked() {
        guard active else { return }
        // A resampler may retain filter-tail samples after its last input buffer.
        // Drain explicitly so the final consonant is not cut at end-of-stream.
        if let converter, let outputFormat {
            for _ in 0..<4 {
                guard let tail = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: 1_024) else {
                    fail("Native speech final buffer allocation failed."); return
                }
                var error: NSError?
                let status = converter.convert(to: tail, error: &error) { _, state in
                    state.pointee = .endOfStream
                    return nil
                }
                guard status != .error, error == nil else { fail("Native speech final conversion failed."); return }
                appendConverted(tail)
                guard active else { return }
                if status == .endOfStream || status == .inputRanDry || tail.frameLength == 0 { break }
            }
        }
        if !pending.isEmpty { emit(pending); pending.removeAll() }
        guard active else { return }
        active = false
        continuation.finish()
    }

    func finish() { lock.lock(); finishLocked(); lock.unlock() }
    func stop() {
        lock.lock()
        active = false
        pending.removeAll(keepingCapacity: false)
        continuation.finish()
        lock.unlock()
    }
}

/// Native TTS only: server transcription receives PCM during listening turns.
/// Synthesized buffers play through the existing voice-processing engine, not a
/// second speaker path. Voice quality depends on what is installed on the iPhone.
@MainActor
final class ConversationNativeSpeech: NSObject, AVSpeechSynthesizerDelegate {
    private let synthesizer = AVSpeechSynthesizer()
    private var utterance: AVSpeechUtterance?
    private var producer: NativeSpeechPCMProducer?
    private var playbackTask: Task<Void, Never>?
    private var timeoutTask: Task<Void, Never>?
    private var epoch = UUID()
    private(set) var voiceDescription = "Native iPhone speech"

    override init() { super.init(); synthesizer.delegate = self }

    func speak(_ text: String, contextID: String, audio: ConversationAudio, onFailure: @escaping (String) -> Void) {
        stop()
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, text.count <= 600 else {
            onFailure("Native speech requires a short reply of at most 600 characters.")
            return
        }
        let request = AVSpeechUtterance(string: text)
        let voices = AVSpeechSynthesisVoice.speechVoices().filter { $0.language == "en-US" }
        request.voice = voices.max(by: { $0.quality.rawValue < $1.quality.rawValue }) ?? AVSpeechSynthesisVoice(language: "en-US")
        request.rate = AVSpeechUtteranceDefaultSpeechRate * 0.96
        request.volume = 0.82
        voiceDescription = request.voice.map { "Native: \($0.name) (\($0.language))" } ?? "Native iPhone speech"
        utterance = request
        // 750 x 40 ms = 30 seconds / 1.44 MB maximum queued producer audio. This
        // permits native synthesis to finish quickly while playback stays paced.
        let (stream, continuation) = AsyncThrowingStream<Data, Error>.makeStream(bufferingPolicy: .bufferingOldest(750))
        let producer = NativeSpeechPCMProducer(continuation: continuation)
        self.producer = producer
        let generation = epoch
        playbackTask = Task { [weak self] in
            do {
                var sequence = 0
                for try await pcm in stream {
                    guard let self, self.epoch == generation, !Task.isCancelled else { return }
                    while audio.queuedPlaybackFrames > 9_600 { // Keep at most about 440 ms scheduled.
                        try await Task.sleep(nanoseconds: 20_000_000)
                        guard self.epoch == generation, !Task.isCancelled else { return }
                    }
                    try audio.append(pcm, contextID: contextID, sequence: sequence, sampleRate: 24_000)
                    sequence += 1
                }
                guard let self, self.epoch == generation, !Task.isCancelled else { return }
                guard sequence > 0 else { throw AppError.message("The installed voice did not generate audio. Try another installed voice.") }
                audio.finishResponse(contextID: contextID)
                self.timeoutTask?.cancel(); self.timeoutTask = nil
            } catch is CancellationError {
                return
            } catch {
                guard let self, self.epoch == generation else { return }
                self.stop()
                audio.flushPlayback()
                onFailure(error.localizedDescription)
            }
        }
        timeoutTask = Task { [weak self] in
            do { try await Task.sleep(nanoseconds: 45_000_000_000) } catch { return }
            guard let self, self.epoch == generation else { return }
            self.stop()
            audio.flushPlayback()
            onFailure("Native speech timed out. The reply remains available as text.")
        }
        // BufferCallback itself has no actor guarantee. Explicit Sendable avoids
        // inheriting this method's MainActor isolation on the synthesis callback.
        synthesizer.write(request) { @Sendable buffer in producer.consume(buffer) }
    }

    func stop() {
        epoch = UUID()
        playbackTask?.cancel(); playbackTask = nil
        timeoutTask?.cancel(); timeoutTask = nil
        producer?.stop(); producer = nil
        utterance = nil
        synthesizer.stopSpeaking(at: .immediate)
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        let identity = NativeUtteranceIdentity(utterance: utterance)
        Task { @MainActor [weak self] in
            guard let self, self.utterance === identity.utterance else { return }
            self.producer?.finish()
        }
    }
}
