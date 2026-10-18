import Foundation

// Run with swiftc ios/ShelfAware/SceneStabilityGate.swift
// ios/Tests/SceneStabilityGateTests.swift -o /tmp/shelf-scene-gate-tests
@main
struct SceneStabilityGateTests {
    static func main() {
        let textured = (0..<(64 * 64)).map { UInt8($0.isMultiple(of: 2) ? 40 : 180) }
        let changed = textured.map { 255 - $0 }
        var gate = SceneStabilityGate()
        precondition(!gate.consume(textured, at: 0, eligible: true))
        precondition(!gate.consume(textured, at: 0.35, eligible: true))
        precondition(gate.consume(textured, at: 0.70, eligible: true), "Three fresh steady frames should trigger")
        for second in 1...30 {
            precondition(!gate.consume(textured, at: Double(second), eligible: true), "The same view must not repeat")
        }
        precondition(!gate.consume(changed, at: 31, eligible: true))
        precondition(!gate.consume(changed, at: 31.35, eligible: true))
        precondition(!gate.consume(changed, at: 31.71, eligible: true))
        precondition(gate.consume(changed, at: 32.06, eligible: true), "Sustained new scene plus steady dwell should rearm")

        gate = SceneStabilityGate()
        precondition(!gate.consume(textured, at: 0, eligible: true))
        precondition(!gate.consume(textured, at: 0, eligible: true), "A repeated timestamp is not a new frame")
        precondition(!gate.consume(textured, at: 0.70, eligible: true), "Two unique frames are insufficient")
        precondition(gate.consume(textured, at: 1.05, eligible: true))
        for step in 0...6 {
            precondition(!gate.consume(changed, at: 1.4 + Double(step), eligible: true), "Motion cannot bypass cooldown")
        }

        for flat in [UInt8(0), UInt8(80), UInt8(255)] {
            gate = SceneStabilityGate()
            for step in 0...6 {
                precondition(!gate.consume([UInt8](repeating: flat, count: 4096), at: Double(step) * 0.35, eligible: true))
            }
            precondition(!gate.hasUsableView, "Covered and flat images should be rejected")
        }

        gate = SceneStabilityGate()
        for step in 0...6 { precondition(!gate.consume(textured, at: Double(step) * 0.35, eligible: false)) }
        precondition(!gate.consume(textured, at: 2.45, eligible: true))
        precondition(!gate.consume(textured, at: 2.8, eligible: true))
        precondition(gate.consume(textured, at: 3.16, eligible: true), "Busy time cannot count toward idle dwell")

        gate = SceneStabilityGate()
        precondition(!gate.consume(textured, at: 0, eligible: true))
        precondition(!gate.consume(textured, at: 2, eligible: true), "A frame gap resets the dwell")
        precondition(!gate.consume(textured, at: 2.35, eligible: true))
        precondition(gate.consume(textured, at: 2.71, eligible: true))

        gate = SceneStabilityGate()
        precondition(!gate.consume(textured, at: 0, eligible: true))
        precondition(!gate.consume(textured, at: 0.35, eligible: true))
        gate.breakStability()
        precondition(!gate.consume(textured, at: 0.70, eligible: true), "Pause/context invalidation must reset dwell")
        precondition(!gate.consume(textured, at: 1.05, eligible: true))
        precondition(gate.consume(textured, at: 1.41, eligible: true))

        gate = SceneStabilityGate()
        gate.recordCapture(textured, at: 0)
        for step in 1...20 {
            precondition(!gate.consume(textured, at: Double(step), eligible: true), "A spoken/manual camera request must also suppress duplicate automatic uploads")
        }
        precondition(!gate.consume(changed, at: 21, eligible: true))
        precondition(!gate.consume(changed, at: 21.35, eligible: true))
        precondition(!gate.consume(changed, at: 21.71, eligible: true))
        precondition(gate.consume(changed, at: 22.06, eligible: true))
        func alter(_ pixels: [UInt8], x: Int, y: Int, width: Int, height: Int,
                   change: (UInt8) -> UInt8) -> [UInt8] {
            var result = pixels
            for row in y..<(y + height) {
                for column in x..<(x + width) {
                    result[row * 64 + column] = change(result[row * 64 + column])
                }
            }
            return result
        }
        let oneProduct = alter(textured, x: 28, y: 32, width: 10, height: 25) { 255 - $0 }
        gate = SceneStabilityGate()
        gate.recordCapture(textured, at: 0)
        precondition(!gate.consume(oneProduct, at: 8.0, eligible: true))
        precondition(!gate.consume(oneProduct, at: 8.35, eligible: true))
        precondition(!gate.consume(oneProduct, at: 8.71, eligible: true))
        precondition(gate.consume(oneProduct, at: 9.06, eligible: true), "A changed 6% product region must rearm without moving the whole background")
        for step in 1...40 {
            precondition(!gate.consume(oneProduct, at: 9.06 + Double(step) * 0.35, eligible: true), "The newly captured product must not repeatedly upload")
        }

        var sparseNoise = textured
        for index in stride(from: 0, to: 4096, by: 12) { sparseNoise[index] = 255 - sparseNoise[index] }
        let globalExposure = textured.map { $0 + 30 }
        let localExposure = alter(textured, x: 24, y: 24, width: 16, height: 24) { $0 + 30 }
        let largeExposure = alter(textured, x: 8, y: 8, width: 40, height: 40) { $0 + 30 }
        for nuisance in [sparseNoise, globalExposure, localExposure, largeExposure] {
            gate = SceneStabilityGate()
            gate.recordCapture(textured, at: 0)
            for step in 0...24 {
                precondition(!gate.consume(nuisance, at: 8 + Double(step) * 0.35, eligible: true), "Noise or exposure alone must not rearm")
            }
        }

        // A hand, head movement, or background blink that returns to the old
        // view before steady dwell must never latch a duplicate upload.
        for motion in [oneProduct, changed] {
            gate = SceneStabilityGate()
            gate.recordCapture(textured, at: 0)
            for cycle in 0..<6 {
                let time = 8 + Double(cycle) * 2
                precondition(!gate.consume(motion, at: time, eligible: true))
                precondition(!gate.consume(motion, at: time + 0.35, eligible: true))
                precondition(!gate.consume(textured, at: time + 0.70, eligible: true))
                precondition(!gate.consume(textured, at: time + 1.05, eligible: true))
                precondition(!gate.consume(textured, at: time + 1.40, eligible: true))
            }
        }

        gate = SceneStabilityGate()
        gate.recordCapture(textured, at: 0)
        for step in 0...12 {
            precondition(!gate.consume(oneProduct, at: 8 + Double(step) * 0.35, eligible: false), "New products cannot interrupt speech or playback")
        }
        precondition(!gate.consume(oneProduct, at: 12.55, eligible: true))
        precondition(!gate.consume(oneProduct, at: 12.90, eligible: true))
        precondition(gate.consume(oneProduct, at: 13.26, eligible: true), "After playback finishes a waiting new product requires fresh idle dwell")

        gate = SceneStabilityGate()
        gate.recordCapture(textured, at: 0)
        precondition(!gate.consume(oneProduct, at: 8, eligible: true))
        precondition(!gate.consume(oneProduct, at: 8, eligible: true), "Duplicate frame time cannot supply regional persistence")
        precondition(!gate.consume(oneProduct, at: 10, eligible: true), "A gap restarts regional persistence")
        precondition(!gate.consume(oneProduct, at: 10.35, eligible: true))
        precondition(!gate.consume(oneProduct, at: 10.70, eligible: true))
        precondition(gate.consume(oneProduct, at: 11.06, eligible: true))
        print("SceneStabilityGate: steady, duplicate, cooldown, regional product changes, transient motion, exposure/noise, dark/flat, speech/playback, gap, and invalidation checks passed")

    }
}
