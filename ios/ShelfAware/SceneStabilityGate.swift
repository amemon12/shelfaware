import Foundation
#if canImport(UIKit)
import UIKit
#endif

/// Local image motion gate, not a product detector. A trigger requests a new
/// still photo; these small preview samples never leave the phone.
struct SceneStabilityGate {
    private var previous: [UInt8]?
    private var previousTime: TimeInterval?
    private var stableSince: TimeInterval?
    private var stableFrames = 0
    private var submittedReference: [UInt8]?
    private var submittedAt: TimeInterval?
    private var changedSince: TimeInterval?
    private var changedFrames = 0
    private(set) var isArmed = true
    private(set) var hasUsableView = false

    mutating func breakStability() {
        previous = nil
        previousTime = nil
        stableSince = nil
        stableFrames = 0
        changedSince = nil
        changedFrames = 0
    }

    mutating func recordCapture(_ pixels: [UInt8], at time: TimeInterval) {
        guard pixels.count == 64 * 64, time.isFinite else { return }
        isArmed = false
        submittedReference = pixels
        submittedAt = time
        breakStability()
    }

    mutating func consume(_ pixels: [UInt8], at time: TimeInterval, eligible: Bool) -> Bool {
        guard time.isFinite, pixels.count == 64 * 64 else { breakStability(); return false }
        if let previousTime, time <= previousTime { return false }
        let minimum = pixels.min() ?? 0
        let maximum = pixels.max() ?? 0
        let mean = pixels.reduce(0.0) { $0 + Double($1) } / Double(pixels.count)
        let variance = pixels.reduce(0.0) { $0 + pow(Double($1) - mean, 2) } / Double(pixels.count)
        // Reject covered/dark or nearly flat views; this does not assert hands
        // or packages are present. Perceptron remains the held-product gate.
        hasUsableView = mean >= 12 && maximum - minimum >= 24 && variance >= 64
        guard hasUsableView else { breakStability(); return false }

        defer { previous = pixels; previousTime = time }
        if let reference = submittedReference, let submittedAt {
            let changed = Self.sceneChanged(reference, pixels)
            if !changed {
                // A brief hand/head movement that returns to the submitted
                // view must not leave a latched rearm and upload it again.
                isArmed = false
                breakStability()
                return false
            }
            if !isArmed {
                guard time - submittedAt >= 8 else {
                    changedSince = nil; changedFrames = 0
                    stableSince = nil; stableFrames = 0
                    return false
                }
                if changedSince == nil || previousTime == nil || time - (previousTime ?? time) > 1 {
                    changedSince = time
                    changedFrames = 1
                } else {
                    changedFrames += 1
                }
                guard changedFrames >= 2, let changedSince, time - changedSince >= 0.25 else {
                    stableSince = nil; stableFrames = 0
                    return false
                }
                isArmed = true
                breakStability()
            }
        }

        guard isArmed, eligible else { stableSince = nil; stableFrames = 0; return false }
        guard let previous, let previousTime, time - previousTime <= 1,
              Self.changedFraction(previous, pixels) < 0.10 else {
            stableSince = time
            stableFrames = 1
            return false
        }
        stableFrames += 1
        if stableSince == nil { stableSince = previousTime }
        guard stableFrames >= 3, let stableSince, time - stableSince >= 0.7 else { return false }
        recordCapture(pixels, at: time)
        return true
    }

    private static func changedFraction(_ lhs: [UInt8], _ rhs: [UInt8]) -> Double {
        guard lhs.count == rhs.count, !lhs.isEmpty else { return 1 }
        let changed = zip(lhs, rhs).reduce(0) { $0 + (abs(Int($1.0) - Int($1.1)) > 20 ? 1 : 0) }
        return Double(changed) / Double(lhs.count)
    }

    /// Exposure shifts alone are not new products. Regional change catches a
    /// held package occupying far less than a quarter of the whole frame, but
    /// requires both substantial pixel and texture change in a contiguous area.
    private static func sceneChanged(_ lhs: [UInt8], _ rhs: [UInt8]) -> Bool {
        guard lhs.count == 4096, rhs.count == 4096 else { return false }
        let offset = zip(lhs, rhs).reduce(0.0) { $0 + Double(Int($1.1) - Int($1.0)) } / 4096
        var changed = [Bool](repeating: false, count: 4096)
        var edges = [Bool](repeating: false, count: 4096)
        var changedCount = 0
        var edgeCount = 0
        for y in 0..<64 {
            for x in 0..<64 {
                let i = y * 64 + x
                changed[i] = abs(Double(Int(rhs[i]) - Int(lhs[i])) - offset) > 20
                if changed[i] { changedCount += 1 }
                guard x < 63, y < 63 else { continue }
                let horizontal = abs((Int(rhs[i + 1]) - Int(rhs[i])) - (Int(lhs[i + 1]) - Int(lhs[i])))
                let vertical = abs((Int(rhs[i + 64]) - Int(rhs[i])) - (Int(lhs[i + 64]) - Int(lhs[i])))
                edges[i] = horizontal > 20 || vertical > 20
                if edges[i] { edgeCount += 1 }
            }
        }
        if Double(changedCount) / 4096 > 0.25 && Double(edgeCount) / 4096 > 0.10 { return true }
        // Four percent overall avoids single tiny highlights/noisy pixels. A
        // 16x16 moving window can still detect one roughly 6–10% can or box.
        guard Double(changedCount) / 4096 >= 0.04 else { return false }
        for y in stride(from: 0, through: 48, by: 8) {
            for x in stride(from: 0, through: 48, by: 8) {
                var regionChanges = 0
                var regionEdges = 0
                for row in y..<(y + 16) {
                    for column in x..<(x + 16) {
                        let i = row * 64 + column
                        if changed[i] { regionChanges += 1 }
                        if edges[i] { regionEdges += 1 }
                    }
                }
                if Double(regionChanges) / 256 >= 0.40 && Double(regionEdges) / 256 >= 0.20 { return true }
            }
        }
        return false
    }

    #if canImport(UIKit)
    static func sample(_ image: UIImage) -> [UInt8]? {
        guard let source = image.cgImage else { return nil }
        var pixels = [UInt8](repeating: 0, count: 64 * 64)
        let drawn = pixels.withUnsafeMutableBytes { storage -> Bool in
            guard let context = CGContext(data: storage.baseAddress, width: 64, height: 64,
                                          bitsPerComponent: 8, bytesPerRow: 64,
                                          space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue) else { return false }
            context.interpolationQuality = .low
            context.draw(source, in: CGRect(x: 0, y: 0, width: 64, height: 64))
            return true
        }
        return drawn ? pixels : nil
    }
    #endif
}
