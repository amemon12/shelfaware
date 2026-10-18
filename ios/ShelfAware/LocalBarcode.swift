import ImageIO
import UIKit
@preconcurrency import Vision

enum LocalBarcode {
    /// Barcode decoding happens on the iPhone. It supplies identity evidence, never nutrition.
    static func detect(in image: UIImage?) async -> String? {
        guard let image, let cgImage = image.cgImage else { return nil }
        let orientation = orientation(image.imageOrientation)
        return await Task.detached(priority: .userInitiated) {
            // A phone fallback still may be 12+ MP. Bound optional barcode work
            // independently of the original photo and never upscale a glasses frame.
            guard let boundedImage = bounded(cgImage, maximumEdge: 1600) else { return nil }
            let request = VNDetectBarcodesRequest()
            // UPC-E requires expansion before GTIN checksum validation; do not mislabel it GTIN-8.
            // UPC-A is conventionally decoded by Vision as EAN-13 with a leading zero.
            request.symbologies = [.ean13, .ean8]
            do {
                try VNImageRequestHandler(cgImage: boundedImage, orientation: orientation).perform([request])
                let candidates = (request.results ?? []).filter { $0.confidence >= 0.8 }
                    .compactMap(\.payloadStringValue)
                    .filter(validGTIN)
                let unique = Set(candidates)
                // Multiple different barcodes in view are ambiguous. Ask backend from image instead.
                return unique.count == 1 ? unique.first : nil
            } catch { return nil } // OCR remains available; no fabricated barcode or identity.
        }.value
    }

    private static func bounded(_ image: CGImage, maximumEdge: Int) -> CGImage? {
        let edge = max(image.width, image.height)
        guard edge > maximumEdge else { return image }
        let factor = Double(maximumEdge) / Double(edge)
        let width = max(1, Int(Double(image.width) * factor))
        let height = max(1, Int(Double(image.height) * factor))
        guard let context = CGContext(data: nil, width: width, height: height,
                                      bitsPerComponent: 8, bytesPerRow: 0,
                                      space: CGColorSpaceCreateDeviceRGB(),
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(x: 0, y: 0, width: CGFloat(width), height: CGFloat(height)))
        return context.makeImage()
    }

    private static func validGTIN(_ value: String) -> Bool {
        let bytes = Array(value.utf8)
        guard [8, 12, 13, 14].contains(bytes.count), bytes.allSatisfy({ $0 >= 48 && $0 <= 57 }) else { return false }
        let digits = bytes.map { Int($0) - 48 }
        guard let check = digits.last else { return false }
        let sum = digits.dropLast().reversed().enumerated().reduce(0) { result, digit in
            result + digit.element * (digit.offset.isMultiple(of: 2) ? 3 : 1)
        }
        return (10 - sum % 10) % 10 == check
    }

    private static func orientation(_ value: UIImage.Orientation) -> CGImagePropertyOrientation {
        switch value {
        case .up: return .up
        case .down: return .down
        case .left: return .left
        case .right: return .right
        case .upMirrored: return .upMirrored
        case .downMirrored: return .downMirrored
        case .leftMirrored: return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default: return .up
        }
    }
}
