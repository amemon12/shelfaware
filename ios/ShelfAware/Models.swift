import Foundation
import CoreGraphics

enum CaptureMode: String, Codable, CaseIterable, Identifiable {
    case liveGlasses = "live_glasses"
    case livePhone = "live_phone"
    var id: String { rawValue }
    var label: String { self == .liveGlasses ? "Live glasses" : "Phone-camera fallback" }
}

enum AnalysisMode: String, Codable, CaseIterable, Identifiable {
    case product
    case label
    var id: String { rawValue }
    var label: String { self == .product ? "Front package · product search" : "Nutrition label · fallback" }
}

struct NutritionContext: Codable, Equatable {
    let synthetic: Bool
    let conditions: [String]
    let allergies: [String]
    let focus: [String]
    let dietary_preferences: [String]

}

struct ShopperProfile: Codable, Equatable {
    var display_name: String
    var goals: [String]
    var avoid_ingredients: [String]
    var consented: Bool
    var nutrition_context: NutritionContext? = nil
}

struct NutritionFacts: Codable {
    let basis_grams: Double
    let carbohydrate_g: Double?
    let fiber_g: Double?
    let sugars_g: Double?
    let protein_g: Double?
    let sodium_mg: Double?
}

struct Provenance: Codable {
    let kind: String
    let source: String
    let captured_at: String
    let verified: Bool
}

struct Product: Codable, Identifiable {
    let id: String
    let name: String
    let brand: String?
    let barcode: String?
    let market: String
    let ingredients: [String]
    let nutrition: NutritionFacts
    let provenance: Provenance
    let uncertain_fields: [String]
}

struct CartItem: Codable, Identifiable {
    let id: String
    let product_id: String
    let product_name: String
    let recipient_id: String
    let quantity: Double
    let confirmed_at: String
}

struct ConversationCartItem: Decodable, Identifiable {
    let id: String
    let product_id: String
    let name: String
    let recipient_id: String
    let quantity: Double
}

struct ConversationProduct: Decodable, Identifiable {
    let id: String
    let name: String
}

struct ConversationSnapshot: Decodable {
    let recipient_id: String
    let cart: [ConversationCartItem]
    let products: [ConversationProduct]
    let catalog_ready: Bool
    let analysis_mode: String?
    let evidence_scope: String?
    let comparison_request: String?
}

struct PendingAction: Codable, Identifiable {
    let id: String
    let label: String
    let product_id: String
    let recipient_id: String
    let quantity: Double
    let expires_at: String
}

struct Message: Codable, Identifiable {
    let id: String
    let role: String
    let text: String
    let created_at: String
}

struct ShoppingSession: Codable, Identifiable {
    let id: String
    let mode: String
    let recipient_id: String
    let portion_grams: Double?
    let generation: Int
    let profile: ShopperProfile
    let products: [Product]
    let current_product_id: String?
    let cart: [CartItem]
    let pending_action: PendingAction?
    let messages: [Message]
    let created_at: String
    var currentProduct: Product? { products.first { $0.id == current_product_id } }
}

struct Evidence: Codable {
    let label: String
    let value: String
    let source: String
    let kind: String
    let verified: Bool

    var sourceURL: URL? {
        guard let components = URLComponents(string: source.trimmingCharacters(in: .whitespacesAndNewlines)),
              let scheme = components.scheme?.lowercased(), ["http", "https"].contains(scheme),
              let host = components.host, !host.isEmpty,
              components.user == nil, components.password == nil else { return nil }
        return components.url
    }
}

struct DetectionBox: Codable {
    let x: Double
    let y: Double
    let w: Double
    let h: Double

    /// Events use normalized original-image coordinates. Account for letterbox
    /// margins from aspect-fit rendering, without stretching or cropping.
    func rect(in container: CGSize, image: CGSize) -> CGRect? {
        guard [x, y, w, h].allSatisfy(\.isFinite), x >= 0, y >= 0, w > 0, h > 0,
              x + w <= 1.001, y + h <= 1.001,
              container.width.isFinite, container.height.isFinite,
              image.width.isFinite, image.height.isFinite,
              container.width > 0, container.height > 0, image.width > 0, image.height > 0 else { return nil }
        let scale = min(container.width / image.width, container.height / image.height)
        let width = image.width * scale
        let height = image.height * scale
        let left = (container.width - width) / 2
        let top = (container.height - height) / 2
        let boxLeft = left + CGFloat(x) * width
        let boxTop = top + CGFloat(y) * height
        let boxWidth = CGFloat(w) * width
        let boxHeight = CGFloat(h) * height
        return CGRect(x: boxLeft, y: boxTop, width: boxWidth, height: boxHeight)
    }
}

struct ResponseEvent: Decodable, Identifiable {
    let id: String
    let type: String
    let box: DetectionBox?
}

struct TurnResponse: Decodable {
    let session: ShoppingSession
    let reply: String
    let status: String
    let evidence: [Evidence]
    let pending_action: PendingAction?
    let events: [ResponseEvent]?
    // Unknown fields, including visual catalog products, remain server-owned.
}

struct CreateSessionRequest: Encodable { let mode: String; let profile: ShopperProfile }
struct TurnRequest: Encodable {
    let text: String
    let client_turn_id: String
    let recipient_id: String
    let portion_grams: Double?
    let image_data_url: String?
    let captured_at: String?
    let barcode: String?
    let analysis_mode: String
}
struct ContextRequest: Encodable { let recipient_id: String; let portion_grams: Double? }
struct ConfirmRequest: Encodable { let action_id: String }
struct ReviewRequest: Encodable { let product_id: String; let confirmed: Bool }
struct DeleteResponse: Decodable { let deleted: Bool }
struct HealthResponse: Decodable {
    let status: String
    let providers: [String: Bool]
    let version: String
}

enum AppError: LocalizedError {
    case message(String)
    var errorDescription: String? { switch self { case .message(let message): return message } }
}
