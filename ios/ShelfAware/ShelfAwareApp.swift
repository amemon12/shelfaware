import SwiftUI

@main
@MainActor
struct ShelfAwareApp: App {
    @StateObject private var model = ShoppingViewModel()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .preferredColorScheme(.dark)
                .tint(Color(red: 0.48, green: 0.90, blue: 0.70))
                .onOpenURL { url in Task { await model.camera.handleURL(url) } }
                .onChange(of: scenePhase) { _, phase in
                    if phase == .background { model.backgrounded() }
                }
        }
    }
}
