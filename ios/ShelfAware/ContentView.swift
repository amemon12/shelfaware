import SwiftUI
import UIKit

struct ContentView: View {
    @ObservedObject var model: ShoppingViewModel

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    VStack(alignment: .leading, spacing: 8) {
                        Label("A little clarity, in the aisle.", systemImage: "sparkle")
                            .font(.title3.weight(.semibold))
                        Text(buildIdentifier)
                            .font(.caption.weight(.semibold)).foregroundStyle(.mint)
                        Text(model.mode.label.uppercased())
                            .font(.caption.weight(.bold)).foregroundStyle(.mint)
                        Text(model.stage).font(.subheadline).foregroundStyle(.secondary)
                        if model.session == nil {
                            Text("Review your goal and allow voice processing, then tap Start conversation below.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        if model.busy { ProgressView().accessibilityLabel("Working") }
                    }.padding(.vertical, 8)
                }
                if model.session == nil { setup }
                else { shopping }
                if !model.conversation.isRunning { AudioStatusView(audio: model.audio) }
                if let error = model.error {
                    Section("Needs attention") {
                        Text(error).foregroundStyle(.orange).textSelection(.enabled)
                        Button("Dismiss") { model.error = nil }
                    }
                }
            }
            .navigationTitle("Shelf Aware")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Stop", systemImage: "stop.fill", role: .destructive) { model.stop() }
                        .accessibilityHint("Immediately stop listening, speaking, capture, and the current request")
                }
            }
            .fullScreenCover(isPresented: $model.showPhoneCamera) {
                PhoneCameraPicker(onCapture: model.capturePhonePhoto, onCancel: { model.showPhoneCamera = false })
                    .ignoresSafeArea()
            }
        }
    }

    private var buildIdentifier: String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "unknown"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "unknown"
        return "Conversation v\(version) · build \(build)"
    }

    private var setup: some View {
        Group {
            Section("Talk with Shelf Aware") {
                Text("Goal: \(model.goalsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none selected" : model.goalsText)")
                    .font(.headline)
                Text("Avoid: \(model.avoidIngredientsText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? "none selected" : model.avoidIngredientsText)")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Allow this profile, questions, and product photos to the configured AI providers", isOn: $model.consented)
                    .font(.subheadline)
                Toggle("Allow microphone audio to the backend for transcription", isOn: $model.voiceConsented)
                    .font(.subheadline)
                if model.mode == .liveGlasses {
                    Text("Steady-view photos start automatically with conversation. Preview frames stay on this iPhone; you can turn automatic capture off in the conversation controls.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Button("Start conversation", systemImage: "waveform.circle.fill") { model.startConversationFromSetup() }
                    .font(.headline).frame(maxWidth: .infinity)
                    .buttonStyle(.borderedProminent)
                    .disabled(!model.consented || !model.voiceConsented || model.busy)
                Text("Creates your shopping session and starts voice in one step. Keep the app open; Stop ends listening and capture.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(model.busy)
            Section("Connection") {
                TextField("Backend origin · http://192.168.1.20:8000", text: $model.serverAddress)
                    .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                SecureField("Optional LAN bearer token", text: $model.bearerToken)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                TextField("Optional dashboard origin · port 5173", text: $model.dashboardAddress)
                    .keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                Button("Check backend") { model.checkHealth() }.disabled(model.busy)
                Text(model.health).font(.caption).foregroundStyle(.secondary)
                Text("The token stays in memory. This app has no provider API keys. Use a trusted LAN or HTTPS backend.")
                    .font(.caption).foregroundStyle(.secondary)
            }.disabled(model.busy)
            Section("Use the dashboard's profile") {
                TextField("Paste existing session ID", text: $model.attachSessionID)
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                Button("Load session for review", systemImage: "link") { model.loadSessionToAttach() }
                    .disabled(model.attachSessionID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Text("Use the same backend address. The dashboard must create Live glasses or Live phone mode. Synthetic and recorded sessions cannot become live captures.")
                    .font(.caption).foregroundStyle(.secondary)
                if let attached = model.pendingAttach {
                    Text(attached.profile.display_name).font(.headline)
                    Text("Source: \(attached.mode) · recipient: \(attached.recipient_id)").font(.caption)
                    Text("Goals: \(attached.profile.goals.isEmpty ? "none selected" : attached.profile.goals.joined(separator: "; "))")
                    Text("Avoid: \(attached.profile.avoid_ingredients.isEmpty ? "none selected" : attached.profile.avoid_ingredients.joined(separator: ", "))")
                    if let context = attached.profile.nutrition_context {
                        NutritionContextView(context: context)
                    }
                    Text("Portion: \(attached.portion_grams.map { String(format: "%g g", $0) } ?? "not set") · \(attached.cart.count) confirmed list items")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("I reviewed this profile and intend to use this session for the current wearer", isOn: $model.attachReviewed)
                    Button("Attach · clear previous pending actions", systemImage: "person.crop.circle.badge.checkmark") { model.attachLoadedSession() }
                        .disabled(!model.attachReviewed)
                }
            }.disabled(model.busy)
            Section("Capture source") {
                Picker("Source", selection: $model.mode) {
                    if GlassesCamera.sdkIncluded { Text("Live glasses").tag(CaptureMode.liveGlasses) }
                    Text("Phone-camera fallback").tag(CaptureMode.livePhone)
                }.disabled(model.pendingAttach != nil)
                if model.mode == .liveGlasses {
                    GlassesStatusView(camera: model.camera)
                    Button("Connect in Meta AI") { Task { await model.camera.register() } }
                    GlassesUpdateButton(camera: model.camera)
                    Text("Pair your glasses in Meta AI first. Camera permission may open Meta AI. Keep this app foreground while shopping.")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("Captures come from the iPhone camera. This mode does not demonstrate a glasses camera integration.")
                        .font(.caption).foregroundStyle(.orange)
                }
            }.disabled(model.busy)
            analysisControls
            Section("Your reviewed profile") {
                Text("Avoid sugar").font(.headline)
                TextField("Goals, separated by commas", text: $model.goalsText, axis: .vertical).lineLimit(2...4)
                TextField("Ingredients to avoid, separated by commas", text: $model.avoidIngredientsText, axis: .vertical)
            }.disabled(model.busy)
            Section {
                Button("Start shopping", systemImage: "basket") { model.startSession() }
                    .font(.headline).disabled(!model.consented || model.busy)
                Text("Starts a session for manual Look & Ask without continuous conversation.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    @ViewBuilder private var shopping: some View {
        ConversationPanel(model: model, conversation: model.conversation)
        if let session = model.session {
            Section("Active reviewed profile") {
                Text(session.profile.display_name).font(.headline)
                Text(session.profile.goals.isEmpty ? "No personal comparison goals selected." : session.profile.goals.joined(separator: " · "))
                    .font(.subheadline)
                Text("Avoid: \(session.profile.avoid_ingredients.isEmpty ? "none selected" : session.profile.avoid_ingredients.joined(separator: ", "))")
                    .font(.caption).foregroundStyle(.secondary)
                if let context = session.profile.nutrition_context {
                    NutritionContextView(context: context)
                    Text(model.recipient == "self" ? "This attached session includes fictional nutrition context." : "Household mode does not apply this session's fictional nutrition context.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("Changing who wears the glasses does not change this profile. Disconnect and attach the intended dashboard session when switching people.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        if !model.conversation.isRunning { analysisControls }
        Section("Who and how much?") {
            Picker("Shopping for", selection: Binding(get: { model.recipient }, set: model.changeRecipient)) {
                Text("Me").tag("self")
                Text("Household").tag("household")
            }.pickerStyle(.segmented)
            if model.analysisMode == .label {
                TextField("Intended portion in grams · optional until asked", text: $model.portionText)
                    .keyboardType(.decimalPad)
                    .disabled(model.conversation.isRunning)
            }
            if model.recipient == "household" {
                Text("Household mode uses the product facts; your personal goals do not carry over.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }.disabled(model.busy)

        if !model.conversation.isRunning {
            Section("Ask about a product") {
                TextField("What would you like to know?", text: $model.question, axis: .vertical).lineLimit(2...5)
                SpeechButton(audio: model.audio, disabled: model.busy, action: model.toggleListening)
                Text("Push to talk, then review the words and tap Ask. Listening stops after 12 seconds. Apple speech may use the network when on-device recognition is unavailable.")
                    .font(.caption).foregroundStyle(.secondary)
                if model.mode == .liveGlasses {
                    Button("Look & Ask · new glasses photo", systemImage: "eyeglasses") { model.ask(captureNewGlassesPhoto: true) }
                        .font(.headline).disabled(model.busy)
                    GlassesStatusView(camera: model.camera)
                    Button("Ask about the previous products", systemImage: "bubble.left") { model.ask() }
                        .disabled(model.busy)
                } else {
                    Button(model.analysisMode == .product ? "Photograph front packages with iPhone" : "Photograph label with iPhone", systemImage: "camera") { model.openPhoneCamera() }.disabled(model.busy)
                    Button(model.pendingPhoto == nil ? "Ask about previous products" : "Ask with this phone photo", systemImage: "arrow.up.circle.fill") { model.ask() }
                        .font(.headline).disabled(model.busy)
                }
            }

        }

        if let photo = model.lastPhoto {
            Section("Most recent requested capture") {
                ProductCaptureView(image: photo.image, boxes: model.lastProductBoxes)
                if !model.lastProductBoxes.isEmpty {
                    Text(model.lastProductBoxes.count == 1 ? "Model-selected held product · unverified" : "Model-selected held products · unverified")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Text("\(photo.mode.label) · received \(photo.capturedAt.formatted(date: .omitted, time: .standard))")
                    .font(.caption).foregroundStyle(.secondary)
                Text("The photo stays in memory. Only captures you submit are sent to the backend.").font(.caption).foregroundStyle(.secondary)
            }
        }

        if !model.conversation.isRunning, let response = model.lastResponse {
            Section("Shelf Aware says") {
                Text(response.reply).font(.title3).textSelection(.enabled)
                ForEach(Array(response.evidence.enumerated()), id: \.offset) { _, item in
                    VStack(alignment: .leading, spacing: 3) {
                        Text("\(item.label): \(item.value)").font(.subheadline)
                        Text("\(item.kind) · \(item.verified ? "reviewed" : "unverified")")
                            .font(.caption).foregroundStyle(.secondary)
                        if let url = item.sourceURL {
                            Link("Source: \(url.host ?? "open source")", destination: url).font(.caption)
                        } else {
                            Text(item.source).font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            if model.analysisMode == .label, response.status == "needs_review", let product = model.session?.currentProduct {
                Section("Check the physical package") {
                    NutritionReviewView(product: product)
                    Toggle("I checked the product, variant, units, serving basis, and all fields against this package", isOn: $model.labelChecked)
                        .font(.subheadline)
                    Button("Confirm this label") { model.reviewLabel(confirmed: true) }.disabled(!model.labelChecked || model.busy)
                    Button("Incorrect or unreadable · reject and recapture", role: .destructive) { model.reviewLabel(confirmed: false) }.disabled(model.busy)
                    Text("This iPhone view supports confirm or reject. Correct individual fields in the companion dashboard, then ask again.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
        }
        if !model.conversation.isRunning, let action = model.session?.pending_action {
            Section("Confirm before adding") {
                Text("\(action.label) · for \(action.recipient_id == "self" ? "me" : "household")")
                Button("Confirm add to list", systemImage: "checkmark.circle.fill") { model.confirmCart() }.disabled(model.busy)
                Text("Looking at a product never adds it to the list. This is a shopping list, not a purchase.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        if let session = model.session {
            Section("Confirmed shopping list") {
                if let snapshot = model.conversationSnapshot {
                    if snapshot.cart.isEmpty { Text("Nothing confirmed yet.").foregroundStyle(.secondary) }
                    ForEach(snapshot.cart) { item in
                        LabeledContent(item.name, value: item.recipient_id == "self" ? "Me" : "Household")
                    }
                } else {
                    if session.cart.isEmpty { Text("Nothing confirmed yet.").foregroundStyle(.secondary) }
                    ForEach(session.cart) { item in
                        LabeledContent(item.product_name, value: item.recipient_id == "self" ? "Me" : "Household")
                    }
                }
            }
            Section("Companion evidence view") {
                Text(session.id).font(.caption.monospaced()).textSelection(.enabled)
                Button("Copy session ID", systemImage: "doc.on.doc") { UIPasteboard.general.string = session.id }
                if let url = model.dashboardURL {
                    ShareLink("Share dashboard link", item: url)
                    Text(url.absoluteString).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                }
                Text("The browser observes this session's real backend results. The glasses have no lens display. The bearer token is never included in this link.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section {
                Button("Disconnect · keep session on server", systemImage: "link.badge.plus") { model.disconnectSession() }
                    .disabled(model.busy)
                Button("End and delete session", systemImage: "trash", role: .destructive) { model.endSession() }.disabled(model.busy)
                Text("Deletes backend session state and in-memory photos after confirmation from the server. External provider retention is governed separately.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var analysisControls: some View {
        Section("What to show") {
            Picker("Read mode", selection: Binding(get: { model.analysisMode }, set: model.changeAnalysisMode)) {
                ForEach(AnalysisMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            Text(model.analysisMode == .product
                 ? "Hold one or two products with their front names and branding facing the glasses. Look & Ask automatically compares two products for the current goal."
                 : "Show the nutrition panel and serving-size line. This fallback needs a printed gram-based serving and physical review before comparison.")
                .font(.caption).foregroundStyle(.secondary)
        }.disabled(model.busy)
    }
}

private struct NutritionContextView: View {
    let context: NutritionContext
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(context.synthetic ? "Fictional nutrition context" : "Nutrition context").font(.subheadline.weight(.semibold))
            Text("Conditions: \(context.conditions.isEmpty ? "none" : context.conditions.joined(separator: ", "))")
            Text("Allergies: \(context.allergies.isEmpty ? "none" : context.allergies.joined(separator: ", "))")
            Text("Focus: \(context.focus.joined(separator: "; "))")
            Text("Preferences: \(context.dietary_preferences.isEmpty ? "none" : context.dietary_preferences.joined(separator: ", "))")
        }.font(.caption).foregroundStyle(.secondary)
    }
}

private struct ProductCaptureView: View {
    let image: UIImage
    let boxes: [DetectionBox]
    var body: some View {
        GeometryReader { geometry in
            let rects = boxes.compactMap { $0.rect(in: geometry.size, image: image.size) }
            ZStack(alignment: .topLeading) {
                Image(uiImage: image).resizable().scaledToFit()
                    .frame(width: geometry.size.width, height: geometry.size.height)
                ForEach(Array(rects.enumerated()), id: \.offset) { index, rect in
                    Rectangle().stroke(.mint, lineWidth: 3)
                        .frame(width: rect.width, height: rect.height)
                        .overlay(alignment: .topLeading) {
                            Text("\(index + 1)").font(.caption.weight(.bold))
                                .foregroundStyle(.black)
                                .padding(.horizontal, 6).padding(.vertical, 3)
                                .background(.mint)
                        }
                        .position(x: rect.midX, y: rect.midY)
                        .allowsHitTesting(false)
                }
            }
        }
        .frame(height: 220)
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .accessibilityLabel(boxes.isEmpty ? "Requested product photo" : "Requested product photo with \(boxes.count) numbered, unverified model-selected product boxes")
    }
}

private struct GlassesUpdateButton: View {
    @ObservedObject var camera: GlassesCamera
    var body: some View {
        Button("Open glasses update in Meta AI") {
            Task { await camera.openGlassesAppUpdate() }
        }
        .disabled(camera.isOpeningGlassesUpdate)
    }
}

private struct GlassesStatusView: View {
    @ObservedObject var camera: GlassesCamera
    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            Text("Glasses: \(camera.state) · registration: \(camera.registration)").font(.caption)
            if let at = camera.frameReceivedAt {
                Text("Actual preview frame received at \(at.formatted(date: .omitted, time: .standard))").font(.caption).foregroundStyle(.secondary)
            }
            if let image = camera.preview {
                Text("Requested 7 fps · received image \(image.cgImage?.width ?? Int(image.size.width * image.scale)) × \(image.cgImage?.height ?? Int(image.size.height * image.scale)) pixels")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if let error = camera.error { Text(error).font(.caption).foregroundStyle(.orange) }
        }
    }
}

private struct AudioStatusView: View {
    @ObservedObject var audio: AudioController
    var body: some View {
        Section("Audio") {
            Label(audio.isListening ? "Listening now" : audio.isSpeaking ? "Speaking" : "Microphone stopped", systemImage: audio.isListening ? "mic.fill" : "speaker.wave.2")
                .foregroundStyle(audio.isListening ? .orange : .secondary)
            Text(audio.route).font(.caption).textSelection(.enabled)
            if let error = audio.error { Text(error).font(.caption).foregroundStyle(.orange) }
        }
    }
}

private struct SpeechButton: View {
    @ObservedObject var audio: AudioController
    let disabled: Bool
    let action: () -> Void
    var body: some View {
        Button(audio.isListening ? "Finish listening" : "Speak a question", systemImage: audio.isListening ? "stop.circle" : "mic") { action() }
            .disabled(disabled && !audio.isListening)
    }
}

private struct NutritionReviewView: View {
    let product: Product
    private func display(_ value: Double?, unit: String) -> String { value.map { String(format: "%g %@", $0, unit) } ?? "Unknown" }
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(product.name).font(.headline)
            Text("\(product.brand ?? "Brand unknown") · \(product.market) · barcode \(product.barcode ?? "unknown")").font(.caption)
            LabeledContent("Serving basis", value: display(product.nutrition.basis_grams, unit: "g"))
            LabeledContent("Total carbohydrate", value: display(product.nutrition.carbohydrate_g, unit: "g"))
            LabeledContent("Fiber", value: display(product.nutrition.fiber_g, unit: "g"))
            LabeledContent("Sugars", value: display(product.nutrition.sugars_g, unit: "g"))
            LabeledContent("Protein", value: display(product.nutrition.protein_g, unit: "g"))
            LabeledContent("Sodium", value: display(product.nutrition.sodium_mg, unit: "mg"))
            Text("Ingredients: \(product.ingredients.isEmpty ? "unknown" : product.ingredients.joined(separator: ", "))").font(.caption)
            if !product.uncertain_fields.isEmpty { Text("Uncertain: \(product.uncertain_fields.joined(separator: ", "))").foregroundStyle(.orange) }
            Text("Source: \(product.provenance.source) · \(product.provenance.captured_at)").font(.caption).foregroundStyle(.secondary)
        }
    }
}
