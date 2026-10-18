import SwiftUI

/// Controls and evidence for the actual microphone/WebSocket/PCM conversation.
/// No provider credentials or simulated transcripts are embedded in this view.
struct ConversationPanel: View {
    @ObservedObject var model: ShoppingViewModel
    @ObservedObject var conversation: ConversationClient

    var body: some View {
        Section("Talk with Shelf Aware") {
            if model.mode == .liveGlasses {
                Toggle("Automatically capture when the view is steady", isOn: Binding(
                    get: { model.automaticCaptureEnabled }, set: model.setAutomaticCapture))
                Text("Preview frames are checked only on this iPhone. After a steady view, one fresh photo is sent for product recognition. Move to new products to rearm; it does not continuously upload video.")
                    .font(.caption).foregroundStyle(.secondary)
                Text(model.automaticCaptureStatus).font(.caption).foregroundStyle(.mint)
            }
            if conversation.isRunning {
                Label(conversation.isConnected ? "Conversation connected" : "Reconnecting — audio paused",
                      systemImage: conversation.isConnected ? "waveform" : "wifi.slash")
                    .foregroundStyle(conversation.isConnected ? .mint : .orange)
                Text(conversation.state).font(.subheadline)
                Text(conversation.microphoneEnabled ? "Microphone listening" : "Microphone paused · processing or playing the answer")
                    .font(.caption).foregroundStyle(conversation.microphoneEnabled ? .mint : .secondary)
                ProgressView(value: Double(conversation.microphoneLevel), total: 1)
                    .accessibilityLabel("Microphone input level")
                Text(conversation.route).font(.caption).textSelection(.enabled)
                Text(conversation.providerDescription).font(.caption).foregroundStyle(.secondary)
                if !conversation.voiceName.isEmpty {
                    Text(conversation.voiceName).font(.caption).foregroundStyle(.secondary)
                }
                Text(model.mode == .liveGlasses && model.automaticCaptureEnabled
                     ? "Hold one or two products steady; I’ll compare them automatically. Microphone input pauses while I process and speak, then resumes after the full answer. Use Interrupt to stop a reply, or ask a follow-up once listening resumes."
                     : "Ask what you want to compare, then pause. Microphone input pauses while I process and speak, then resumes after the full answer. Use Interrupt to stop a reply.")
                    .font(.caption).foregroundStyle(.secondary)
                HStack {
                    Button("Interrupt", systemImage: "hand.raised") { conversation.interrupt() }
                    Spacer()
                    Button("End conversation", systemImage: "mic.slash", role: .destructive) { model.endConversation() }
                }
                TextField("Or type a follow-up", text: $model.question, axis: .vertical).lineLimit(1...3)
                Button("Send typed question", systemImage: "arrow.up.circle") { conversation.sendText(model.question) }
                    .disabled(!conversation.isConnected || model.question.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                if model.mode == .livePhone {
                    Button("Take a fresh phone photo", systemImage: "camera") { model.openPhoneCamera() }
                    Text("Phone-camera fallback: taking a photo may pause the voice session. Resume conversation afterward.")
                        .font(.caption).foregroundStyle(.orange)
                }
            } else {
                Toggle("Allow microphone audio to the backend for transcription, and questions and requested product photos to the configured AI providers", isOn: $model.voiceConsented)
                    .font(.subheadline).disabled(model.busy)
                Button("Start conversation", systemImage: "waveform.circle.fill") { model.startConversation() }
                    .font(.headline).disabled(!model.voiceConsented || model.busy)
                Text("Listens for each question, pauses input while processing and speaking, and listens again after playback. Product photos come from your questions or the automatic steady-view setting. Backgrounding or Stop ends microphone and camera capture.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            if !conversation.transcript.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Heard").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Text(conversation.transcript).textSelection(.enabled)
                }
            }
            if !conversation.reply.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Shelf Aware").font(.caption.weight(.semibold)).foregroundStyle(.secondary)
                    Text(conversation.reply).font(.title3).textSelection(.enabled)
                }
            }
            if let snapshot = model.conversationSnapshot {
                if let request = snapshot.comparison_request, !request.isEmpty {
                    Text("Comparing for: \(request)").font(.subheadline)
                }
                if snapshot.analysis_mode == "product" || snapshot.evidence_scope == "catalog_search" {
                    Text("Answers use product listings. Check the shown product match against the package.")
                        .font(.caption).foregroundStyle(.secondary)
                } else if !snapshot.catalog_ready {
                    Text("The backend has no reviewed demo products yet. Add the real package facts to its private catalog before testing a comparison.")
                        .font(.caption).foregroundStyle(.orange)
                }
                ForEach(snapshot.products) { product in
                    Text(product.name).font(.caption).foregroundStyle(.secondary)
                }
            }
            if let error = conversation.error {
                Text(error).foregroundStyle(.orange).textSelection(.enabled)
            }
        }
    }
}
