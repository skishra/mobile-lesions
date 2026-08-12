import SwiftUI

struct ContentView: View {
    @StateObject private var camera = CameraManager()

    var body: some View {
        ZStack(alignment: .bottom) {
            if camera.isAuthorized {
                CameraPreview(session: camera.session)
                    .ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("Camera access is required.\nEnable it in Settings.")
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.white)
            }

            overlay
        }
        .onAppear { camera.start() }
        .onDisappear { camera.stop() }
    }

    private var overlay: some View {
        VStack(alignment: .leading, spacing: 6) {
            if camera.predictions.isEmpty {
                Text("Point the camera at a lesion…")
                    .foregroundStyle(.white.opacity(0.7))
            } else {
                ForEach(camera.predictions) { p in
                    HStack {
                        Text(p.label.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.headline)
                        Spacer()
                        Text(String(format: "%.1f%%", p.confidence * 100))
                            .font(.headline.monospacedDigit())
                    }
                    .foregroundStyle(.white)
                }
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.black.opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .padding()
    }
}

#Preview {
    ContentView()
}
