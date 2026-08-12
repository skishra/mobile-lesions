import AVFoundation
import Vision
import CoreML
import Combine

struct Prediction: Identifiable {
    let id = UUID()
    let label: String
    let confidence: Float
}

/// Owns the camera session and runs the Core ML classifier on the live feed.
final class CameraManager: NSObject, ObservableObject {

    // UI-observable state
    @Published var predictions: [Prediction] = []
    @Published var isAuthorized = false

    // Capture
    let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "camera.session")
    private let videoQueue   = DispatchQueue(label: "camera.video")

    // Inference
    private var visionRequests: [VNRequest] = []
    private var busy = false                 // skip frames while one is in flight
    private var frameIndex = 0
    private let runEveryNthFrame = 3         // ~throttle; tune for your device
    private var configured = false

    /// Must match the .mlpackage filename you drag into the Xcode project.
    private let modelResourceName = "SkinLesionClassifier"

    // MARK: - Lifecycle

    func start() {
        requestAuthorization { [weak self] granted in
            guard let self, granted else { return }
            self.sessionQueue.async {
                self.configureSessionIfNeeded()
                self.loadModelIfNeeded()
                if !self.session.isRunning { self.session.startRunning() }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    // MARK: - Permission

    private func requestAuthorization(completion: @escaping (Bool) -> Void) {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            DispatchQueue.main.async { self.isAuthorized = true }
            completion(true)
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { granted in
                DispatchQueue.main.async { self.isAuthorized = granted }
                completion(granted)
            }
        default:
            DispatchQueue.main.async { self.isAuthorized = false }
            completion(false)
        }
    }

    // MARK: - Session

    private func configureSessionIfNeeded() {
        guard !configured else { return }
        configured = true

        session.beginConfiguration()
        session.sessionPreset = .high

        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
            let input  = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            session.commitConfiguration()
            return
        }
        session.addInput(input)

        videoOutput.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: videoQueue)
        if session.canAddOutput(videoOutput) { session.addOutput(videoOutput) }

        session.commitConfiguration()
    }

    // MARK: - Model

    private func loadModelIfNeeded() {
        guard visionRequests.isEmpty else { return }
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all                 // let it use the Neural Engine
            guard let url = Bundle.main.url(forResource: modelResourceName,
                                            withExtension: "mlmodelc") else {
                print("⚠️ \(modelResourceName).mlmodelc not found in bundle.")
                return
            }
            let mlModel     = try MLModel(contentsOf: url, configuration: config)
            let visionModel = try VNCoreMLModel(for: mlModel)

            let request = VNCoreMLRequest(model: visionModel) { [weak self] req, _ in
                self?.handleResults(req.results)
            }
            // Your training/eval transforms use Resize((N,N)) — a square squash
            // that distorts aspect ratio, no crop. .scaleFill reproduces that.
            // (Use .centerCrop only if you switch training to shorter-side + crop.)
            request.imageCropAndScaleOption = .scaleFill
            visionRequests = [request]
        } catch {
            print("⚠️ Failed to load model: \(error)")
        }
    }

    private func handleResults(_ results: [VNObservation]?) {
        defer { busy = false }
        guard let observations = results as? [VNClassificationObservation] else { return }
        let top = observations
            .sorted { $0.confidence > $1.confidence }
            .prefix(3)
            .map { Prediction(label: $0.identifier, confidence: $0.confidence) }
        DispatchQueue.main.async { self.predictions = Array(top) }
    }
}

// MARK: - Frame delegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput,
                       didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        frameIndex &+= 1
        guard frameIndex % runEveryNthFrame == 0 else { return }
        guard !busy, !visionRequests.isEmpty else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        busy = true
        // .right is correct for the back camera in portrait. If predictions look
        // rotated, this is the first thing to change.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: .right,
                                            options: [:])
        do {
            try handler.perform(visionRequests)
        } catch {
            busy = false
        }
    }
}
