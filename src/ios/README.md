# Skin-Lesion Classifier — iOS Foundation

Live camera feed → fine-tuned timm MobileNetV4 (via Core ML) → top-3 labels on screen.

## Files
- `convert_to_coreml.py` — one-time: PyTorch/timm weights → `SkinLesionClassifier.mlpackage`
- `SkinLesionApp.swift` — app entry point
- `ContentView.swift` — preview + prediction overlay
- `CameraPreview.swift` — SwiftUI wrapper over the camera preview layer
- `CameraManager.swift` — capture session + Vision/Core ML inference

## Step 1 — Convert the model
Edit the marked constants at the top of `convert_to_coreml.py` (`MODEL_NAME`,
`WEIGHTS_PATH`, and especially `CLASS_LABELS` — these must be in the exact index
order your model was trained on), then:

```bash
pip install "timm>=1.0" torch "coremltools>=7.2"
python convert_to_coreml.py
```

This prints the input size and normalization it read from timm, and writes
`SkinLesionClassifier.mlpackage`. Before trusting it, uncomment the block at the
bottom to confirm Core ML's output matches PyTorch on a sample image.

## Step 2 — Build the app
1. New Xcode project → **iOS App**, SwiftUI, name it (e.g.) `SkinLesion`.
2. Delete the template `ContentView.swift`, then drag in all five files here
   plus `SkinLesionClassifier.mlpackage`. Tick **Copy items** and your app target.
   (Xcode auto-compiles the `.mlpackage` to `.mlmodelc` in the bundle — that's
   what `CameraManager` loads.)
3. Add the camera permission string. In the target's **Info** tab add
   **Privacy - Camera Usage Description** (`NSCameraUsageDescription`) with a
   sentence like *"Used to analyze skin lesions with the camera."* The app
   crashes on launch without it.
4. Run on a **real device** — the camera and Neural Engine don't exist in the
   Simulator.

## Things to get right (matched to your pipeline)
- **Class order.** The convert script reads labels from `train.csv` the same way
  `dataset.py` does (every column except `image`), so it stays in sync. For ISIC
  2019 this is 8 classes — `UNK` is dropped as zero-variance in `splitter.py`,
  `SCC` is kept. Point `TRAIN_CSV` at your actual split file.
- **Resolution.** Defaults to 320×320 — your val/test resolution and the one the
  best checkpoint was selected on. Set `INPUT_SIZE = 256` (train res) for a
  faster model and compare.
- **Crop mode.** `CameraManager` uses `.scaleFill`, matching your `Resize((N,N))`
  square squash. Don't change it to `.centerCrop` unless you also change training.
- **Preprocessing is baked in.** ImageNet normalization + softmax live inside the
  model; Vision only resizes. Don't re-normalize on the Swift side.
- **Orientation.** Frames use `.right` (back camera, portrait). If labels seem
  off and you suspect the image is sideways, change that value first.
- **Throttle.** Inference runs every 3rd frame and skips while busy. Adjust
  `runEveryNthFrame` for the accuracy/latency/battery trade-off you want.

## Note
This is a solid technical prototype. If it ever moves toward real diagnostic use
rather than experimentation, on-device skin-lesion classifiers fall under
medical-device regulation and need clinical validation and appropriate
disclaimers before anyone relies on the output.
