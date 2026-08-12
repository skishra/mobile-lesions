"""
convert_to_coreml.py
--------------------
Convert the fine-tuned timm MobileNetV4 skin-lesion model to a Core ML .mlpackage.

Matched to your training/eval pipeline (trainer.py / evaluator.py / dataset.py):
  - Class order is read straight from train.csv, exactly like dataset.py builds it
    (every column except 'image'). For ISIC 2019 this is 8 classes: UNK is dropped
    as zero-variance in splitter.py, SCC is kept.
  - ImageNet mean/std normalization (matches your Normalize()).
  - 320x320 input: the resolution your val/test transforms use and the one your
    best checkpoint was selected on. Set INPUT_SIZE = 256 for a lighter model.
  - Your transforms squash to a square (aspect distorted), so the iOS side must
    use .scaleFill (already set in CameraManager.swift).

    pip install "timm>=1.0" torch "coremltools>=7.2"
"""

import pandas as pd
import torch
import torch.nn as nn
import timm
import coremltools as ct

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
MODEL_NAME   = "mobilenetv4_conv_medium.e500_r256_in1k"          # same string you trained with
WEIGHTS_PATH = "best_model_202607170159.pt"                                   # the state_dict your Trainer saved
TRAIN_CSV    = "src/dataset/archive/after_split/train.csv"       # used to read the class order
OUTPUT_PATH  = "SkinLesionClassifier.mlpackage"                  # keep in sync with the iOS side
INPUT_SIZE   = 320   # your val/test/eval resolution. Use 256 (train res) for a lighter/faster model.
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Human-readable names for the ISIC 2019 short codes (display only; order is what matters).
CODE_TO_NAME = {
    "MEL": "Melanoma",
    "NV":  "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK":  "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF":  "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
    "UNK": "Unknown",
}

# Read the class order the SAME way dataset.py does, so it can never drift from
# what the model was trained on.
codes = [c for c in pd.read_csv(TRAIN_CSV, nrows=0).columns if c != "image"]
CLASS_LABELS = [CODE_TO_NAME.get(c, c) for c in codes]
NUM_CLASSES  = len(CLASS_LABELS)
print(f"{NUM_CLASSES} classes (index order): {list(zip(codes, CLASS_LABELS))}")

# Rebuild the architecture and load your fine-tuned weights (a plain state_dict).
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
state = torch.load(WEIGHTS_PATH, map_location="cpu")
if isinstance(state, dict) and "state_dict" in state:      # just in case
    state = state["state_dict"]
state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=True)
model.eval()

# Bake normalization + softmax into the model. The Core ML image input (below)
# scales pixels to 0..1; this wrapper then applies (x - mean) / std per channel,
# matching your Normalize() exactly, and returns real probabilities.
class Wrapped(nn.Module):
    def __init__(self, m, mean, std):
        super().__init__()
        self.m = m
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):                 # x arrives in 0..1
        x = (x - self.mean) / self.std
        return self.m(x).softmax(dim=1)

wrapped = Wrapped(model, IMAGENET_MEAN, IMAGENET_STD).eval()

example = torch.rand(1, 3, INPUT_SIZE, INPUT_SIZE)
with torch.no_grad():
    traced = torch.jit.trace(wrapped, example)

# scale = 1/255 turns the 0..255 image the app feeds in into 0..1.
image_input = ct.ImageType(
    name="image",
    shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
    scale=1.0 / 255.0,
    bias=[0.0, 0.0, 0.0],
    color_layout=ct.colorlayout.RGB,
)

mlmodel = ct.convert(
    traced,
    inputs=[image_input],
    classifier_config=ct.ClassifierConfig(class_labels=CLASS_LABELS),
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,
)

mlmodel.short_description = "Fine-tuned MobileNetV4 skin-lesion classifier (ISIC 2019)"
mlmodel.input_description["image"] = f"{INPUT_SIZE}x{INPUT_SIZE} RGB image"
mlmodel.save(OUTPUT_PATH)
print(f"Saved {OUTPUT_PATH}")

# Sanity-check that Core ML agrees with PyTorch before shipping. Because your
# eval squashes to a square, resize to (INPUT_SIZE, INPUT_SIZE) with NO crop here too.
#   from PIL import Image
#   img = Image.open("test_lesion.jpg").convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
#   print(mlmodel.predict({"image": img}))
