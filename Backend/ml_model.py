"""
Loads the ConvNeXt-Small leaf disease classifier from ML model/Latest_Plant_Model_.pth
and exposes predict_top_k() for inference on a single uploaded image.

The checkpoint is a raw state_dict (no architecture or class names embedded),
so the architecture is reconstructed here and the class names are kept in a
separate, hand-maintained list (CLASS_NAMES below) that must match the
training-time ImageFolder class order exactly.
"""

import io
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small

# PIL allows ~89 megapixel decoded images by default — enough for a
# malicious 7 MB compressed PNG to decode to hundreds of millions of
# pixels and OOM the server. Lower the ceiling to 25 MP (≈ 5000x5000),
# more than enough headroom for any real leaf photo.
Image.MAX_IMAGE_PIXELS = 25_000_000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "ML model", "Latest_Plant_Model_.pth")

NUM_CLASSES = 78

# TODO: replace with the real ordered class names from training (placeholders
# for now — predictions will run but the names won't mean anything until
# these are filled in).
CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# Inference preprocessing must match the training notebook's eval_transforms
# exactly — anything else feeds the model a distribution shift and silently
# tanks accuracy. Training used Resize(256) → CenterCrop(224), not a direct
# Resize((224, 224)) stretch, so we replicate that here. ImageNet mean/std
# match torchvision's pretrained ConvNeXt-Small expectations.
_PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

_model = None


def _load_model() -> nn.Module:
    model = convnext_small()
    model.classifier[2] = nn.Linear(768, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def get_model() -> nn.Module:
    """Lazily loads the model once and caches it for subsequent requests."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def predict_top_k(image_bytes: bytes, k: int = 3) -> list[dict]:
    """
    Runs the classifier on a single image and returns the top_k predictions
    as a list of {"label": str, "confidence": float} dicts, sorted by
    confidence descending.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = _PREPROCESS(image).unsqueeze(0)

    model = get_model()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(k, NUM_CLASSES))

    return [
        {"label": CLASS_NAMES[idx], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]
