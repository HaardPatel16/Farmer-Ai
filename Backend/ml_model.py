"""
Loads the ConvNeXt-Small leaf disease classifier and exposes predict_top_k()
for inference on a single uploaded image.

The .pth checkpoint is a raw state_dict (no architecture or class names
embedded), so the architecture is reconstructed here and class names are
loaded from a sibling JSON file ('ML model/class_names.json') whose order
must match the training-time ImageFolder class order exactly. If that JSON
isn't present, predict_top_k() still runs but returns generic placeholder
labels (class_0…class_N-1) — the model file itself is the hard requirement,
the names file is a soft requirement that turns those placeholders into
real disease names.
"""

import io
import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small

# rembg = https://github.com/danielgatis/rembg — U²-Net-based background
# removal. Run before the ConvNeXt preprocess so the classifier sees only
# the leaf, not the soil/hand/desk behind it. Imported lazily (inside
# _get_bg_session) so a broken onnxruntime install doesn't take the whole
# backend down on import — /chat, /weather, /market-price all still work
# without rembg; only /diagnose with remove_bg=True needs it.

# PIL allows ~89 megapixel decoded images by default — enough for a
# malicious 7 MB compressed PNG to decode to hundreds of millions of
# pixels and OOM the server. Lower the ceiling to 25 MP (≈ 5000x5000),
# more than enough headroom for any real leaf photo.
Image.MAX_IMAGE_PIXELS = 25_000_000

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "ML model")
MODEL_PATH = os.path.join(MODEL_DIR, "best_crop_model.pth")
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, "class_names.json")

NUM_CLASSES = 78


def _load_class_names() -> list[str]:
    """Reads the training-time class-name list from class_names.json next
    to the .pth file. The JSON must be a flat array of strings whose length
    equals NUM_CLASSES — the order must mirror the ImageFolder class index
    order used during training (alphabetical by directory name, by default).

    If the file is missing or malformed, returns placeholder labels
    ('class_0' … 'class_N-1') and logs a clear warning, so inference still
    runs end-to-end (useful for smoke-testing the pipeline) but the
    operator knows the labels are not yet trustworthy.
    """
    if not os.path.exists(CLASS_NAMES_PATH):
        print(
            f"[ml_model] class_names.json not found at '{CLASS_NAMES_PATH}'. "
            "Using placeholder labels — fill in the real ordered class names "
            "(matching your ImageFolder training order) for meaningful "
            "diagnosis output."
        )
        return [f"class_{i}" for i in range(NUM_CLASSES)]

    try:
        with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[ml_model] failed to read class_names.json: {type(e).__name__}: {e}. Using placeholders.")
        return [f"class_{i}" for i in range(NUM_CLASSES)]

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        print("[ml_model] class_names.json must be a JSON array of strings. Using placeholders.")
        return [f"class_{i}" for i in range(NUM_CLASSES)]

    if len(data) != NUM_CLASSES:
        print(
            f"[ml_model] class_names.json has {len(data)} entries but the "
            f"classifier head expects {NUM_CLASSES}. Using placeholders."
        )
        return [f"class_{i}" for i in range(NUM_CLASSES)]

    return data


CLASS_NAMES = _load_class_names()

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

# rembg model used for the cutout. "u2net" is the general-purpose default,
# ~170 MB, works well for plant leaves against natural backgrounds.
# Alternatives if you ever want to swap: "isnet-general-use" (newer, often
# crisper edges), "u2netp" (smaller/faster, lower quality). Weights are
# auto-downloaded to ~/.u2net/ on first use.
_BG_MODEL_NAME = "u2net"
_bg_session = None


def _get_bg_session():
    """Lazily creates and caches the rembg ONNX session. First call triggers
    a one-time model download (~170 MB) — warm it up at server startup
    (see main.py _warm_bg_remover_on_startup) so the first /diagnose
    request isn't slow."""
    global _bg_session
    if _bg_session is None:
        from rembg import new_session  # local import — see header note
        _bg_session = new_session(_BG_MODEL_NAME)
    return _bg_session


def _remove_background(image_bytes: bytes) -> Image.Image:
    """Runs rembg on the raw upload bytes and composites the foreground
    onto a solid white background. The classifier's preprocess pipeline
    expects RGB (3 channels), so we cannot return rembg's native RGBA
    directly — white was chosen over black because most ImageNet-style
    training data assumes light-dominant backgrounds."""
    from rembg import remove  # local import — see header note
    cutout_bytes = remove(image_bytes, session=_get_bg_session())
    cutout = Image.open(io.BytesIO(cutout_bytes)).convert("RGBA")
    background = Image.new("RGB", cutout.size, (255, 255, 255))
    background.paste(cutout, mask=cutout.split()[3])
    return background


def warm_bg_remover() -> None:
    """Forces the rembg session to load now (and download weights if this
    is the first ever run) so the first /diagnose request doesn't pay
    the cost. Safe to call from a daemon thread at server startup."""
    _get_bg_session()


def _load_model() -> nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Leaf diagnosis model not found at '{MODEL_PATH}'. "
            f"Place the trained .pth file there (filename: '{os.path.basename(MODEL_PATH)}')."
        )
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


def warm_classifier() -> None:
    """Forces the ConvNeXt-Small weights to load now (paying ~5-10 s of
    torch.load + state-dict copy) so the first /diagnose request doesn't
    eat that cost. Safe to call from a daemon thread at server startup;
    raises FileNotFoundError if the .pth file is missing, which the
    caller should catch and log."""
    get_model()


def _open_raw_rgb(image_bytes: bytes) -> Image.Image:
    """Decode bytes → RGB. Raises ValueError on un-decodable input so the
    caller can map it to a clean 400 instead of leaking the underlying
    PIL exception text to the client."""
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Image could not be decoded ({type(e).__name__}).") from e


def predict_top_k(image_bytes: bytes, k: int = 3, remove_bg: bool = True) -> list[dict]:
    """
    Runs the classifier on a single image and returns the top_k predictions
    as a list of {"label": str, "confidence": float} dicts, sorted by
    confidence descending.

    When remove_bg is True (default), rembg strips the background first so
    the model focuses on the leaf. If rembg fails for any reason
    (download failure, corrupt session, exotic image format), we fall
    back to the raw image rather than fail the whole request.

    Model-file existence is checked lazily by get_model() — a missing
    .pth raises FileNotFoundError, which the route maps to 503.
    """
    if remove_bg:
        try:
            image = _remove_background(image_bytes)
        except Exception as e:
            print(f"[ml_model] background removal failed, using raw image: {type(e).__name__}: {e}")
            image = _open_raw_rgb(image_bytes)
    else:
        image = _open_raw_rgb(image_bytes)

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
