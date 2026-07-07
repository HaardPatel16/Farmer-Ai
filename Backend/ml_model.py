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

import base64
import io
import json
import os
import threading

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_small

# Pillow 9.1 introduced PIL.Image.Resampling.* and 10.0 removed the
# top-level aliases (Image.LANCZOS, Image.BICUBIC, …). Resolve once at
# import time so the rest of the file doesn't care which install is on
# the system — if Resampling is missing (Pillow < 9.1), fall back to the
# legacy attribute, which still exists in that range.
try:
    _LANCZOS = Image.Resampling.LANCZOS
except AttributeError:
    _LANCZOS = Image.LANCZOS

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
_model_lock = threading.Lock()

# rembg model used for the cutout. "u2net" is the general-purpose default,
# ~170 MB, works well for plant leaves against natural backgrounds.
# Alternatives if you ever want to swap: "isnet-general-use" (newer, often
# crisper edges), "u2netp" (smaller/faster, lower quality). Weights are
# auto-downloaded to ~/.u2net/ on first use.
_BG_MODEL_NAME = "u2net"
_bg_session = None
_bg_session_lock = threading.Lock()


def _get_bg_session():
    """Lazily creates and caches the rembg ONNX session. First call triggers
    a one-time model download (~170 MB) — warm it up at server startup
    (see main.py's _warm_diagnosis_stack, invoked from the lifespan hook)
    so the first /diagnose request isn't slow. The lock prevents two
    concurrent first-time callers from each spinning up a separate ONNX
    session (rembg's new_session is not safe to invoke concurrently)."""
    global _bg_session
    if _bg_session is not None:
        return _bg_session
    with _bg_session_lock:
        if _bg_session is None:
            from rembg import new_session  # local import — see header note
            _bg_session = new_session(_BG_MODEL_NAME)
            # rembg's BaseSession already auto-picks CUDAExecutionProvider
            # over CPUExecutionProvider when onnxruntime-gpu + a CUDA GPU
            # are both present (see rembg.sessions.base.BaseSession.__init__)
            # — but it does so silently. Mirror the explicit device print
            # used by embeddings.py / ml_model.py's classifier so an
            # operator can tell from the logs whether background removal
            # is actually running on GPU or fell back to CPU.
            try:
                active_provider = _bg_session.inner_session.get_providers()[0]
            except Exception:
                active_provider = "unknown"
            device_label = "GPU" if "CUDA" in active_provider or "ROCM" in active_provider else "CPU"
            print(f"[ml_model] rembg background remover running on {device_label} ({active_provider})")
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


def _pick_device() -> torch.device:
    """Return a torch.device for the leaf classifier. CUDA when available,
    CPU otherwise. Mirrors the helper in embeddings.py so both ML stacks
    answer the GPU/CPU question identically."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Resolved once on first model load and reused for the input tensor in
# predict_top_k(). Without a single source of truth, a model on CUDA and
# input on CPU silently raises RuntimeError on .forward().
_device = _pick_device()


def _load_model() -> nn.Module:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Leaf diagnosis model not found at '{MODEL_PATH}'. "
            f"Place the trained .pth file there (filename: '{os.path.basename(MODEL_PATH)}')."
        )
    model = convnext_small()
    model.classifier[2] = nn.Linear(768, NUM_CLASSES)
    # weights_only=True refuses pickle code execution; the checkpoint is a
    # plain state_dict, so this is the safe path. Fall back to the legacy
    # loader only if a future checkpoint format requires it, and log loudly
    # so the operator knows untrusted bytes are being deserialized.
    try:
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    except Exception as e:
        print(
            f"[ml_model] torch.load(weights_only=True) failed ({type(e).__name__}: {e}); "
            "retrying with weights_only=False. Only use trusted .pth files."
        )
        state_dict = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.to(_device)
    model.eval()
    print(f"[ml_model] ConvNeXt-Small running on {str(_device).upper()}")
    return model


def get_model() -> nn.Module:
    """Lazily loads the model once and caches it for subsequent requests.
    The lock prevents two concurrent first-time /diagnose requests from each
    paying ~10s of torch.load + state-dict copy and double-loading weights."""
    global _model
    if _model is not None:
        return _model
    with _model_lock:
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


def _encode_preview_webp(image: Image.Image, max_side: int = 512, quality: int = 78) -> bytes:
    """Encode the classifier's input image as a small WebP preview suitable
    for sending back to the chat UI. We downscale to a max side of 512 px
    so the base64 payload stays in the ~30–80 KB range — the frontend only
    renders this at ~220 px wide, so anything bigger is wasted bytes. WebP
    quality 78 is roughly visually lossless for foliage on a white field
    and compresses ~4× better than the equivalent JPEG."""
    preview = image.copy()
    preview.thumbnail((max_side, max_side), _LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue()


def predict_top_k(image_bytes: bytes, k: int = 3, remove_bg: bool = True) -> dict:
    """
    Runs the classifier on a single image and returns:

        {
            "predictions": list[{"label": str, "confidence": float}],  # top_k, descending
            "processed_image_b64": str | None,   # WebP bytes, base64-encoded; None on encode failure
            "bg_removed": bool,                  # True iff rembg actually ran successfully
        }

    When remove_bg is True (default), rembg strips the background first so
    the model focuses on the leaf. If rembg fails for any reason
    (download failure, corrupt session, exotic image format), we fall
    back to the raw image rather than fail the whole request, and
    bg_removed comes back False so the frontend can decide whether to
    advertise "Background removed" to the user.

    processed_image_b64 is the *same* image the classifier actually saw
    (post-rembg if it ran, the raw decode otherwise), downscaled and
    re-encoded as WebP — the chat UI swaps the user's upload thumbnail to
    this once the response lands, so the farmer sees what the model saw.

    Model-file existence is checked lazily by get_model() — a missing
    .pth raises FileNotFoundError, which the route maps to 503.
    """
    bg_removed = False
    if remove_bg:
        try:
            image = _remove_background(image_bytes)
            bg_removed = True
        except Exception as e:
            print(f"[ml_model] background removal failed, using raw image: {type(e).__name__}: {e}")
            image = _open_raw_rgb(image_bytes)
    else:
        image = _open_raw_rgb(image_bytes)

    input_tensor = _PREPROCESS(image).unsqueeze(0).to(_device)

    model = get_model()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probs, k=min(k, NUM_CLASSES))

    # .item() on the 0-dim LongTensor — current PyTorch happens to
    # accept a 0-dim tensor as a list index via __index__, but that's
    # an implementation detail to not rely on. .item() also future-proofs
    # if NUM_CLASSES > 2**31 (it won't, but be explicit).
    predictions = [
        {"label": CLASS_NAMES[idx.item()], "confidence": round(prob.item(), 4)}
        for prob, idx in zip(top_probs, top_indices)
    ]

    # Encode the preview AFTER inference rather than before so the encode
    # failure mode (rare — basically only out-of-memory) doesn't take down
    # the diagnosis itself.
    try:
        webp = _encode_preview_webp(image)
        processed_image_b64 = base64.b64encode(webp).decode("ascii")
    except Exception as e:
        print(f"[ml_model] preview encode failed: {type(e).__name__}: {e}")
        processed_image_b64 = None

    return {
        "predictions": predictions,
        "processed_image_b64": processed_image_b64,
        "bg_removed": bg_removed,
    }
