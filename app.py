"""
app.py — Brain Tumour Detection Web App
Uses the dual-branch DualBranchTumorClassifier for inference.
Preserves all existing routes and template names.
"""

import io
import os
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision.transforms import v2

from model_architecture import DualBranchTumorClassifier

# ──────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────

CHECKPOINT  = os.getenv("MODEL_PATH", "models/best_model.pt")
NUM_CLASSES = 3
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS = {0: "Meningioma", 1: "Glioma", 2: "Pituitary"}

# ──────────────────────────────────────────────
#  Model — loaded once at startup
# ──────────────────────────────────────────────

def load_model() -> DualBranchTumorClassifier:
    model = DualBranchTumorClassifier(
        num_classes=NUM_CLASSES,
        pretrained=False,   # weights come from checkpoint
    )
    if Path(CHECKPOINT).exists():
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state)
        print(f"[app] Loaded checkpoint: {CHECKPOINT}")
    else:
        print(f"[app] WARNING — checkpoint not found at '{CHECKPOINT}'. "
              f"Run train.py first.  Predictions will be random.")
    model.to(DEVICE).eval()
    return model


model = load_model()

# ──────────────────────────────────────────────
#  Inference transform  (same as val_tf in train.py)
# ──────────────────────────────────────────────

infer_tf = v2.Compose([
    v2.Resize((IMG_SIZE, IMG_SIZE)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.485, 0.456, 0.406),
                 std =(0.229, 0.224, 0.225)),
])

# ──────────────────────────────────────────────
#  Flask app
# ──────────────────────────────────────────────

app = Flask(__name__)


def predict_image(pil_img: Image.Image) -> dict:
    """Run inference on a PIL image. Returns label + confidence dict."""
    rgb   = pil_img.convert("RGB")
    tensor = infer_tf(rgb).unsqueeze(0).to(DEVICE)   # (1, 3, H, W)

    with torch.no_grad():
        logits = model(tensor)                        # (1, num_classes)
        probs  = torch.softmax(logits, dim=1)[0]     # (num_classes,)

    pred_idx    = probs.argmax().item()
    confidence  = probs[pred_idx].item()
    label       = LABELS[pred_idx]

    all_probs = {LABELS[i]: round(probs[i].item(), 4) for i in range(NUM_CLASSES)}

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "all_probs":  all_probs,
    }


# ── Routes ─────────────────────────────────────

@app.route("/")
def index():
    return render_template("MainPage.html")


@app.route("/detect", methods=["GET", "POST"])
def detect():
    return render_template("Diseasedet.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Accepts a multipart image upload, returns prediction JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes))
        result    = predict_image(pil_img)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(result)


@app.route("/result")
def result():
    """Render prediction result page (populated via JS fetch of /predict)."""
    return render_template("pred.html")


@app.route("/app-info")
def app_info():
    return render_template("app.html")


@app.route("/uimg")
def uimg():
    return render_template("uimg.html")


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Server error"), 500


# ──────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)