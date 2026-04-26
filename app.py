"""
app.py — Brain Tumour Detection Web App
Uses the dual-branch DualBranchTumorClassifier for inference.
"""

import io
import os
from pathlib import Path

import torch
from flask import Flask, render_template, request, jsonify, redirect, url_for
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
        pretrained=False,
    )
    if Path(CHECKPOINT).exists():
        state = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(state)
        print(f"[app] Loaded checkpoint: {CHECKPOINT}")
    else:
        print(f"[app] WARNING — checkpoint not found at '{CHECKPOINT}'. "
              f"Run train.py first. Predictions will be random.")
    model.to(DEVICE).eval()
    return model


model = load_model()

# ──────────────────────────────────────────────
#  Inference transform
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
    rgb    = pil_img.convert("RGB")
    tensor = infer_tf(rgb).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    pred_idx   = probs.argmax().item()
    confidence = probs[pred_idx].item()
    label      = LABELS[pred_idx]
    all_probs  = {LABELS[i]: round(probs[i].item() * 100, 2) for i in range(NUM_CLASSES)}

    return {
        "label":      label,
        "confidence": round(confidence * 100, 2),
        "all_probs":  all_probs,
    }


# ── Routes ─────────────────────────────────────

@app.route("/")
def index():
    return render_template("MainPage.html")


@app.route("/detect")
def detect():
    return render_template("Diseasedet.html")


@app.route("/uimg", methods=["GET", "POST"])
def uimg():
    """GET — show upload form. POST — run prediction and show result."""
    if request.method == "GET":
        return render_template("uimg.html")

    # POST — handle file upload and predict
    if "file" not in request.files:
        return render_template("error.html", error="No file uploaded"), 400

    file = request.files["file"]
    if file.filename == "":
        return render_template("error.html", error="No file selected"), 400

    try:
        img_bytes = file.read()
        pil_img   = Image.open(io.BytesIO(img_bytes))
        result    = predict_image(pil_img)
    except Exception as exc:
        return render_template("error.html", error=str(exc)), 500

    return render_template(
        "pred.html",
        result         = result["label"],
        confidence     = result["confidence"],
        all_probs      = result["all_probs"],
        uploaded_image = None,
        gradcam_image  = None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """JSON API endpoint for programmatic access."""
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


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", error="Page not found"), 404


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", error="Server error"), 500


# ──────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)