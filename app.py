import flask
from io import BytesIO
import uuid

import numpy as np
import torch
from torch import argmax, load
from torch import device as DEVICE
from torch.cuda import is_available
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor, Resize
from werkzeug.utils import secure_filename
import os

from model_architecture import build_resnet50_model

UPLOAD_FOLDER = os.path.join('static', 'photos')
MODEL_PATH = os.path.join('models', 'bt_resnet50_model.pt')
app = flask.Flask(__name__, template_folder='templates')
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']

device = "cuda" if is_available() else "cpu"
resnet_model = None


def load_model():
    global resnet_model

    if resnet_model is not None:
        return resnet_model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Missing trained model file: {MODEL_PATH}. "
            "Train or download bt_resnet50_model.pt and place it in the models folder."
        )

    model = build_resnet50_model(weights=None, attention="cbam")
    model.to(device)
    model.load_state_dict(load(MODEL_PATH, map_location=DEVICE(device)))
    model.eval()
    resnet_model = model
    return resnet_model

def preprocess_image(image_bytes):
    transform = Compose([
        Resize((512, 512)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    return transform(img).unsqueeze(0)


def get_prediction(image_bytes):
    model = load_model()
    tensor = preprocess_image(image_bytes=image_bytes)
    with torch.no_grad():
        y_hat = model(tensor.to(device))
    class_id = argmax(y_hat, dim=1)
    return str(int(class_id)), LABELS[int(class_id)]


def save_upload(image_bytes, filename):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    extension = filename.rsplit('.', 1)[1].lower()
    safe_stem = secure_filename(filename.rsplit('.', 1)[0]) or "upload"
    saved_name = f"{safe_stem}-{uuid.uuid4().hex[:8]}.{extension}"
    saved_path = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
    Image.open(BytesIO(image_bytes)).convert('RGB').save(saved_path)
    return saved_name


def generate_gradcam(image_bytes, class_id):
    model = load_model()
    tensor = preprocess_image(image_bytes=image_bytes).to(device)
    activations = {}
    gradients = {}

    def forward_hook(module, inputs, output):
        activations["value"] = output

        def gradient_hook(gradient):
            gradients["value"] = gradient

        output.register_hook(gradient_hook)

    hook = model.layer4.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model(tensor)
        score = output[0, class_id]
        score.backward()
    finally:
        hook.remove()

    activation = activations["value"].detach()[0]
    gradient = gradients["value"].detach()[0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    heatmap = torch.relu((weights * activation).sum(dim=0))
    heatmap = heatmap - heatmap.min()
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.cpu().numpy()
    heatmap_image = Image.fromarray(np.uint8(heatmap * 255)).resize((512, 512), Image.BILINEAR)
    heatmap = np.asarray(heatmap_image, dtype=np.float32) / 255.0

    original = Image.open(BytesIO(image_bytes)).convert('RGB').resize((512, 512))
    original_array = np.asarray(original, dtype=np.float32)
    heat_color = np.zeros_like(original_array)
    heat_color[..., 0] = 255
    heat_color[..., 1] = 120 * heatmap
    alpha = 0.55 * heatmap[..., None]
    overlay = original_array * (1 - alpha) + heat_color * alpha

    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    gradcam_name = f"gradcam-{uuid.uuid4().hex[:8]}.jpg"
    gradcam_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_name)
    Image.fromarray(np.uint8(np.clip(overlay, 0, 255))).save(gradcam_path)
    return gradcam_name

@app.route('/', methods=['GET'])
def main():
    return flask.render_template('DiseaseDet.html')

@app.route("/uimg",methods=['GET','POST'])
def uimg():
    if flask.request.method == 'GET':
        return flask.render_template('uimg.html')
    if flask.request.method == 'POST':
        file = flask.request.files['file']
        if not file or not allowed_file(file.filename):
            return flask.render_template('error.html', message='Please upload a PNG, JPG, JPEG, or GIF image.'), 400
        img_bytes = file.read()   
        try:
            class_id, class_name = get_prediction(img_bytes)
            upload_name = save_upload(img_bytes, file.filename)
            gradcam_name = generate_gradcam(img_bytes, int(class_id))
        except FileNotFoundError as error:
            return flask.render_template('error.html', message=str(error)), 503
        return flask.render_template(
            'pred.html',
            result=class_name,
            uploaded_image=upload_name,
            gradcam_image=gradcam_name,
        )
      
@app.errorhandler(500)
def server_error(error):
    return flask.render_template('error.html', message='Something went wrong. Please try again.'), 500

if __name__ == '__main__':
    app.run(debug=True)
