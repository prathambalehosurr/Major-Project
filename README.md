# 🧠 Brain Tumor Detection — End to End

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Val F1](https://img.shields.io/badge/Val%20F1-98.67%25-brightgreen?style=for-the-badge)](https://github.com/prathambalehosurr/brain-tumor-detection)
[![License](https://img.shields.io/badge/License-MIT-1abc9c?style=for-the-badge)](LICENSE.md)

> A full **end-to-end deep learning web application** that classifies brain tumors in MRI scans into three tumor types using a **dual-branch architecture** — ResNet50+CBAM and ConvNeXt-Small with feature-level fusion — achieving **98.67% validation F1** — deployed as a live Flask web app.

[🔙 Back to Main Repository](https://github.com/prathambalehosurr)

---

## ⚠️ Medical Disclaimer

> **This tool is for educational and research purposes only.** It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist or medical professional for clinical decisions.

---

## 📌 Table of Contents

- [About the Project](#-about-the-project)
- [How It Works](#-how-it-works)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Model Performance](#-model-performance)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Tech Stack](#-tech-stack)
- [References & Citation](#-references--citation)

---

## 🔬 About the Project

Brain tumors are among the most critical conditions in medicine — accurate classification directly guides treatment decisions (surgery, radiation, chemotherapy). This project demonstrates how a **dual-branch deep learning architecture** combining **ResNet50+CBAM** and **ConvNeXt-Small** with feature-level fusion can achieve near-perfect classification accuracy on MRI scans.

The model is trained on the **Jun Cheng Figshare brain tumor dataset** (3,064 T1-weighted CE-MRI images from 233 patients) and deployed as a **Flask web application** where users can upload an MRI image and receive a real-time classification with confidence score.

**What this project covers:**

- Reading `.mat` (MATLAB) MRI files and extracting image data via `h5py`
- Data augmentation with real-time transformations using `torchvision.transforms.v2`
- Dual-branch architecture: ResNet50+CBAM ⊕ ConvNeXt-Small with feature-level fusion
- Two-stage training: head warm-up (frozen backbones) → full fine-tuning with differential LR
- Mixed-precision training with `torch.amp` for speed and memory efficiency
- Per-epoch validation metrics: accuracy, precision, recall, F1 (macro)
- Best-model checkpointing by validation F1
- Serving predictions via a Flask web app

---

## ⚙️ How It Works

```
User Uploads MRI Scan (.jpg / .png)
              │
              ▼
    Image Preprocessing
  (Resize 224×224 → Normalize → Tensor)
              │
         ┌────┴────┐
         ▼         ▼
  ResNet50+CBAM  ConvNeXt-Small
   2048-d feat    768-d feat
         └────┬────┘
              ▼
    Concatenate → 2816-d
              │
              ▼
       FusionHead MLP
    (LayerNorm → Dropout → Linear → GELU → Linear)
              │
              ▼
    3-Class Softmax Output
  ┌───────────┬─────────────┬────────────┐
  │  Glioma   │ Meningioma  │ Pituitary  │
  └───────────┴─────────────┴────────────┘
              │
              ▼
  Predicted Class + Confidence Score
       Displayed in Browser
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | Brain Tumor Dataset |
| **Author** | Jun Cheng |
| **Source** | [Figshare — DOI: 10.6084/m9.figshare.1512427](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427) |
| **Total Images** | 3,064 T1-weighted CE-MRI scans |
| **Patients** | 233 |
| **Format** | `.mat` (MATLAB) — loaded directly via `h5py` |
| **Task** | Multi-class classification (3 tumor types) |

### Class Distribution

| Class | Description | Slices |
|---|---|---|
| 🔴 **Glioma** | Arises from glial cells; most common & aggressive brain tumor | 1,426 |
| 🟡 **Meningioma** | Grows on membranes surrounding the brain; often benign | 708 |
| 🟢 **Pituitary** | Forms on the pituitary gland at the brain's base; usually slow-growing | 930 |
| **Total** | | **3,064** |

### Data Augmentation

| Technique | Purpose |
|---|---|
| Horizontal & Vertical Flip | Positional variance |
| Random Rotation (±15°) | Scan orientation variance |
| Brightness / Contrast Jitter | Scanner setting variance |
| Random Crop | Variable tumor scale |
| Normalization (ImageNet μ/σ) | Stable gradient flow |

---

## 🏗️ Model Architecture

This project uses a **dual-branch feature-fusion architecture** rather than a single backbone.

```
Input MRI Image (224 × 224 × 3)
          │
    ┌─────┴──────┐
    ▼            ▼
┌──────────┐  ┌──────────────┐
│ ResNet50 │  │ ConvNeXt-    │
│  + CBAM  │  │   Small      │
│          │  │              │
│ Channel  │  │ Depthwise    │
│ + Spatial│  │ separable    │
│ attention│  │ convolutions │
│          │  │              │
│ 2048-d   │  │ 768-d        │
│ features │  │ features     │
└──────────┘  └──────────────┘
    └─────┬──────┘
          ▼
   Concatenate (2816-d)
          │
          ▼
  ┌──────────────────┐
  │   FusionHead     │
  │  LayerNorm       │
  │  Dropout (0.4)   │
  │  Linear → GELU   │
  │  Dropout (0.2)   │
  │  Linear (→ 3)    │
  └──────────────────┘
          │
          ▼
  Glioma / Meningioma / Pituitary
```

### Why dual-branch?

| Branch | Role |
|---|---|
| **ResNet50 + CBAM** | Explicit spatial + channel attention on CNN features — highlights the tumour region in MRI slices |
| **ConvNeXt-Small** | Modern depthwise-separable design with stronger ImageNet priors — captures texture and global context |
| **Feature fusion** | Concatenating before the classifier lets the head learn cross-branch interactions, outperforming simple prediction averaging |

### Training Strategy

| Stage | Epochs | What trains | LR |
|---|---|---|---|
| **Stage 1 — Warm-up** | 10 | FusionHead only (backbones frozen) | 3e-4 |
| **Stage 2 — Fine-tuning** | 30 | All layers (differential LR) | backbone 5e-5 / head 3e-4 |

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing per stage |
| Loss | CrossEntropyLoss (label smoothing 0.1) |
| Mixed Precision | torch.amp (AMP) |
| Grad Clipping | max_norm = 1.0 |
| Batch Size | 16 |
| Train / Val Split | 80% / 20% (stratified) |

---

## 📈 Model Performance

### Dual-Branch Model (ResNet50+CBAM ⊕ ConvNeXt-Small)

| Stage | Val Accuracy | Val F1 |
|---|---|---|
| After Stage 1 (head warm-up) | 89.4% | 0.8856 |
| After Stage 2 (full fine-tune) | **98.86%** | **0.9867** |

### Per-class metrics (best checkpoint)

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Glioma | ~0.987 | ~0.987 | ~0.987 |
| Meningioma | ~0.981 | ~0.982 | ~0.981 |
| Pituitary | ~0.991 | ~0.990 | ~0.990 |

### Comparison with baseline

| Model | Val F1 |
|---|---|
| ResNet50 (single branch, original) | 0.973 |
| **ResNet50+CBAM ⊕ ConvNeXt-Small (this project)** | **0.9867** |

> Meningioma scores slightly lower due to class imbalance (708 vs 1,426 glioma images) and its visual similarity to surrounding tissue.

---

## 📁 Project Structure

```
BRAIN TUMOR DETECTION [END 2 END]/
│
├── 📂 Brain-Tumor-Test-Images/     # Sample MRI scans for testing
│
├── 📂 dataset/                     # Downloaded automatically by train.py
│   ├── bt_set1/                    # .mat files 1–766
│   ├── bt_set2/                    # .mat files 767–1532
│   ├── bt_set3/                    # .mat files 1533–2298
│   └── bt_set4/                    # .mat files 2299–3064
│
├── 📂 models/
│   └── best_model.pt               # Trained dual-branch weights (download separately)
│
├── 📂 static/                      # CSS, images
├── 📂 templates/                   # HTML templates (Flask)
│   ├── MainPage.html
│   ├── Diseasedet.html
│   ├── pred.html
│   ├── uimg.html
│   ├── app.html
│   └── error.html
│
├── app.py                          # Flask web application
├── model_architecture.py           # Dual-branch model definition
├── train.py                        # Training script (dual-branch)
├── train_resnet50.py               # Original single-branch training (kept for reference)
├── requirements.txt                # Python dependencies
├── COLAB_TRAINING.md               # Google Colab training guide
└── README.md                       # You are here
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- `best_model.pt` placed in the `models/` folder ([download here](#))

### 1. Clone the repository

```bash
git clone https://github.com/prathambalehosurr/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Set up virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate.bat

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Place `best_model.pt` inside the `models/` folder.  
Download link: **[Google Drive](#)** ← *(update this link)*

### 5. Run the app

```bash
python app.py
```

Open your browser at → **http://127.0.0.1:5000**

---

## 🏋️ Training

Training requires the Figshare dataset (~500MB). The script downloads it automatically.

**Google Colab (recommended — free GPU):**

```python
!git clone https://github.com/prathambalehosurr/brain-tumor-detection.git
%cd brain-tumor-detection
!pip install -q -r requirements.txt

!python train.py \
    --data_dir dataset \
    --checkpoint_dir models \
    --stage1_epochs 10 \
    --stage2_epochs 30 \
    --batch_size 16 \
    --download

from google.colab import files
files.download("models/best_model.pt")
```

**Local training:**

```bash
python train.py \
    --data_dir dataset \
    --checkpoint_dir models \
    --stage1_epochs 10 \
    --stage2_epochs 30 \
    --batch_size 16 \
    --download
```

All available arguments:

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | `data` | Dataset root directory |
| `--checkpoint_dir` | `checkpoints` | Where to save model weights |
| `--download` | flag | Auto-download Figshare dataset |
| `--stage1_epochs` | `10` | Head warm-up epochs |
| `--stage2_epochs` | `30` | Full fine-tuning epochs |
| `--batch_size` | `32` | Training batch size |
| `--lr_head` | `3e-4` | Learning rate for FusionHead |
| `--lr_backbone` | `5e-5` | Learning rate for backbones (stage 2) |
| `--img_size` | `224` | Input image size |
| `--dropout` | `0.4` | FusionHead dropout rate |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Deep Learning | PyTorch 2.1+, Torchvision 0.16+ |
| Model | ResNet50 + CBAM ⊕ ConvNeXt-Small |
| Image Processing | PIL (Pillow), torchvision.transforms.v2 |
| Data Loading | h5py (reads .mat files directly) |
| Training Utilities | scikit-learn (metrics), torch.amp (AMP) |
| Web Framework | Flask 3.0+ |
| Frontend | HTML5, CSS3 |
| Model Serialization | `torch.save` / `.pt` |
| Training Platform | Google Colab (T4 GPU) |

---

## 📚 References & Citation

**Dataset — please cite if you use this work:**

```bibtex
@article{Cheng2015,
  author  = {Cheng, Jun and others},
  title   = {Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition},
  journal = {PLoS ONE},
  volume  = {10},
  number  = {10},
  year    = {2015}
}

@article{Cheng2016,
  author  = {Cheng, Jun and others},
  title   = {Retrieval of Brain Tumors by Adaptive Spatial Pooling and Fisher Vector Representation},
  journal = {PLoS ONE},
  volume  = {11},
  number  = {6},
  year    = {2016}
}
```

**Further reading:**

- [Jun Cheng Brain Tumor Dataset — Figshare](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)
- [Deep Residual Learning for Image Recognition — He et al. (2015)](https://arxiv.org/abs/1512.03385)
- [A ConvNet for the 2020s — Liu et al. (2022)](https://arxiv.org/abs/2201.03545)
- [CBAM: Convolutional Block Attention Module — Woo et al. (2018)](https://arxiv.org/abs/1807.06521)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

Made with ❤️ by [Pratham Balehosur](https://github.com/prathambalehosurr)

⭐ Star this repo if it helped you!
