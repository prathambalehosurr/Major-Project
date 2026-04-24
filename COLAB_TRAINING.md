# Train the ResNet50 + CBAM Model in Google Colab

The original Google Drive link for `models/bt_resnet50_model.pt` is no longer accessible. Use this Colab flow to train a ResNet50 + CBAM attention replacement from the Figshare dataset and export a file that the Flask app can load directly.

## 1. Use a GPU Runtime

In Colab, choose:

```text
Runtime > Change runtime type > T4 GPU
```

## 2. Clone the Repository

```bash
!git clone https://github.com/shsarv/Machine-Learning-Projects.git
%cd "Machine-Learning-Projects/BRAIN TUMOR DETECTION [END 2 END]"
```

If you are using your local edited copy instead, upload this project folder to Drive and `%cd` into it.

## 3. Install Dependencies

```bash
!pip install -q torch torchvision flask pillow h5py scikit-learn
```

## 4. Upload Local Project Files

If you cloned the original GitHub repo, upload these local files from this project folder into the current Colab directory:

```text
train_resnet50.py
model_architecture.py
```

If you pushed your edited repo to GitHub first, this step is not needed.

## 5. Train and Export the Model

Fast starter run, trains the classifier head and CBAM blocks:

```bash
!python train_resnet50.py --download-data --attention cbam --epochs 10 --batch-size 16 --output models/bt_resnet50_model.pt
```

Higher-quality run, fine-tunes all ResNet50 + CBAM layers:

```bash
!python train_resnet50.py --download-data --attention cbam --fine-tune --epochs 35 --image-size 512 --batch-size 8 --lr 3e-4 --backbone-lr 3e-5 --weight-decay 1e-4 --label-smoothing 0.03 --amp --output models/bt_resnet50_model.pt
```

The script downloads the Figshare dataset from:

```text
https://ndownloader.figshare.com/articles/1512427/versions/5
```

It expects 3,064 `.mat` files and trains with the same output shape as the Flask app:

```text
0 = None
1 = Meningioma
2 = Glioma
3 = Pituitary
```

## 6. Download the Trained Weights

```python
from google.colab import files
files.download("models/bt_resnet50_model.pt")
```

Put the downloaded file here on your Windows machine:

```text
C:\Users\prath\OneDrive\Documents\New project\brain-tumor-detection\BRAIN TUMOR DETECTION [END 2 END]\models\bt_resnet50_model.pt
```

Then restart Flask and upload one of the images from `Brain-Tumor-Test-Images`.
