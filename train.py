"""
train.py
────────
Refactored training script for the Dual-Branch brain-tumour classifier.

Key upgrades over original
──────────────────────────
* DualBranchTumorClassifier  (ResNet50+CBAM  ⊕  ConvNeXt-Small)
* Mixed-precision training   (torch.amp)
* Two-stage training:
    Stage 1 – frozen backbones, train FusionHead only  (warm-up)
    Stage 2 – unfreeze all, full fine-tuning with lower LR
* Cosine-annealing LR scheduler per stage
* Per-epoch validation metrics: accuracy, precision, recall, F1 (macro)
* Best-model checkpoint (by val F1)
* Configurable via argparse
"""

from __future__ import annotations

import argparse
import os
import random
import zipfile
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from model_architecture import (
    DualBranchTumorClassifier,
    build_dual_branch_model,
    is_classifier_or_attention_parameter,
)

# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────

LABELS = {1: "Meningioma", 2: "Glioma", 3: "Pituitary"}
NUM_CLASSES = len(LABELS)

FIGSHARE_URL = "https://ndownloader.figshare.com/articles/1512427/versions/5"
ARCHIVE_NAME = "figshare_1512427_v5.zip"
ZIP_TO_DIR = {
    "brainTumorDataPublic_1-766.zip":    "bt_set1",
    "brainTumorDataPublic_767-1532.zip": "bt_set2",
    "brainTumorDataPublic_1533-2298.zip":"bt_set3",
    "brainTumorDataPublic_2299-3064.zip":"bt_set4",
}

# ──────────────────────────────────────────────
#  Reproducibility
# ──────────────────────────────────────────────

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────
#  Data download / extraction  (unchanged logic)
# ──────────────────────────────────────────────

def download_dataset(data_dir: Path):
    raw_dir      = data_dir / "raw"
    contents_dir = raw_dir / "figshare_v5_contents"
    archive_path = raw_dir / ARCHIVE_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    contents_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        import urllib.request
        print(f"Downloading Figshare dataset → {archive_path} …")
        urllib.request.urlretrieve(FIGSHARE_URL, archive_path)

    if not all((contents_dir / name).exists() for name in ZIP_TO_DIR):
        print("Extracting Figshare archive …")
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(contents_dir)

    for zip_name, target_dir in ZIP_TO_DIR.items():
        destination = data_dir / target_dir
        destination.mkdir(parents=True, exist_ok=True)
        if len(list(destination.glob("*.mat"))) == 766:
            continue
        print(f"Extracting {zip_name} …")
        with zipfile.ZipFile(contents_dir / zip_name, "r") as zf:
            zf.extractall(destination)


def collect_examples(data_dir: Path) -> List[Tuple[Path, int]]:
    paths = sorted(data_dir.glob("bt_set*/*.mat"), key=lambda p: int(p.stem))
    examples = []
    for path in paths:
        with h5py.File(path, "r") as f:
            label = int(f["cjdata"]["label"][()][0][0])
        examples.append((path, label))
    if len(examples) != 3064:
        raise RuntimeError(f"Expected 3064 .mat files, found {len(examples)}.")
    return examples


# ──────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    def __init__(self, examples: List[Tuple[Path, int]], transform):
        self.examples  = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        path, label = self.examples[index]
        with h5py.File(path, "r") as f:
            image = np.array(f["cjdata"]["image"], dtype=np.float32)

        # normalise to [0, 255] then convert to uint8 PIL image
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val
        image = (image * 255).astype(np.uint8)
        pil   = Image.fromarray(image).convert("RGB")

        tensor = self.transform(pil)
        # labels are 1-indexed → shift to 0-indexed for CrossEntropy
        return tensor, label - 1


# ──────────────────────────────────────────────
#  Transforms
# ──────────────────────────────────────────────

def get_transforms(img_size: int = 224):
    """
    ConvNeXt's official weights expect 224×224 with ImageNet stats.
    ResNet50 is the same. Use one consistent pipeline for both branches.
    """
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)

    train_tf = v2.Compose([
        v2.Resize((img_size + 32, img_size + 32)),
        v2.RandomCrop(img_size),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.3, contrast=0.3),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    val_tf = v2.Compose([
        v2.Resize((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])
    return train_tf, val_tf


# ──────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────

def compute_metrics(all_labels: List[int], all_preds: List[int]) -> dict:
    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def fmt_metrics(m: dict) -> str:
    return (f"Acc={m['accuracy']:.4f}  "
            f"P={m['precision']:.4f}  "
            f"R={m['recall']:.4f}  "
            f"F1={m['f1']:.4f}")


# ──────────────────────────────────────────────
#  One epoch: train
# ──────────────────────────────────────────────

def train_one_epoch(
    model:     DualBranchTumorClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
) -> Tuple[float, dict]:
    model.train()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()

        with autocast():                     # mixed precision
            logits = model(images)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ──────────────────────────────────────────────
#  One epoch: validate
# ──────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:     DualBranchTumorClassifier,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> Tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics


# ──────────────────────────────────────────────
#  Full training pipeline
# ──────────────────────────────────────────────

def run_training(args: argparse.Namespace):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── data ───────────────────────────────────
    data_dir = Path(args.data_dir)
    if args.download:
        download_dataset(data_dir)

    examples = collect_examples(data_dir)
    labels   = [lab for _, lab in examples]

    train_ex, val_ex = train_test_split(
        examples, test_size=args.val_split, stratify=labels,
        random_state=args.seed,
    )
    print(f"Train: {len(train_ex)}  |  Val: {len(val_ex)}")

    train_tf, val_tf = get_transforms(args.img_size)

    train_ds = BrainTumorDataset(train_ex, train_tf)
    val_ds   = BrainTumorDataset(val_ex,   val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── model ──────────────────────────────────
    model = build_dual_branch_model(
        num_classes    = NUM_CLASSES,
        pretrained     = True,
        hidden_dim     = args.hidden_dim,
        dropout        = args.dropout,
        cbam_reduction = args.cbam_reduction,
    ).to(device)

    model.parameter_summary()

    # ── loss ───────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── AMP scaler ─────────────────────────────
    scaler = GradScaler()

    # ── checkpoint dir ─────────────────────────
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "best_model.pt"
    best_f1   = 0.0

    # ══════════════════════════════════════════
    #  STAGE 1 — Frozen backbones, train head
    # ══════════════════════════════════════════
    print("\n" + "═"*55)
    print(f"  STAGE 1 — Head warm-up  ({args.stage1_epochs} epochs)")
    print("═"*55)
    model.freeze_backbones()
    model.parameter_summary()

    opt1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_head, weight_decay=args.weight_decay,
    )
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt1, T_max=args.stage1_epochs, eta_min=args.lr_head * 0.1,
    )

    for epoch in range(1, args.stage1_epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, criterion, opt1, scaler, device)
        vl_loss, vl_m = validate(model, val_loader, criterion, device)
        sched1.step()

        print(f"[S1 E{epoch:02d}] "
              f"train_loss={tr_loss:.4f}  {fmt_metrics(tr_m)}  |  "
              f"val_loss={vl_loss:.4f}  {fmt_metrics(vl_m)}")

        if vl_m["f1"] > best_f1:
            best_f1 = vl_m["f1"]
            model.save(str(best_ckpt))
            print(f"  ✔ New best val F1={best_f1:.4f} — checkpoint saved.")

    # ══════════════════════════════════════════
    #  STAGE 2 — Full fine-tuning
    # ══════════════════════════════════════════
    print("\n" + "═"*55)
    print(f"  STAGE 2 — Full fine-tuning  ({args.stage2_epochs} epochs)")
    print("═"*55)
    model.unfreeze_backbones()
    model.parameter_summary()

    # Differential learning rates: backbones get lower LR than head
    param_groups = [
        {"params": model.branch_resnet.parameters(),  "lr": args.lr_backbone},
        {"params": model.branch_convnxt.parameters(), "lr": args.lr_backbone},
        {"params": model.fusion_head.parameters(),    "lr": args.lr_head},
    ]
    opt2 = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt2, T_max=args.stage2_epochs, eta_min=args.lr_backbone * 0.01,
    )

    for epoch in range(1, args.stage2_epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, criterion, opt2, scaler, device)
        vl_loss, vl_m = validate(model, val_loader, criterion, device)
        sched2.step()

        print(f"[S2 E{epoch:02d}] "
              f"train_loss={tr_loss:.4f}  {fmt_metrics(tr_m)}  |  "
              f"val_loss={vl_loss:.4f}  {fmt_metrics(vl_m)}")

        if vl_m["f1"] > best_f1:
            best_f1 = vl_m["f1"]
            model.save(str(best_ckpt))
            print(f"  ✔ New best val F1={best_f1:.4f} — checkpoint saved.")

    print(f"\nTraining complete.  Best val F1 = {best_f1:.4f}")
    print(f"Best checkpoint    : {best_ckpt}")


# ──────────────────────────────────────────────
#  Inference helper
# ──────────────────────────────────────────────

def load_model_for_inference(checkpoint_path: str, device: str = "cpu") -> DualBranchTumorClassifier:
    """Load saved model weights for inference."""
    model = DualBranchTumorClassifier.load(
        checkpoint_path,
        num_classes    = NUM_CLASSES,
        pretrained     = False,    # weights come from checkpoint
    )
    model.to(device).eval()
    return model


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dual-Branch Brain Tumour Classifier")

    # paths
    p.add_argument("--data_dir",       default="data",         help="Root data directory")
    p.add_argument("--checkpoint_dir", default="checkpoints",  help="Where to save checkpoints")
    p.add_argument("--download",       action="store_true",    help="Download dataset if not present")

    # data
    p.add_argument("--val_split",    type=float, default=0.2,  help="Validation fraction")
    p.add_argument("--img_size",     type=int,   default=224,  help="Input image size")
    p.add_argument("--batch_size",   type=int,   default=32,   help="Batch size")
    p.add_argument("--num_workers",  type=int,   default=4,    help="DataLoader workers")
    p.add_argument("--seed",         type=int,   default=42)

    # model
    p.add_argument("--hidden_dim",     type=int,   default=512,  help="FusionHead hidden dim")
    p.add_argument("--dropout",        type=float, default=0.4,  help="FusionHead dropout")
    p.add_argument("--cbam_reduction", type=int,   default=16,   help="CBAM reduction ratio")

    # training
    p.add_argument("--stage1_epochs", type=int,   default=10,    help="Head warm-up epochs")
    p.add_argument("--stage2_epochs", type=int,   default=30,    help="Fine-tuning epochs")
    p.add_argument("--lr_head",       type=float, default=3e-4,  help="LR for FusionHead")
    p.add_argument("--lr_backbone",   type=float, default=5e-5,  help="LR for backbones in stage 2")
    p.add_argument("--weight_decay",  type=float, default=1e-4)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training(args)