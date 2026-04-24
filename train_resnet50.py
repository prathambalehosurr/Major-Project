import argparse
import os
import random
import zipfile
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights
from torchvision.transforms import v2

from model_architecture import build_resnet50_model, is_classifier_or_attention_parameter


LABELS = {
    1: "Meningioma",
    2: "Glioma",
    3: "Pituitary",
}

FIGSHARE_URL = "https://ndownloader.figshare.com/articles/1512427/versions/5"
ARCHIVE_NAME = "figshare_1512427_v5.zip"
ZIP_TO_DIR = {
    "brainTumorDataPublic_1-766.zip": "bt_set1",
    "brainTumorDataPublic_767-1532.zip": "bt_set2",
    "brainTumorDataPublic_1533-2298.zip": "bt_set3",
    "brainTumorDataPublic_2299-3064.zip": "bt_set4",
}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download_dataset(data_dir):
    raw_dir = data_dir / "raw"
    contents_dir = raw_dir / "figshare_v5_contents"
    archive_path = raw_dir / ARCHIVE_NAME
    raw_dir.mkdir(parents=True, exist_ok=True)
    contents_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        import urllib.request

        print(f"Downloading Figshare dataset to {archive_path}...")
        urllib.request.urlretrieve(FIGSHARE_URL, archive_path)

    if not all((contents_dir / name).exists() for name in ZIP_TO_DIR):
        print("Extracting Figshare archive...")
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(contents_dir)

    for zip_name, target_dir in ZIP_TO_DIR.items():
        destination = data_dir / target_dir
        destination.mkdir(parents=True, exist_ok=True)
        if len(list(destination.glob("*.mat"))) == 766:
            continue
        print(f"Extracting {zip_name}...")
        with zipfile.ZipFile(contents_dir / zip_name, "r") as archive:
            archive.extractall(destination)


def collect_examples(data_dir):
    paths = sorted(data_dir.glob("bt_set*/*.mat"), key=lambda path: int(path.stem))
    examples = []

    for path in paths:
        with h5py.File(path, "r") as file:
            label = int(file["cjdata"]["label"][()][0][0])
        examples.append((path, label))

    if len(examples) != 3064:
        raise RuntimeError(f"Expected 3064 .mat files, found {len(examples)} in {data_dir}")

    return examples


class BrainTumorDataset(Dataset):
    def __init__(self, examples, transform):
        self.examples = examples
        self.transform = transform

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        path, label = self.examples[index]
        with h5py.File(path, "r") as file:
            image = np.array(file["cjdata"]["image"], dtype=np.float32)

        image = image - image.min()
        max_value = image.max()
        if max_value > 0:
            image = image / max_value
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image).convert("RGB")

        return self.transform(pil_image), label


def make_loaders(examples, image_size, batch_size, num_workers, seed):
    labels = [label for _, label in examples]
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=0.3,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [label for _, label in temp_examples]
    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    train_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip(),
        v2.RandomRotation(15),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_transform = v2.Compose([
        v2.Resize((image_size, image_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_loader = DataLoader(
        BrainTumorDataset(train_examples, train_transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        BrainTumorDataset(val_examples, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        BrainTumorDataset(test_examples, eval_transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                if scaler is None:
                    loss.backward()
                    optimizer.step()
                else:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

            running_loss += loss.item() * labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train the ResNet50 brain tumor classifier.")
    parser.add_argument("--data-dir", default="dataset", type=Path)
    parser.add_argument("--output", default="models/bt_resnet50_model.pt", type=Path)
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--epochs", default=8, type=int)
    parser.add_argument("--batch-size", default=24, type=int)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--backbone-lr", default=None, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--label-smoothing", default=0.05, type=float)
    parser.add_argument("--num-workers", default=2, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--attention", default="cbam", choices=["cbam", "none"])
    parser.add_argument("--fine-tune", action="store_true", help="Train the whole ResNet50 instead of only the head.")
    args = parser.parse_args()

    seed_everything(args.seed)

    if args.download_data:
        download_dataset(args.data_dir)

    examples = collect_examples(args.data_dir)
    counts = {name: sum(label == key for _, label in examples) for key, name in LABELS.items()}
    print(f"Found {len(examples)} examples: {counts}")

    train_loader, val_loader, test_loader = make_loaders(
        examples,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    weights = ResNet50_Weights.IMAGENET1K_V2
    model = build_resnet50_model(weights=weights, attention=args.attention)

    if not args.fine_tune:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = is_classifier_or_attention_parameter(name)

    model.to(device)
    if args.fine_tune:
        backbone_lr = args.backbone_lr if args.backbone_lr is not None else args.lr
        backbone_parameters = []
        head_and_attention_parameters = []
        for name, parameter in model.named_parameters():
            if is_classifier_or_attention_parameter(name):
                head_and_attention_parameters.append(parameter)
            else:
                backbone_parameters.append(parameter)

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_parameters, "lr": backbone_lr},
                {"params": head_and_attention_parameters, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if args.amp and device.type == "cuda" else None

    best_val_accuracy = 0.0
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        scheduler.step()
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)

        print(
            f"epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output)
            print(f"Saved new best model to {args.output}")

    model.load_state_dict(torch.load(args.output, map_location=device))
    test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f} test_acc={test_accuracy:.4f}")
    print(f"Done. Copy {args.output} into the Flask app's models folder if needed.")


if __name__ == "__main__":
    main()
