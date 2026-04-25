import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utility helpers
# -----------------------------

def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_rgb_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.tensor(arr, dtype=torch.float32)


def load_mask_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    mask = Image.open(path).convert("L").resize(size, Image.NEAREST)
    arr = np.asarray(mask, dtype=np.float32)
    arr = (arr > 127).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)
    return torch.tensor(arr, dtype=torch.float32)


def save_mask(mask_tensor: torch.Tensor, save_path: Path) -> None:
    mask = mask_tensor.squeeze().detach().cpu().numpy()
    mask = (mask * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(mask).save(save_path)


def save_overlay(image_path: Path, pred_mask: torch.Tensor, save_path: Path, size: Tuple[int, int]) -> None:
    image = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    image_np = np.asarray(image).copy()

    mask = pred_mask.squeeze().detach().cpu().numpy() > 0.5
    overlay = image_np.copy()
    overlay[mask] = (0.4 * overlay[mask] + 0.6 * np.array([255, 0, 0])).astype(np.uint8)

    Image.fromarray(overlay).save(save_path)


# -----------------------------
# Dataset
# -----------------------------

class CrackDataset(Dataset):
    """
    Directory layout expected:

    dataset/
      train/
        images/
        masks/
      val/
        images/
        masks/

    Masks must have matching filenames with the images.
    """
    def __init__(self, images_dir: Path, masks_dir: Path, image_size=(256, 256), augment=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = image_size
        self.augment = augment

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])

        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        mask_path = self.masks_dir / image_path.name

        if not mask_path.exists():
            raise FileNotFoundError(f"Missing mask for {image_path.name}")

        image = load_rgb_image(image_path, self.image_size)
        mask = load_mask_image(mask_path, self.image_size)

        if self.augment:
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if np.random.rand() > 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])

        return image, mask


class PredictDataset(Dataset):
    def __init__(self, images_dir: Path, image_size=(256, 256)):
        self.images_dir = images_dir
        self.image_size = image_size
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        self.image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in valid_exts])

        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        image_path = self.image_files[idx]
        image = load_rgb_image(image_path, self.image_size)
        return image, str(image_path)


# -----------------------------
# U-Net model
# -----------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


# -----------------------------
# Loss and metrics
# -----------------------------

def dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    intersection = (probs * targets).sum(dim=1)
    dice = (2.0 * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss(logits, targets)
    return bce + dloss


def segmentation_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    tp = (preds * targets).sum(dim=1)
    fp = (preds * (1 - targets)).sum(dim=1)
    fn = ((1 - preds) * targets).sum(dim=1)

    precision = (tp / (tp + fp + 1e-8)).mean().item()
    recall = (tp / (tp + fn + 1e-8)).mean().item()
    f1 = (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()
    iou = (tp / (tp + fp + fn + 1e-8)).mean().item()

    return precision, recall, f1, iou


# -----------------------------
# Training / evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    val_loss = 0.0

    all_precision = 0.0
    all_recall = 0.0
    all_f1 = 0.0
    all_iou = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = bce_dice_loss(logits, masks)
        val_loss += loss.item()

        precision, recall, f1, iou = segmentation_metrics(logits, masks)
        all_precision += precision
        all_recall += recall
        all_f1 += f1
        all_iou += iou

    n = max(1, len(loader))
    return {
        "loss": val_loss / n,
        "precision": all_precision / n,
        "recall": all_recall / n,
        "f1": all_f1 / n,
        "iou": all_iou / n,
    }


def run_training(args):
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_size, args.image_size)

    train_dataset = CrackDataset(
        Path(args.data_dir) / "train" / "images",
        Path(args.data_dir) / "train" / "masks",
        image_size=image_size,
        augment=True,
    )
    val_dataset = CrackDataset(
        Path(args.data_dir) / "val" / "images",
        Path(args.data_dir) / "val" / "masks",
        image_size=image_size,
        augment=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    best_model_path = save_dir / "best_unet_crack_model.pt"

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {metrics['loss']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"IoU: {metrics['iou']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model to: {best_model_path}")

    print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")


@torch.no_grad()
def run_prediction(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (args.image_size, args.image_size)

    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    predict_dataset = PredictDataset(Path(args.predict_dir), image_size=image_size)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=0)

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    for images, image_paths in predict_loader:
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        image_path = Path(image_paths[0])
        save_mask(preds[0], mask_dir / f"{image_path.stem}_mask.png")
        save_overlay(image_path, preds[0], overlay_dir / f"{image_path.stem}_overlay.png", image_size)

        print(f"Predicted: {image_path.name}")

    print(f"\nPrediction complete. Results saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="U-Net pavement crack segmentation")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train the U-Net")
    train_parser.add_argument("--data_dir", type=str, required=True, help="Dataset root folder")
    train_parser.add_argument("--save_dir", type=str, default="unet_checkpoints", help="Folder to save model")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--batch_size", type=int, default=4)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--image_size", type=int, default=256)

    predict_parser = subparsers.add_parser("predict", help="Predict masks on new images")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to saved .pt model")
    predict_parser.add_argument("--predict_dir", type=str, required=True, help="Folder of images to predict")
    predict_parser.add_argument("--output_dir", type=str, default="unet_predictions", help="Output folder")
    predict_parser.add_argument("--image_size", type=int, default=256)

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_prediction(args)
    else:
        raise ValueError("Invalid mode.")


if __name__ == "__main__":
    main()
