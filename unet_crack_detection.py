import argparse
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

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


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_sort_key(path: Path):
    return int(path.stem) if path.stem.isdigit() else path.stem.lower()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python environment cannot see a CUDA-enabled PyTorch build. "
            "Install a CUDA build of PyTorch, then rerun the command."
        )

    return torch.device(device_arg)


def load_rgb_image(path: Path, size: Tuple[int, int]) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB").resize(size, Image.BILINEAR)
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
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB").resize(size, Image.BILINEAR)
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
    Supports either a pre-split folder layout:

    dataset/
      train/
        images/
        masks/
      val/
        images/
        masks/

    or paired SUT segmentation files where images and masks share a stem:

    SUT Dataset/
      1-Segmentation/
        Original Image/
        Ground Truth/
    """
    def __init__(
        self,
        samples: Sequence[Tuple[Path, Path]],
        image_size=(256, 256),
        augment=False,
    ):
        self.samples = list(samples)
        self.image_size = image_size
        self.augment = augment

        if not self.samples:
            raise ValueError("No paired image/mask samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, mask_path = self.samples[idx]

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


def collect_paired_samples(images_dir: Path, masks_dir: Path) -> List[Tuple[Path, Path]]:
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    images = sorted(
        [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS],
        key=natural_sort_key,
    )
    masks_by_stem = {
        p.stem: p
        for p in masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS
    }

    samples = []
    missing = []
    for image_path in images:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            missing.append(image_path.name)
        else:
            samples.append((image_path, mask_path))

    if missing:
        preview = ", ".join(missing[:5])
        raise FileNotFoundError(f"Missing masks for {len(missing)} image(s): {preview}")

    return samples


def resolve_sut_segmentation_dirs(root: Path) -> Optional[Tuple[Path, Path]]:
    segmentation_dir = root / "1-Segmentation"
    images_dir = segmentation_dir / "Original Image"
    masks_dir = segmentation_dir / "Ground Truth"
    if images_dir.exists() and masks_dir.exists():
        return images_dir, masks_dir
    return None


def split_samples(samples: Sequence[Tuple[Path, Path]], val_split: float, seed: int):
    samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(samples)

    val_count = max(1, int(round(len(samples) * val_split)))
    val_count = min(val_count, len(samples) - 1)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]
    return train_samples, val_samples


def build_datasets(args, image_size):
    data_root = Path(args.data_dir)
    sut_dirs = resolve_sut_segmentation_dirs(data_root)

    if sut_dirs is not None:
        images_dir, masks_dir = sut_dirs
        samples = collect_paired_samples(images_dir, masks_dir)
        train_samples, val_samples = split_samples(samples, args.val_split, args.seed)
        print(f"Using SUT segmentation dataset: {len(train_samples)} train / {len(val_samples)} val pairs")
    else:
        train_samples = collect_paired_samples(data_root / "train" / "images", data_root / "train" / "masks")
        val_samples = collect_paired_samples(data_root / "val" / "images", data_root / "val" / "masks")
        print(f"Using pre-split dataset: {len(train_samples)} train / {len(val_samples)} val pairs")

    train_dataset = CrackDataset(train_samples, image_size=image_size, augment=True)
    val_dataset = CrackDataset(val_samples, image_size=image_size, augment=False)
    return train_dataset, val_dataset


class PredictDataset(Dataset):
    def __init__(self, images_dir: Path, image_size=(256, 256)):
        self.images_dir = images_dir
        self.image_size = image_size
        self.image_files = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTS],
            key=natural_sort_key,
        )

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
    def __init__(self, n_channels=3, n_classes=1, base_channels=32):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.up1 = Up(base_channels * 16, base_channels * 8)
        self.up2 = Up(base_channels * 8, base_channels * 4)
        self.up3 = Up(base_channels * 4, base_channels * 2)
        self.up4 = Up(base_channels * 2, base_channels)

        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

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


def bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
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

def estimate_pos_weight(dataset: CrackDataset, image_size: Tuple[int, int], max_weight: float = 50.0) -> float:
    positive = 0.0
    total = 0.0
    for _, mask_path in dataset.samples:
        mask = load_mask_image(mask_path, image_size)
        positive += mask.sum().item()
        total += mask.numel()

    negative = total - positive
    if positive <= 0:
        return 1.0

    return min(max_weight, negative / positive)


def train_one_epoch(model, loader, optimizer, device, pos_weight, scaler, use_amp: bool):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks, pos_weight=pos_weight)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    return running_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, device, pos_weight, use_amp: bool):
    model.eval()
    val_loss = 0.0

    all_precision = 0.0
    all_recall = 0.0
    all_f1 = 0.0
    all_iou = 0.0

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = bce_dice_loss(logits, masks, pos_weight=pos_weight)
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


@torch.no_grad()
def export_overlay_predictions(
    model,
    samples: Sequence[Tuple[Path, Path]],
    output_dir: Path,
    image_size: Tuple[int, int],
    device,
    use_amp: bool,
    threshold: float = 0.5,
    limit: int = 0,
) -> None:
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    model.eval()
    selected_samples = list(samples[:limit]) if limit and limit > 0 else list(samples)

    for image_path, _ in selected_samples:
        image = load_rgb_image(image_path, image_size).unsqueeze(0).to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(image)

        pred = (torch.sigmoid(logits)[0] > threshold).float()
        save_mask(pred, mask_dir / f"{image_path.stem}_mask.png")
        save_overlay(image_path, pred, overlay_dir / f"{image_path.stem}_overlay.png", image_size)

    print(f"Saved {len(selected_samples)} red overlay prediction(s) to: {overlay_dir.resolve()}")


def run_training(args):
    set_seed(args.seed)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    image_size = (args.image_size, args.image_size)

    train_dataset, val_dataset = build_datasets(args, image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = UNet(base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    use_amp = args.amp and device.type == "cuda"
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )

    save_dir = Path(args.save_dir)
    ensure_dir(save_dir)
    best_model_path = save_dir / "best_unet_crack_model.pt"

    pos_weight_value = estimate_pos_weight(train_dataset, image_size, max_weight=args.max_pos_weight)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(
        f"Device: {device} ({device_name}) | "
        f"AMP: {'on' if use_amp else 'off'} | "
        f"Image size: {args.image_size} | "
        f"BCE pos_weight: {pos_weight_value:.2f}"
    )

    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, pos_weight, scaler, use_amp)
        metrics = evaluate(model, val_loader, device, pos_weight, use_amp)
        scheduler.step(metrics["loss"])

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
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "base_channels": args.base_channels,
                    "image_size": args.image_size,
                    "best_f1": best_f1,
                },
                best_model_path,
            )
            print(f"Saved best model to: {best_model_path}")

    print(f"\nTraining complete. Best validation F1: {best_f1:.4f}")

    if args.export_overlays:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        export_overlay_predictions(
            model=model,
            samples=val_dataset.samples,
            output_dir=Path(args.output_dir),
            image_size=image_size,
            device=device,
            use_amp=use_amp,
            threshold=args.threshold,
            limit=args.max_overlay_images,
        )


@torch.no_grad()
def run_prediction(args):
    device = resolve_device(args.device)
    image_size = (args.image_size, args.image_size)

    checkpoint = torch.load(args.model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        base_channels = checkpoint.get("base_channels", args.base_channels)
        model_state = checkpoint["model_state_dict"]
    else:
        base_channels = args.base_channels
        model_state = checkpoint

    model = UNet(base_channels=base_channels).to(device)
    model.load_state_dict(model_state)
    model.eval()

    predict_dataset = PredictDataset(Path(args.predict_dir), image_size=image_size)
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=0)

    output_dir = Path(args.output_dir)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    ensure_dir(mask_dir)
    ensure_dir(overlay_dir)

    for images, image_paths in predict_loader:
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=args.amp and device.type == "cuda"):
            logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > args.threshold).float()

        image_path = Path(image_paths[0])
        save_mask(preds[0], mask_dir / f"{image_path.stem}_mask.png")
        save_overlay(image_path, preds[0], overlay_dir / f"{image_path.stem}_overlay.png", image_size)

        print(f"Predicted: {image_path.name}")

    print(f"\nPrediction complete. Results saved to: {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="U-Net pavement crack segmentation")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="Train the U-Net")
    train_parser.add_argument("--data_dir", type=str, default="SUT Dataset", help="Dataset root folder")
    train_parser.add_argument("--save_dir", type=str, default="unet_checkpoints", help="Folder to save model")
    train_parser.add_argument("--epochs", type=int, default=25)
    train_parser.add_argument("--batch_size", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--image_size", type=int, default=256)
    train_parser.add_argument("--base_channels", type=int, default=32)
    train_parser.add_argument("--val_split", type=float, default=0.2)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--max_pos_weight", type=float, default=50.0)
    train_parser.add_argument("--num_workers", type=int, default=4)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--output_dir", type=str, default="unet_outputs", help="Folder for red overlay outputs")
    train_parser.add_argument("--export_overlays", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--max_overlay_images", type=int, default=0, help="0 exports all validation overlays")
    train_parser.add_argument("--threshold", type=float, default=0.5)

    predict_parser = subparsers.add_parser("predict", help="Predict masks on new images")
    predict_parser.add_argument("--model_path", type=str, required=True, help="Path to saved .pt model")
    predict_parser.add_argument("--predict_dir", type=str, required=True, help="Folder of images to predict")
    predict_parser.add_argument("--output_dir", type=str, default="unet_predictions", help="Output folder")
    predict_parser.add_argument("--image_size", type=int, default=256)
    predict_parser.add_argument("--base_channels", type=int, default=32)
    predict_parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    predict_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    predict_parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "predict":
        run_prediction(args)
    else:
        raise ValueError("Invalid mode.")


if __name__ == "__main__":
    main()
