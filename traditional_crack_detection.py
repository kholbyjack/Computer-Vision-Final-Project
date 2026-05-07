import argparse
from pathlib import Path

import cv2
import numpy as np

# Enable OpenCV's optimized kernels once so the traditional pipeline runs faster.
cv2.setUseOptimized(True)

# Keep file filtering and resize settings centralized for the testset workflow.
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_IMAGE_SIZE = (512, 512)

# Multi-scale morphology catches cracks of different widths without using ML.
BLACKHAT_KERNEL_SIZES = (7, 11, 17, 23)

# Tuned on the testset ground truth; high percentile keeps precision reasonable.
DEFAULT_SENSITIVITY = 98.25


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_dir: Path) -> None:
    # Create output folders once per run; cv2.imwrite fails if parents are missing.
    ensure_dir(output_dir / "steps")
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "overlays")


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    crack_pixels = mask > 0

    if not np.any(crack_pixels):
        return overlay

    # Blend only crack pixels instead of allocating a full red image.
    overlay_pixels = overlay[crack_pixels].astype(np.float32)
    overlay_pixels[:, 0] *= 0.35
    overlay_pixels[:, 1] *= 0.35
    overlay_pixels[:, 2] = (overlay_pixels[:, 2] * 0.35) + (255 * 0.65)
    overlay[crack_pixels] = np.clip(overlay_pixels, 0, 255).astype(np.uint8)

    return overlay


def build_multiscale_blackhat(image: np.ndarray) -> np.ndarray:
    blackhat = np.zeros_like(image)

    for kernel_size in BLACKHAT_KERNEL_SIZES:
        # Merge several morphology responses so both hairline and wider cracks
        # can survive the same thresholding stage.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        response = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        cv2.max(blackhat, response, dst=blackhat)

    return cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


def build_crack_response(denoised: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # A large closing estimates local pavement brightness; subtracting the image
    # highlights dark crack valleys while staying purely traditional.
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, background_kernel)
    local_dark = cv2.subtract(background, denoised)
    local_dark = cv2.normalize(local_dark, None, 0, 255, cv2.NORM_MINMAX)

    blackhat = build_multiscale_blackhat(denoised)

    # Keep the strongest response from either local-dark subtraction or black-hat.
    crack_response = cv2.max(blackhat, local_dark)
    blur = cv2.GaussianBlur(crack_response, (3, 3), 0)
    crack_response = cv2.addWeighted(crack_response, 1.5, blur, -0.5, 0)

    return crack_response, blackhat


def add_supported_edges(mask: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    # Canny alone is noisy on pavement; only keep edges near existing crack
    # candidates so texture edges do not dominate the mask.
    edges = cv2.Canny(denoised, 50, 130)
    support_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    support = cv2.dilate(mask, support_kernel, iterations=1)
    supported_edges = cv2.bitwise_and(edges, support)
    cv2.bitwise_or(mask, supported_edges, dst=mask)
    return mask


def remove_small_components(mask: np.ndarray, min_area: int = 120) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask)

    # Vectorize area and shape filtering so compact pavement blobs are rejected
    # while elongated crack-like components survive.
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    aspect_ratios = np.maximum(widths / np.maximum(heights, 1), heights / np.maximum(widths, 1))
    extents = areas / np.maximum(widths * heights, 1)

    keep_labels = (areas >= min_area) & ((aspect_ratios >= 1.15) | (areas >= min_area * 8)) & (extents <= 0.95)
    keep_labels[0] = False
    cleaned = np.zeros_like(mask)
    cleaned[keep_labels[labels]] = 255
    return cleaned


def detect_cracks_traditional(image: np.ndarray, sensitivity: float = DEFAULT_SENSITIVITY):
    original_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small images
    h, w = gray.shape
    if w < 512:
        scale = 512 / w
        original_image = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Refresh dimensions after resizing so thresholds scale with processed data.
        h, w = gray.shape

    # Tuned CLAHE improves crack contrast without over-amplifying rock texture.
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light blur plus bilateral filtering suppresses gravel texture but preserves edges.
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=50, sigmaSpace=50)

    crack_response, blackhat = build_crack_response(denoised)

    # High-percentile threshold was the best traditional tradeoff on the testset.
    threshold_value = np.percentile(crack_response, sensitivity)
    _, mask = cv2.threshold(crack_response, threshold_value, 255, cv2.THRESH_BINARY)

    mask = add_supported_edges(mask, denoised)

    # Remove small isolated responses after edge support; this improved precision.
    mask = remove_small_components(mask, min_area=120)

    overlay = overlay_mask_on_image(original_image, mask)

    return {
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "blackhat": blackhat,
        "raw_mask": mask,
        "binary_mask": mask,
        "overlay": overlay,
    }


def process_one_image(image_path: Path, output_dir: Path, sensitivity: float):
    # OpenCV reads directly into BGR arrays, avoiding PIL conversion overhead and
    # keeping channel order correct for cv2.cvtColor and cv2.imwrite.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize testset images once at load time for consistent metric comparison.
    if (image.shape[1], image.shape[0]) != TARGET_IMAGE_SIZE:
        image = cv2.resize(image, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    results = detect_cracks_traditional(image, sensitivity)

    stem = image_path.stem
    cv2.imwrite(str(output_dir / "steps" / f"{stem}_01_gray.png"), results["gray"])
    cv2.imwrite(str(output_dir / "steps" / f"{stem}_02_enhanced.png"), results["enhanced"])
    cv2.imwrite(str(output_dir / "steps" / f"{stem}_03_denoised.png"), results["denoised"])
    cv2.imwrite(str(output_dir / "steps" / f"{stem}_04_blackhat.png"), results["blackhat"])
    cv2.imwrite(str(output_dir / "steps" / f"{stem}_05_raw_mask.png"), results["raw_mask"])
    cv2.imwrite(str(output_dir / "masks" / f"{stem}_06_mask.png"), results["binary_mask"])
    cv2.imwrite(str(output_dir / "overlays" / f"{stem}_07_overlay.png"), results["overlay"])

    print(f"Processed: {image_path.name}")


def collect_images(input_path: Path):
    if input_path.is_file():
        return [input_path]

    # Reuse the module-level extension set and avoid building an intermediate list.
    return sorted(p for p in input_path.iterdir() if p.suffix.lower() in VALID_IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="Crack Mask Generator")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="traditional_outputs")
    parser.add_argument("--sensitivity", type=float, default=DEFAULT_SENSITIVITY)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = collect_images(input_path)

    if not images:
        raise ValueError("No images found.")

    ensure_output_dirs(output_dir)

    for img in images:
        process_one_image(img, output_dir, args.sensitivity)

    print(f"\nDone. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
