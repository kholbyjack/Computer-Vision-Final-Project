import argparse
from pathlib import Path

import cv2
import numpy as np


# Image settings
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
TARGET_IMAGE_SIZE = (512, 512)
DEFAULT_SENSITIVITY = 98.25

# Enable optimized kernels
cv2.setUseOptimized(True)


# ----- Helper Functions
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_output_dirs(output_dir: Path) -> None:
    ensure_dir(output_dir / "steps")
    ensure_dir(output_dir / "masks")
    ensure_dir(output_dir / "overlays")


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    crack_pixels = mask > 0

    if not np.any(crack_pixels):
        return overlay

    # Overlay the crack's pixels onto a copy of the image
    overlay_pixels = overlay[crack_pixels].astype(np.float32)
    overlay_pixels[:, 0] *= 0.35
    overlay_pixels[:, 1] *= 0.35
    overlay_pixels[:, 2] = (overlay_pixels[:, 2] * 0.35) + (255 * 0.65)
    overlay[crack_pixels] = np.clip(overlay_pixels, 0, 255).astype(np.uint8)

    return overlay


# ----- Edge Detection Steps
def build_multiscale_blackhat(image: np.ndarray) -> np.ndarray:
    blackhat = np.zeros_like(image)
    blackhat_kernel_sizes = (7, 11, 17, 23)

    # Merges the morphological responses to four different kernel sizes
    for kernel_size in blackhat_kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        response = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        cv2.max(blackhat, response, dst=blackhat)

    return cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


def build_crack_response(denoised: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Subtraction to emphasive dark crack shadows
    background_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    background = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, background_kernel)
    local_dark = cv2.subtract(background, denoised)
    local_dark = cv2.normalize(local_dark, None, 0, 255, cv2.NORM_MINMAX)

    blackhat = build_multiscale_blackhat(denoised)

    # Use only the maximum value out of the blackhat and local dark images
    crack_response = cv2.max(blackhat, local_dark)

    # Blending a blurred version of the crack repsonse with itself
    blur = cv2.GaussianBlur(crack_response, (3, 3), 0)
    crack_response = cv2.addWeighted(crack_response, 1.5, blur, -0.5, 0)

    return crack_response, blackhat


def add_supported_edges(mask: np.ndarray, denoised: np.ndarray) -> np.ndarray:
    # Canny edge detection
    edges = cv2.Canny(denoised, 50, 130)

    # convolving the canny image with another kernel, since Canny is sensitive to noise 
    support_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    support = cv2.dilate(mask, support_kernel, iterations=1)
    supported_edges = cv2.bitwise_and(edges, support)
    cv2.bitwise_or(mask, supported_edges, dst=mask)

    return mask


def remove_small_components(mask: np.ndarray, min_area: int = 120) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(mask)

    # 
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.float32)
    widths = stats[:, cv2.CC_STAT_WIDTH].astype(np.float32)
    heights = stats[:, cv2.CC_STAT_HEIGHT].astype(np.float32)
    aspect_ratios = np.maximum(widths / np.maximum(heights, 1), heights / np.maximum(widths, 1))
    extents = areas / np.maximum(widths * heights, 1)

    # Only keep labels within a certain range
    kept_labels = (areas >= min_area) & ((aspect_ratios >= 1.15) | (areas >= min_area * 8)) & (extents <= 0.95)
    kept_labels[0] = False

    # Copy kept labels 
    cleaned = np.zeros_like(mask)
    cleaned[kept_labels[labels]] = 255

    return cleaned


def detect_cracks_traditional(image: np.ndarray, sensitivity: float = DEFAULT_SENSITIVITY):
    original_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 0.5: Upscale small images
    h, w = gray.shape
    if w < 512:
        scale = 512 / w
        original_image = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        # Refresh dimensions after resizing so thresholds scale with processed data.
        h, w = gray.shape

    # 1: Improve image contrast
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # 2: Texture suppresion
    denoised = cv2.GaussianBlur(enhanced, (3, 3), 0)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=50, sigmaSpace=50)

    # 3: Blackhat image
    crack_response, blackhat = build_crack_response(denoised)

    # 4: Image thresholding: making the image black and white, with white representing cracks in the pavement
    threshold_value = np.percentile(crack_response, sensitivity)
    _, mask = cv2.threshold(crack_response, threshold_value, 255, cv2.THRESH_BINARY)

    # 5: Canny edge detection
    mask = add_supported_edges(mask, denoised)

    # 6: Remove small isolated responses after edge support; this improved precision.
    mask = remove_small_components(mask, min_area=120)

    # 6.5: Overlay the mask on the image to emphasize cracks
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
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Resize the image to be 512 x 512
    if (image.shape[1], image.shape[0]) != TARGET_IMAGE_SIZE:
        image = cv2.resize(image, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    # Crack detection
    results = detect_cracks_traditional(image, sensitivity)

    # Saving images
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
