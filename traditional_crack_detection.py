import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    crack_pixels = mask > 0

    if not np.any(crack_pixels):
        return overlay

    red = np.zeros_like(image)
    red[:, :, 2] = 255

    overlay[crack_pixels] = cv2.addWeighted(
        image[crack_pixels], 0.35,
        red[crack_pixels], 0.65,
        0
    )

    return overlay


def remove_small_components(mask: np.ndarray, min_area: int = 300):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area >= min_area:
            cleaned[labels == label] = 255

    return cleaned


def detect_cracks_traditional(image: np.ndarray, sensitivity: float = 85.0):
    original_image = image.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Upscale small images
    h, w = gray.shape
    if w < 512:
        scale = 512 / w
        original_image = cv2.resize(original_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(5, 5))
    enhanced = clahe.apply(gray)

    # Strong texture suppression (VERY IMPORTANT)
    denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    denoised = cv2.bilateralFilter(denoised, d=5, sigmaColor=75, sigmaSpace=75)

    # correcting light non-uniformities
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    bg_noise=cv2.morphologyEx(denoised, cv2.MORPH_DILATE, kernel)
    denoised=cv2.divide(denoised, bg_noise, scale=255)

    # Black-hat to extract dark cracks
    k = max(15, int(min(h, w) * 0.03))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
    blur = cv2.GaussianBlur(blackhat, (5,5), 0)
    blackhat = cv2.addWeighted(blackhat, 1.8, blur, -0.8, 0)

    threshold_value = np.percentile(blackhat, sensitivity)
    _, mask_blackhat = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)

    # 
    blur = cv2.GaussianBlur(blackhat, (5,5), 0)
    boosted = cv2.addWeighted(blackhat, 1.8, blur, -0.8, 0)
  
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        61,
        3
    )

    otsu_val, _ = cv2.threshold(boosted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = max(5, int(0.4 * otsu_val))
    high = max(20, int(2.5 * low))

    # Edge detection
    edges = cv2.Canny(denoised, low, high)

    # Combine all signals
    mask = cv2.bitwise_or(mask_blackhat, adaptive)
    mask = cv2.bitwise_or(mask, edges)

    # Remove speckle noise
    # changing from (3, 3) to (5, 5) causes a drastic improvement
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=4)

    # Connect crack segments
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # Remove tiny components (FINAL CLEANUP)
    min_area = max(500, min(h, w)*0.05)
    mask = remove_small_components(mask, min_area=min_area)

    # add a line shaped kernel
    line_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))
    line_response = cv2.morphologyEx(mask, cv2.MORPH_BLACKHAT, line_kernel)
    mask = cv2.bitwise_or(mask, line_response)

    # Slight dilation for training usability
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    final_mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    overlay = overlay_mask_on_image(original_image, final_mask)

    return {
        "gray": gray,
        "enhanced": enhanced,
        "denoised": denoised,
        "blackhat": blackhat,
        "raw_mask": mask,
        "binary_mask": final_mask,
        "overlay": overlay,
    }

    # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    # enhanced = clahe.apply(gray)

    # # Strong texture suppression (VERY IMPORTANT)
    # # denoised = cv2.GaussianBlur(enhanced, (5, 5), 0)
    # denoised = cv2.fastNlMeansDenoising(enhanced, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # denoised = cv2.bilateralFilter(denoised, d=9, sigmaColor=75, sigmaSpace=75)

    # # light
    # kernel=cv2.getStructuringElement(cv2.MORPH_RECT , (8, 8))
    # bg_noise=cv2.morphologyEx(denoised, cv2.MORPH_DILATE, kernel)
    # denoised=cv2.divide(denoised, bg_noise, scale=255)

    # # Black-hat to extract dark cracks
    # k = max(15, int(min(h, w) * 0.03))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    # blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
    # blackhat = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)


    # threshold_value = np.percentile(blackhat, sensitivity)
    # _, mask_blackhat = cv2.threshold(blackhat, threshold_value, 255, cv2.THRESH_BINARY)

    # # Adaptive threshold
    # adaptive = cv2.adaptiveThreshold(
    #     denoised,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY_INV,
    #     91,
    #     3
    # )

    # # Edge detection
    # edges = cv2.Canny(denoised, 120, 240)

    # # Combine all signals
    # mask = cv2.bitwise_or(mask_blackhat, adaptive)
    # mask = cv2.bitwise_or(mask, edges)

    # # Remove speckle noise
    # open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=2)

    # # Connect crack segments
    # close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)

    # # Remove tiny components (FINAL CLEANUP)
    # min_area = max(500, min(h, w)*0.05)
    # mask = remove_small_components(mask, min_area=min_area)


    # # Slight dilation for training usability
    # dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # final_mask = cv2.dilate(mask, dilate_kernel, iterations=1)

    # overlay = overlay_mask_on_image(original_image, final_mask)

    # return {
    #     "gray": gray,
    #     "enhanced": enhanced,
    #     "denoised": denoised,
    #     "blackhat": blackhat,
    #     "raw_mask": mask,
    #     "binary_mask": final_mask,
    #     "overlay": overlay,
    # }
    


def process_one_image(image_path: Path, output_dir: Path, sensitivity: float):
    # image = cv2.imread(str(image_path))
    image = Image.open(image_path).convert("RGB").resize((512, 512), Image.NEAREST)
    image = np.array(image)


    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    

    results = detect_cracks_traditional(image, sensitivity)

    stem = image_path.stem
    ensure_dir(output_dir)

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

    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    return sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_ext])


def main():
    parser = argparse.ArgumentParser(description="Crack Mask Generator")

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="traditional_outputs")
    parser.add_argument("--sensitivity", type=float, default=85.0)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images = collect_images(input_path)

    if not images:
        raise ValueError("No images found.")

    for img in images:
        process_one_image(img, output_dir, args.sensitivity)

    print(f"\nDone. Results saved in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
