import hashlib
import os
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ExifTags

# I-004: cap decoded image size (decompression-bomb guard).
MAX_PIXELS = 50_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Shared with the upload API so CLI and server agree on what is an image.
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.jp2', '.png')
# Same shape as the (rows, cols) tuple returned by np.where() on a 2D mask.
# Used as the stand-in for black/white pixel indices when an image is invalid,
# so indexing (e.g. black_pixels[0].size) keeps working and reports 0.
_EMPTY_PIXELS = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

def calculate_and_save_histogram_image(cv2_image, image_path, output_dir):
    """
    Calculate color histograms for an image and save them as an image.

    Args:
        cv2_image (numpy.ndarray): The input image in BGR format.
        image_path (str): Path to the input image file.
        output_dir (str): Directory to save the histogram image.

    Returns:
        str: Path to the saved histogram image.
    """
    if cv2_image is None:
        raise ValueError(f"Image not found or could not be loaded at {image_path}")

    # Create a blank image for the histogram
    hist_image = np.zeros((300, 256, 3), dtype=np.uint8)

    # Colors for each channel
    colors = ['B', 'G', 'R']
    channel_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, color in enumerate(colors):
        # Calculate histogram for the channel
        hist = cv2.calcHist([cv2_image], [i], None, [256], [0, 256])
        cv2.normalize(hist, hist, 0, hist_image.shape[0], cv2.NORM_MINMAX)

        # Draw the histogram
        for x in range(1, 256):
            cv2.line(hist_image,
                     (x - 1, 300 - int(hist[x - 1])),
                     (x, 300 - int(hist[x])),
                     channel_colors[i], 1)

    # Save the histogram image
    output_file = os.path.join(output_dir, f"{os.path.basename(image_path)}_histogram.png")
    cv2.imwrite(output_file, hist_image)

    return output_file

def binarize_image(cv2_image, image_path):
    """
    Load an image, convert it to grayscale, and binarize it.
    """
    if cv2_image is None:
        raise ValueError(f"Image not found or could not be loaded at {image_path}")

    gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def extract_bounding_box_and_normalize_angle(binary_image):
    """
    Extract the bounding box and normalize the angle to be close to 0.

    Args:
        binary_image (numpy.ndarray): Binary image.

    Returns:
        tuple: Bounding box and normalized angle.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("No contours found in the image")

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.intp)

    # Normalize the angle
    angle = rect[2]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    return box, angle


def find_image_files(directory, depth=3):
    for root, _, files in os.walk(directory):
        # Check if the current directory is within the specified depth
        if root[len(directory):].count(os.sep) >= depth:
            continue
        for file in files:
            if file.lower().endswith(IMAGE_EXTENSIONS):
                yield os.path.join(root, file)

def extract_exif(pil_image, image_path):
    try:
        exif_data = {
            ExifTags.TAGS[k]: v
            for k, v in pil_image._getexif().items()
            if k in ExifTags.TAGS
        } if pil_image._getexif() else {}
        dpi = pil_image.info.get('dpi', (None, None))  # Extract DPI if available
        exif_data['DPI'] = dpi
        return exif_data
    except Exception:
        return {}

def calculate_compression_ratio(pil_image, image_path):
    try:
        raw_size = pil_image.size[0] * pil_image.size[1] * 3  # Assuming 3 bytes per pixel (RGB)
        file_size = os.path.getsize(image_path)
        return raw_size / file_size if file_size > 0 else None
    except Exception:
        return None


def find_black_white_pixels(cv2image, image_path):
    black_pixels = np.where((cv2image == [0, 0, 0]).all(axis=2))
    white_pixels = np.where((cv2image == [255, 255, 255]).all(axis=2))
    return black_pixels, white_pixels


def draw_boxes(image, black_pixels, white_pixels, output_path):
    """
    Annotate an image by highlighting pure-black (green) and pure-white (red) pixels.

    The previous implementation drew one thick cv2.rectangle per flagged pixel,
    which took minutes on large scans and produced solid color blocks. Building a
    mask and dilating it scales with the image size instead of the number of
    flagged pixels.

    Args:
        image (numpy.ndarray): The already-decoded input image in BGR format.
        black_pixels (tuple): (rows, cols) index arrays of pure-black pixels.
        white_pixels (tuple): (rows, cols) index arrays of pure-white pixels.
        output_path (str): Path to save the annotated image.
    """
    if image is None:
        raise ValueError("Image not found or could not be loaded")

    # 1 = pure-black pixel, 2 = pure-white pixel. Assigning white after black
    # keeps white on top where the two overlap, matching the old behaviour.
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[black_pixels[0], black_pixels[1]] = 1
    mask[white_pixels[0], white_pixels[1]] = 2
    dilated = cv2.dilate(mask, np.ones((11, 11), np.uint8))  # ~10-px box around each pixel

    annotated = image.copy()
    annotated[dilated == 1] = (0, 255, 0)   # Green where black pixels were
    annotated[dilated == 2] = (0, 0, 255)   # Red where white pixels were
    cv2.imwrite(output_path, annotated)


def is_image_overexposed(cv2image, image_path, threshold=70):
    """
    Check if an image has too much lighting (overexposed).

    Args:
        image_path (str): Path to the image file.
        threshold (float): Percentage of bright pixels to classify as overexposed.

    Returns:
        bool: True if the image is overexposed, False otherwise.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found or could not be loaded at {image_path}")

    # Calculate the percentage of bright pixels
    bright_pixels = np.sum(image >= 240)  # Pixels close to 255
    total_pixels = image.size
    bright_percentage = (bright_pixels / total_pixels) * 100

    return bright_percentage > threshold


def calculate_color_profile(cv2_image, image_path, output_dir):
    """
    Calculate the RGB color profile of an image and save it to a file.

    Args:
        image_path (str): Path to the image file.
        output_dir (str): Directory to save the color profile file.

    Returns:
        dict: Average RGB values as the color profile.
    """
    if cv2_image is None:
        raise ValueError(f"Image not found or could not be loaded at {image_path}")

    # Calculate the average color in each channel
    avg_color_per_channel = cv2.mean(cv2_image)[:3]  # Ignore the alpha channel if present
    color_profile = {
        "R": avg_color_per_channel[2],  # OpenCV uses BGR, so reverse to RGB
        "G": avg_color_per_channel[1],
        "B": avg_color_per_channel[0],
    }

    # Save the color profile to a file
    output_file = os.path.join(output_dir, f"{os.path.basename(image_path)}_color_profile.txt")
    with open(output_file, "w") as f:
        f.write(f"Color Profile (RGB):\n")
        f.write(f"R: {color_profile['R']:.2f}\n")
        f.write(f"G: {color_profile['G']:.2f}\n")
        f.write(f"B: {color_profile['B']:.2f}\n")

    return color_profile


def calculate_color_histograms(cv2_image, image_path, output_dir):
    """
    Calculate color histograms for an image and save them to a file.

    Args:
        image_path (str): Path to the image file.
        output_dir (str): Directory to save the histogram file.

    Returns:
        dict: Histograms for R, G, and B channels.
    """
    image = cv2_image

    # Calculate histograms for each channel
    histograms = {}
    for i, color in enumerate(['B', 'G', 'R']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        hist = hist.flatten()  # Flatten the histogram for easier storage
        histograms[color] = hist

    # Save histograms to a file
    output_file = os.path.join(output_dir, f"{os.path.basename(image_path)}_histograms.txt")
    with open(output_file, "w") as f:
        for color, hist in histograms.items():
            f.write(f"{color} Histogram:\n")
            f.write(", ".join(map(str, hist)) + "\n")

    return histograms


def get_jpeg_quality_with_identify(image_path):
    """
    Use ImageMagick's identify command to get the JPEG quality.

    Args:
        image_path (str): Path to the JPEG image.

    Returns:
        int or None: JPEG quality level, or None if not applicable.
    """
    try:
        # Run the identify command (I-005: bounded, so a hung identify
        # can never stall a worker thread forever).
        result = subprocess.run(
            ["identify", "-format", "%Q", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            print(f"Error: {result.stderr.strip()}")
            return None
    except subprocess.TimeoutExpired:
        print("Warning: identify timed out after 30s")
        return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def process_image(image_path, output_dir, pixels_threshold):
    """
    Run all per-image measurements and return one result row.

    Every substep is isolated by safe(), so a single failure (e.g. "No contours
    found" on a pure-black image, or a file PIL cannot parse) is recorded in the
    "Errors" column instead of raising and dropping the whole row.
    """
    errors = []

    def safe(name, fn, default=None):
        try:
            return fn()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            return default

    def _probe_pixel_count(image_path):
        try:
            with Image.open(image_path) as im:
                return im.width * im.height
        except Exception:
            return None

    pixel_count = safe("size probe", lambda: _probe_pixel_count(image_path))
    if pixel_count is not None and pixel_count > MAX_PIXELS:
        errors.append(f"image too large: {pixel_count} pixels > {MAX_PIXELS}")
        return {
            "File Path": image_path,
            "Is Valid": False,
            "Is Overexposed": None,
            "File Size (bytes)": safe("file size", lambda: calculate_file_size(image_path)),
            "MD5 Checksum": None,
            "SHA-512 Hash": None,
            "EXIF Metadata": {},
            "DPI": (None, None),
            "Compression Ratio": None,
            "JPEG Quality": None,
            "Black Pixels": 0,
            "White Pixels": 0,
            "Angle": None,
            "Color Profile": None,
            "Color Histograms": None,
            "Errors": "; ".join(errors),
        }

    cv2image = safe("cv2 load", lambda: cv2.imread(image_path))
    pil_image = safe("PIL load", lambda: Image.open(image_path))
    is_valid = is_image_valid(pil_image) if pil_image is not None else False
    # is_truncated = is_image_truncated(pil_image)
    is_jpeg = image_path.lower().endswith(('.jpg', '.jpeg'))
    is_overexposed = safe("overexposure check", lambda: is_image_overexposed(cv2image, image_path)) if is_valid else None
    exif_data = safe("EXIF", lambda: extract_exif(pil_image, image_path), default={}) if is_valid else {}
    compression_ratio = safe("compression ratio", lambda: calculate_compression_ratio(pil_image, image_path)) if is_valid and is_jpeg else None
    jpeg_quality = safe("JPEG quality", lambda: get_jpeg_quality_with_identify(image_path)) if is_valid and is_jpeg else None
    if is_valid:
        black_pixels, white_pixels = safe("black/white pixels", lambda: find_black_white_pixels(cv2image, image_path), default=(_EMPTY_PIXELS, _EMPTY_PIXELS))
    else:
        black_pixels, white_pixels = _EMPTY_PIXELS, _EMPTY_PIXELS
    file_size = safe("file size", lambda: calculate_file_size(image_path))
    md5sum = safe("MD5", lambda: calculate_md5(image_path))
    sha512 = safe("SHA-512", lambda: calculate_sha512(image_path))
    box, angle = safe("bounding box", lambda: extract_bounding_box_and_normalize_angle(binarize_image(cv2image, image_path)), default=(None, None)) if is_valid else (None, None)
    color_profile = safe("color profile", lambda: calculate_color_profile(cv2image, image_path, output_dir)) if is_valid else None
    color_histograms = safe("color histograms", lambda: calculate_color_histograms(cv2image, image_path, output_dir)) if is_valid else None

    # calculate_and_save_histogram_image(cv2image, image_path, output_dir) if is_valid else None

    if is_valid and (black_pixels[0].size > pixels_threshold or white_pixels[0].size > pixels_threshold):
        output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        safe("annotation", lambda: draw_boxes(cv2image, black_pixels, white_pixels, output_path))

    return {
        "File Path": image_path,
        "Is Valid": bool(is_valid),
        # "Is Truncated": is_truncated,
        "Is Overexposed": is_overexposed,
        "File Size (bytes)": file_size,
        "MD5 Checksum": md5sum,
        "SHA-512 Hash": sha512,
        "EXIF Metadata": exif_data,
        "DPI": exif_data.get('DPI', (None, None)) if is_valid else (None, None),
        "Compression Ratio": compression_ratio,
        "JPEG Quality": jpeg_quality,
        "Black Pixels": black_pixels[0].size,
        "White Pixels": white_pixels[0].size,
        "Angle": angle if is_valid else None,
        "Color Profile": color_profile,
        "Color Histograms": color_histograms,
        "Errors": "; ".join(errors),
    }

def calculate_file_size(image_path):
    try:
        return os.path.getsize(image_path)
    except Exception:
        return None


def calculate_md5(image_path):
    try:
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def calculate_sha512(image_path):
    try:
        hash_sha512 = hashlib.sha512()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha512.update(chunk)
        return hash_sha512.hexdigest()
    except Exception:
        return None


def is_image_valid(pil_image):
    try:
        pil_image.verify()  # Verify the image integrity
        return True
    except Exception:
        return False


def check_imagemagick() -> None:
    """Raise a clear error when ImageMagick 'identify' is missing."""
    if shutil.which("identify") is None:
        raise RuntimeError(
            "ImageMagick is not installed or 'identify' is not in your PATH.\n"
            "Install it on Linux with:\n"
            "  sudo apt-get install imagemagick\n"
            "or see https://imagemagick.org/script/download.php"
        )


def formula_safe(value):
    """I-006: prefix spreadsheet-formula characters so openpyxl stores text."""
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def jsonable(value):
    """Convert measurement values (tuples, numpy scalars) to JSON-safe types."""
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
