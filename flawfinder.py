import hashlib
import os
import subprocess
import shutil
import sys

import cv2
import pandas as pd
from PIL import Image, ExifTags
from openpyxl import Workbook
import numpy as np
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    image_extensions = ('.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.jp2', '.png')
    for root, _, files in os.walk(directory):
        # Check if the current directory is within the specified depth
        if root[len(directory):].count(os.sep) >= depth:
            continue
        for file in files:
            if file.lower().endswith(image_extensions):
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


def draw_boxes(image_path, black_pixels, white_pixels, output_path):
    image = cv2.imread(image_path)
    for y, x in zip(*black_pixels):
        cv2.rectangle(image, (x, y), (x+10, y+10), (0, 255, 0), 10)  # Green box with 10-pixel thickness
    for y, x in zip(*white_pixels):
        cv2.rectangle(image, (x, y), (x+10, y+10), (0, 0, 255), 10)  # Red box with 10-pixel thickness
    cv2.imwrite(output_path, image)


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
        # Run the identify command
        result = subprocess.run(
            ["identify", "-format", "%Q", image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Check if the command was successful
        if result.returncode == 0:
            return int(result.stdout.strip())
        else:
            print(f"Error: {result.stderr.strip()}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None


def process_image(image_path, output_dir, pixels_threshold):
    cv2image = cv2.imread(image_path)
    pil_image = Image.open(image_path)
    is_valid = is_image_valid(pil_image)
    # is_truncated = is_image_truncated(pil_image)
    is_overexposed = is_image_overexposed(cv2image, image_path) if is_valid else None
    exif_data = extract_exif(pil_image, image_path) if is_valid else {}
    compression_ratio = calculate_compression_ratio(pil_image, image_path) if is_valid and image_path.lower().endswith(('.jpg', '.jpeg')) else None
    jpeg_quality = get_jpeg_quality_with_identify(image_path) if is_valid and image_path.lower().endswith(('.jpg', '.jpeg')) else None
    black_pixels, white_pixels = find_black_white_pixels(cv2image, image_path) if is_valid else ([], [])
    file_size = calculate_file_size(image_path)
    md5sum = calculate_md5(image_path)
    sha512 = calculate_sha512(image_path)
    box, angle = extract_bounding_box_and_normalize_angle(binarize_image(cv2image, image_path)) if is_valid else (None, None)
    color_profile = calculate_color_profile(cv2image, image_path, output_dir) if is_valid else None
    color_histograms = calculate_color_histograms(cv2image, image_path, output_dir) if is_valid else None

    # calculate_and_save_histogram_image(cv2image, image_path, output_dir) if is_valid else None

    if is_valid and (black_pixels[0].size > pixels_threshold or white_pixels[0].size > pixels_threshold):
        output_path = os.path.join(output_dir, f"annotated_{os.path.basename(image_path)}")
        draw_boxes(image_path, black_pixels, white_pixels, output_path)

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


def main():
    # Check if ImageMagick's identify is available
    if shutil.which("identify") is None:
        print(
            "Error: ImageMagick is not installed or 'identify' is not in your PATH.\n"
            "Install it on Linux with:\n"
            "  sudo apt-get install imagemagick\n"
            "or see https://imagemagick.org/script/download.php"
        )
        sys.exit(1)
    parser = argparse.ArgumentParser(
        description='Analyze scanned images for flaws, extract metadata, and annotate images with detected black/white pixels.'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Path to the directory containing input images to be analyzed.'
    )
    parser.add_argument(
        '--output_excel',
        type=str,
        required=True,
        help='Path to the Excel file where extracted metadata will be saved.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory where annotated images and analysis results will be stored.'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=16,
        help='Number of parallel threads to use for image processing (default: 16).'
    )
    parser.add_argument(
        '--pixels_threshold',
        type=int,
        default=5,
        help='Minimum number of black or white pixels required to annotate an image (default: 5).'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=3,
        help='Depth of the directory structure to search for images (default: 3).'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    image_files = list(find_image_files(args.input_dir, args.depth))
    results = []

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {executor.submit(process_image, image_path, args.output_dir, args.pixels_threshold): image_path for image_path in image_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"Error processing {futures[future]}: {e}")

    # Save results to Excel
    df = pd.DataFrame(results)
    df.to_excel(args.output_excel, index=False)
    print(f"Metadata saved to {args.output_excel}")

    # Calculate and print the average JPEG compression ratio
    compression_ratios = [result["Compression Ratio"] for result in results if result["Compression Ratio"] is not None]
    if compression_ratios:
        average_compression = sum(compression_ratios) / len(compression_ratios)
        print(f"Average JPEG Compression Ratio: {average_compression:.2f}")
    else:
        print("No valid JPEG compression ratios found.")

if __name__ == "__main__":
    main()