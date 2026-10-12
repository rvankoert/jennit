"""jennit analyze — the original CLI (moved from flawfinder.py main())."""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ..analysis import check_imagemagick, find_image_files, formula_safe, process_image


def main():
    try:
        check_imagemagick()
    except RuntimeError as exc:
        print(f"Error: {exc}")
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

    # Save results to Excel (I-006: formula-safe string cells)
    df = pd.DataFrame([{k: formula_safe(v) for k, v in row.items()} for row in results])
    df.to_excel(args.output_excel, index=False)
    print(f"Metadata saved to {args.output_excel}")

    # Calculate and print the average JPEG compression ratio
    compression_ratios = [result["Compression Ratio"] for result in results if result["Compression Ratio"] is not None]
    if compression_ratios:
        average_compression = sum(compression_ratios) / len(compression_ratios)
        print(f"Average JPEG Compression Ratio: {average_compression:.2f}")
    else:
        print("No valid JPEG compression ratios found.")
