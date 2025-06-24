# jennie

Jennie finds flaws in scanned documents.

## Installation

Clone the repository and install the dependencies:

```bash
git clone git@github.com:rvankoert/jennie.git
cd jennie
pip install -r requirements.txt
```

install imagemagick on your system, e.g. on Ubuntu:

```bash
sudo apt install imagemagick
```

## Arguments

| Argument             | Description                                                                                  | Required | Default |
|----------------------|----------------------------------------------------------------------------------------------|----------|---------|
| `--input_dir`        | Path to the directory containing input images to be analyzed.                                 | Yes      |         |
| `--output_excel`     | Path to the Excel file where extracted metadata will be saved.                               | Yes      |         |
| `--output_dir`       | Directory where annotated images and analysis results will be stored.                        | Yes      |         |
| `--threads`          | Number of parallel threads to use for image processing.                                      | No       | 16      |
| `--pixels_threshold` | Minimum number of black or white pixels required to annotate an image.                       | No       | 5       |
| `--depth`            | Maximum depth of subdirectories to search for images.                                        | No       | 3       |

## Example Usage

```bash
python flawfinder.py \
  --input_dir ./input_images \
  --output_excel ./results/metadata.xlsx \
  --output_dir ./results/annotated \
  --threads 8 \
  --pixels_threshold 10