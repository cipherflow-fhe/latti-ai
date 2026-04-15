"""Standalone image preprocessor: convert a raw image to model-input CSV.

Reproduces the exact test-time preprocessing from DRDataset (transform=False):
  1. Load image as RGB via PIL
  2. Resize to (img_size, img_size) using bilinear interpolation
  3. Scale pixel values to [0, 1]  (divide by 255)
  4. Transpose from (H, W, C) to (C, H, W)
  5. (Optional) Apply ImageNet mean/std normalization
  6. Flatten each channel row-major -> shape (C, H*W)
  7. Save as CSV with float64 precision
  当前默认保存的图片是：/home/linghm/latti-ai-fix/latti-ai/examples/test_retina_dkd/data/cls_dkd/Test/DN/y30494450-142307.jpg
Usage:
    python preprocess_image.py --image path/to/image.jpg --input-shape 128
    python preprocess_image.py --image /home/linghm/latti-ai-fix/latti-ai/examples/test_retina_dkd/data/cls_dkd/Test/DN/y30494450-142307.jpg   --input-shape 128 --normalize
    python preprocess_image.py --image path/to/image.jpg --input-shape 128 --output out.csv
"""

import argparse
import os

import numpy as np
from PIL import Image


IMAGENET_MEAN = [0.485, 0.456, 0.406]  # per-channel, RGB order
IMAGENET_STD = [0.229, 0.224, 0.225]


def preprocess(image_path: str, img_size: int, normalize: bool = False) -> np.ndarray:
    """Load and preprocess a single image.

    Args:
        image_path:  Path to the input image file.
        img_size:    Target spatial size (both height and width).
        normalize:   Whether to apply ImageNet mean/std normalization.

    Returns:
        np.ndarray of shape (C, H*W), dtype float64.
    """
    # 1. Load as RGB
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    # 2. Resize to (img_size, img_size) — bilinear (PIL default)
    img = img.resize((img_size, img_size), Image.BILINEAR)

    # 3. To numpy, scale to [0, 1]
    image = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)

    # 4. HWC -> CHW
    image = np.transpose(image, (2, 0, 1))  # (C, H, W)

    # 5. Optional ImageNet normalization
    if normalize:
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std

    # 6. Flatten each channel: (C, H, W) -> (C, H*W)
    C, H, W = image.shape
    image_flat = image.reshape(C, H * W).astype(np.float64)

    return image_flat


def main():
    parser = argparse.ArgumentParser(description='Preprocess an image to model-input CSV')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--input-shape', type=int, default=128, help='Target image size (default: 128)')
    parser.add_argument('--normalize', action='store_true', default=False, help='Apply ImageNet mean/std normalization')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (default: <image_stem>_input.csv)')
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f'Error: image not found: {args.image}')
        return

    # Preprocess
    image_flat = preprocess(args.image, args.input_shape, args.normalize)
    C, HW = image_flat.shape
    H = W = args.input_shape

    # Determine output path
    if args.output:
        csv_path = args.output
    else:
        stem = os.path.splitext(os.path.basename(args.image))[0]
        csv_path = stem + '_input.csv'

    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None

    # Save
    np.savetxt(csv_path, image_flat, delimiter=',')
    print(f'Saved: {csv_path}')
    print(f'  shape: ({C}, {HW})  =  {C} channels x ({H} x {W}) pixels')
    print(f'  dtype: float64')
    print(f'  normalize: {args.normalize}')


if __name__ == '__main__':
    main()
