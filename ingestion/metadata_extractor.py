import cv2
import numpy as np


def compute_blur_score(image) -> float:
    """
    Sharpness detection using Laplacian variance.
    Higher value = sharper image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image) -> float:
    """
    Mean brightness from HSV value channel, normalised to [0, 1].
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]) / 255.0)


def extract_metadata(image_path: str) -> dict | None:
    """
    Extract visual metadata from an image file.
    Returns None if the image cannot be read.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    height, width = image.shape[:2]

    return {
        "width":      width,
        "height":     height,
        "brightness": compute_brightness(image),
        "blur_score": compute_blur_score(image),
        "aspect_ratio": round(width / height, 3)
    }
