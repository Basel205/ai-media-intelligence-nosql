import cv2
import numpy as np


def compute_blur_score(image):
    """
    Blur detection using Laplacian variance.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()

    return float(score)


def compute_brightness(image):
    """
    Compute average brightness.
    """

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    brightness = np.mean(hsv[:, :, 2])

    return float(brightness / 255)


def extract_metadata(image_path):
    """
    Extract metadata from an image.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    height, width = image.shape[:2]

    metadata = {
        "width": width,
        "height": height,
        "brightness": compute_brightness(image),
        "blur_score": compute_blur_score(image)
    }

    return metadata