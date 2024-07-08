from typing import (
    Dict,
)
import json
import numpy as np
import cv2
from skimage import measure
from scipy import ndimage

def load_json(file_path: str) -> Dict:
    """Load a json file as dictionary."""
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_image(image, min_value, max_value):
    """
    Normalize an image to a custom minimum and maximum value.

    Parameters:
    - image: numpy.ndarray
        The input image to be normalized.
    - min_value: float
        The desired minimum value for the normalized image.
    - max_value: float
        The desired maximum value for the normalized image.

    Returns:
    - normalized_image: numpy.ndarray
        The normalized image with values in the range [min_value, max_value].
    """
    # Find the current minimum and maximum values in the image
    current_min = np.min(image)
    current_max = np.max(image)

    # Normalize the image to the desired range
    normalized_image = (image - current_min) / (current_max - current_min) * (max_value - min_value) + min_value

    return normalized_image


def round_mask(mask, roundness_factor=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (roundness_factor, roundness_factor))
    # Perform morphological closing to round the mask
    rounded_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return rounded_mask

def fill_gaps(mask):
    inverted_mask = 1.0 - mask
    contours, _ = cv2.findContours((inverted_mask * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled_mask = np.ones_like(mask)
    cv2.drawContours(filled_mask, contours, -1, 0, thickness=cv2.FILLED)
    return filled_mask

def fill_gaps_with_morphology(mask, iterations=1):
    # Perform dilation to fill gaps
    filled_mask = ndimage.binary_fill_holes(mask).astype(int)
    # kernel = np.ones((10, 10), np.uint8)
    # filled_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

    # # Convert back to binary
    # filled_mask = filled_mask.astype(bool)
    return filled_mask

def remove_small_areas(binary_mask, max_size=10):
    labeled_mask, num_features = ndimage.label(binary_mask)
    u, counts = np.unique(labeled_mask, return_counts=True)
    for i, val in enumerate(u):
        if counts[i] <= max_size:
            mask = labeled_mask == val
            binary_mask[mask] = 0
    return binary_mask

