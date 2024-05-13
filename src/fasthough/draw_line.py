"""Module to draw hough transformed images."""

import cv2 as cv
import numpy as np

from .hough_transform import find_max_value, hough_image


def draw_line(
    img: np.ndarray, s: int, t: int, color: tuple = (64, 0, 0), lineThickness: int = 2
) -> np.ndarray:
    """Returns copy of 'img' with drawn line with given parameters s, t.

    Args:
        img: image represented as 2D or 3D array.
        s: line intercept.
        t: line slope.
        color: drawn line color as 3D array.
        lineThickness: thickness of drawn line.

    Returns:
        Copy of image with line.

    Raises:
        ValueError: if provided 'color' argument is not length 3.
    """
    image = img.copy()
    height, width = image.shape
    if len(color) != 3:
        raise ValueError("Color length must be 3! (red, green, blue).")

    if s + t > height:
        x_0 = _find_point(height, s, t)
        zero_matrix = np.zeros((height, width))
        stacked_upper = np.vstack([zero_matrix, image])[s : s + height, :]
        stacked_lower = np.vstack([image, zero_matrix])[s : s + height, :]
        max_up = find_max_value(hough_image(stacked_upper))[0]
        max_low = find_max_value(hough_image(stacked_lower))[0]

        if max_up > max_low:
            first_point = (x_0, 0)
            second_point = (width - 1, s + t - height)
        else:
            first_point = (x_0, height)
            second_point = (0, s)
    else:
        first_point = (0, s)
        second_point = (width - 1, s + t)

    cv.line(image, first_point, second_point, color, lineThickness)
    return image


def hough_transform(img: np.ndarray, threshold_ratio: float = 0.9) -> np.ndarray:
    """Returns processed 'img' with edges and gradients.

    Args:
        img: image represented as 2D or 3D array.
        threshold_ratio: threshold ratio to exclude odd lines on processsed image.

    Returns:
        Copy of image with line.
    """
    hough_img = hough_image(img)
    max_intensity, _ = find_max_value(hough_img)
    threshold = max_intensity * threshold_ratio
    lines = np.argwhere(hough_img >= threshold)

    lined_image = img.copy()
    for s, t in lines:
        lined_image = draw_line(lined_image, s, t)
    return lined_image


def _find_point(n: int, s: int, t: int) -> int:
    return round((n - s) * n / t)


def read_image(path: str) -> np.ndarray:
    """Returns image with new shape as a multiple of 2^k.

    Args:
        path: image path.

    Returns:
        image as numpy array.
    """
    final_img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    h, w = final_img.shape[:2]

    new_h = 2 ** int(np.ceil(np.log2(h)))
    new_w = 2 ** int(np.ceil(np.log2(w)))

    resized_img = cv.resize(final_img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    return resized_img
