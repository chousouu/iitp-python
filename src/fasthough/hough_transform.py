"""Fast Hough Transform (FHT) implementation."""

import numpy as np


def _merge_histograms(hist1: np.ndarray, hist2: np.ndarray) -> np.ndarray:
    height = hist1.shape[0]
    width = 1 if len(hist1.shape) == 1 else hist1.shape[1]
    n0 = width * 2
    r = (width - 1) / (n0 - 1)
    hist1 = hist1.reshape((height, width))
    hist2 = hist2.reshape((height, width))

    result = np.zeros((height, n0))
    for t in range(n0):
        t0 = int(t * r)
        s = t - t0
        result[:, t] = hist1[:, t0] + np.concatenate(
            [hist2[s:height, t0], hist2[0:s, t0]], axis=0
        )
    return result


def _fast_hough_transform(img: np.ndarray) -> np.ndarray:
    n = img.shape[1]
    if n < 2:
        return img[:, 0]

    return _merge_histograms(
        _fast_hough_transform(img[:, 0 : int(n / 2)]),
        _fast_hough_transform(img[:, int(n / 2) : n]),
    )


def hough_image(img: np.ndarray) -> np.ndarray:
    """Returns Dyadetic pattern of image acquired by Fast Hough Transform.

    Args:
        img: image represented as 2D or 3D array

    Returns:
        Image containing edges and gradient of the image
    """
    return _fast_hough_transform(img)


def find_max_value(img: np.ndarray) -> tuple:
    """Returns max value over the image.

    Args:
        img: image represented as 2D or 3D array

    Returns:
        Max value over image and its position
    """
    intensity = 0
    s, t = 0, 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > intensity:
                intensity = img[i, j]
                s = i
                t = j
    return intensity, (s, t)
