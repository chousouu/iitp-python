"""File with tests for Fast Hough Transform."""

from typing import TypeVar

import cv2
import numpy as np

from fasthough.draw_line import hough_transform, read_image  # type: ignore

FOLDER: str = "/home/yasin/py/iitp-python/iitp-python/src/fasthough/images/"
SelfShape = TypeVar("SelfShape", bound="TestClass")


def _transform_pixels(array: np.ndarray) -> np.ndarray:
    transformed_array = np.where(array != 0, 64, array)
    return transformed_array


def _transform_rgb_pixels(array: np.ndarray) -> np.ndarray:
    array_1d = array.flatten()
    unique_values, counts = np.unique(array_1d, return_counts=True)
    most_common_index = np.argmax(counts)
    most_common_value = unique_values[most_common_index]
    transformed_array = np.where(array != most_common_value, 64, array)
    return transformed_array


def compare_images_with_lines(
    image1: np.ndarray, image2: np.ndarray, threshold: float = 0.7
) -> bool:
    """Function to compare difference between 'image1' and 'image2'."""
    image1 = _transform_pixels(image1)
    correlation = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)[0][0]
    return correlation >= threshold


def compare_rgb_images_with_lines(
    image1: np.ndarray, image2: np.ndarray, threshold: float = 0.7
) -> bool:
    """Function to compare difference between rgb 'image1' and 'image2'."""
    image1 = _transform_rgb_pixels(image1)
    correlation = cv2.matchTemplate(image1, image2, cv2.TM_CCOEFF_NORMED)[0][0]
    return correlation >= threshold


def fill_blanks_rows(array_2d: np.ndarray) -> np.ndarray:
    """Fills blanks rows."""
    nonzero_rows = np.nonzero(np.sum(array_2d, axis=1))[0]

    for row in nonzero_rows:
        value = array_2d[row, np.nonzero(array_2d[row])[0][0]]
        array_2d[row][array_2d[row] == 0] = value
    return array_2d


def fill_blanks_cols(array_2d: np.ndarray) -> np.ndarray:
    """Fills blanks columns."""
    nonzero_cols = np.nonzero(np.sum(array_2d, axis=0))[0]
    for col in nonzero_cols:
        value = array_2d[np.nonzero(array_2d[:, col])[0][0], col]
        array_2d[:, col][array_2d[:, col] == 0] = value
    return array_2d


# In these tests we compare the lines from the original picture
# with the lines drawn after the function hough_transform
class TestClass:
    """Test class for hough transform."""

    def test_horizontal(self: SelfShape) -> bool:
        """Test 1. Horizontal lines."""
        img_new = read_image(FOLDER + "horizontal_line.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_vertical(self: SelfShape) -> bool:
        """Test 2. Vertical lines."""
        img_new = read_image(FOLDER + "vertical_line.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_small_hor(self: SelfShape) -> bool:
        """Test 3. Small horizontal lines."""
        img_new = read_image(FOLDER + "horizontal_line_3x3.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_small_ver(self: SelfShape) -> bool:
        """Test 4. Small vertical lines."""
        img_new = read_image(FOLDER + "vertical_line_3x3.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_long_hor_1(self: SelfShape) -> bool:
        """Test 5. Long horizontal lines."""
        img_new = read_image(FOLDER + "horizontal_line_10x500.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_long_hor_2(self: SelfShape) -> bool:
        """Test 6. Long vertical lines."""
        img_new = read_image(FOLDER + "vertical_line_500x10.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_many_lines(self: SelfShape) -> bool:
        """Test 7. Multiple lines."""
        img_new = read_image(FOLDER + "lines.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_many_rotated_lines(self: SelfShape) -> bool:
        """Test 8. Multiple rotated lines."""
        img_new = read_image(FOLDER + "image_2.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_blured(self: SelfShape) -> bool:
        """Test 9. Blurred images."""
        img_new = read_image(FOLDER + "blured.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_rectangle(self: SelfShape) -> bool:
        """Test 10. Size crop."""
        img_new = read_image(FOLDER + "rectangle1.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_rotated(self: SelfShape) -> bool:
        """Test 11. Non vertical/horizontal lines."""
        img_new = read_image(FOLDER + "rotated.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_intermittent_2_small_holes(self: SelfShape) -> bool:
        """Test 12. Lines with holes."""
        img_new = read_image(FOLDER + "intermittent.png")
        line_img = hough_transform(img_new)
        img = fill_blanks_rows(img_new)
        assert compare_images_with_lines(img, line_img)

    def test_intermittent_4_wide_holes(self: SelfShape) -> bool:
        """Test 13. Lines with holes."""
        img_new = read_image(FOLDER + "intermittent2.png")
        line_img = hough_transform(img_new)
        img = fill_blanks_rows(img_new)
        assert compare_images_with_lines(img, line_img)

    def test_intermittent_vertical(self: SelfShape) -> bool:
        """Test 14. Lines with holes."""
        img_new = read_image(FOLDER + "intermittent3.png")
        line_img = hough_transform(img_new)
        img = fill_blanks_cols(img_new)
        assert compare_images_with_lines(img, line_img)

    def test_wide(self: SelfShape) -> bool:
        """Test 15. Horizontal wide line."""
        img_new = read_image(FOLDER + "wide_line.png")
        line_img = hough_transform(img_new)
        assert compare_images_with_lines(img_new, line_img)

    def test_rgb_1(self: SelfShape) -> bool:
        """Test 16. RGB image."""
        img_new = read_image(FOLDER + "rgb1.png")
        line_img = hough_transform(img_new)
        assert compare_rgb_images_with_lines(img_new, line_img)

    def test_rgb_2(self: SelfShape) -> bool:
        """Test 17. RGB image."""
        img_new = read_image(FOLDER + "rgb2.png")
        line_img = hough_transform(img_new)
        assert compare_rgb_images_with_lines(img_new, line_img)

    def test_rgb_3(self: SelfShape) -> bool:
        """Test 18. RGB image."""
        img_new = read_image(FOLDER + "rgb3.png")
        line_img = hough_transform(img_new)
        assert compare_rgb_images_with_lines(img_new, line_img)

    def test_jpg(self: SelfShape) -> bool:
        """Test 19. JPG extension."""
        img_new = read_image(FOLDER + "horizontal_line.jpg")
        line_img = hough_transform(img_new)
        assert compare_rgb_images_with_lines(img_new, line_img)

    def test_jpeg(self: SelfShape) -> bool:
        """Test 20. JPEG extension."""
        img_new = read_image(FOLDER + "horizontal_line.jpeg")
        line_img = hough_transform(img_new)
        assert compare_rgb_images_with_lines(img_new, line_img)
