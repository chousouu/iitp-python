"""Example how to use implemented fast hough transform."""

import matplotlib.pyplot as plt

from .draw_line import read_image, hough_transform

if __name__ == "__main__":
    FOLDER = "images/"
    print("Input picture file name from /images folder:")

    pic_name = input()
    img_new = read_image(FOLDER + pic_name)

    line_img = hough_transform(img_new)

    plt.imshow(line_img)
    plt.set_figwidth(12)
    plt.set_figheight(12)
    plt.suptitle("Hough Lines Image")
    plt.savefig(f"test_output/line_img_{pic_name}")
