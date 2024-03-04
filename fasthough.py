import numpy as np
from pathlib import Path
from PIL.Image import Image as PIL_Image
from PIL import Image, ImageDraw
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt

def fast_hough_transform(image):
    imgh, imgw = image.shape
    hough_levels = imgw.bit_length() + 1
    
    h = imgh
    w = 2 ** (hough_levels - int(imgw == (2**hough_levels)))
    
    hough = np.zeros((h, w))
    hough[:imgh, :imgw] = image
    
    for level in range(1, hough_levels):
        bs = 2 ** level # block size
        new_hough = np.zeros((h, w))
        for x in range(bs):
            shift = -((x + 1) // 2)
            lpos = x // 2
            rpos = (bs + x) // 2
            left_values = hough[:, lpos::bs]
            right_values = np.roll(hough[:, rpos::bs], shift, axis=0)
            new_hough[:, x::bs] = left_values + right_values
        hough = new_hough
    result = hough[:imgh, :imgw]
    return MinMaxImg(result)    

def MinMaxImg(img):
    return (img - img.min()) / (img.max() - img.min())

def get_grayscale_img(img):
    if isinstance(img, np.ndarray):
        tmp_img = Image.fromarray(img)
    elif isinstance(img, PIL_Image):
        tmp_img = img.copy()
    else:
        tmp_img = Image.open(img)
    return np.array(tmp_img.convert(mode='L'))

def get_horiz_line(hough_image, width, max_clusters=None, threshold=0.99):
    quantile_value = float(np.quantile(hough_image.squeeze(), threshold))
    houghmap = hough_image >= quantile_value
    
    clustermap, num_ids = ndimage.label(houghmap, structure=np.ones((3, 3)))
    clusters = [np.where(clustermap==v) for v in range(num_ids)]
    clusters.sort(key=lambda c: hough_image[c].mean(), reverse=True)
    clusters = clusters[:-1]
    clusters = clusters[:max_clusters]

    centroids = [_get_centroids(hough_image, cluster) for cluster in clusters]
    scores = [hough_image[cluster].mean() for cluster in clusters]
    
    lines = []
    for centroid, score in zip(centroids, scores):
        x, y = centroid
        lines.append(((0, x, width, x + y), score))
    return lines

def _get_centroids(image, cluster):
    points = np.array(list(zip(*cluster)))
    centroids = np.average(points, weights=image[cluster] ** 4, axis=0)
    return centroids

def get_lines(image):
    shape = image.shape
    lines = []
    for rotate_times in range(4):
        rotated_image = np.rot90(image, k=rotate_times)
        hough_image = fast_hough_transform(rotated_image)

        new_shape = (shape[rotate_times % 2], shape[(rotate_times + 1) % 2])
        lines_tmp = get_horiz_line(hough_image, new_shape[1])

        [lines.append(rotate_scored_line90(line, new_shape, rotate_times)) for line in lines_tmp]
 
    lines.sort(key=lambda line: line[1], reverse=True)
    lines = [line[0] for line in lines] #remove scores
    return lines

def draw_lines(image, lines):
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.line(line, fill='red')
    return image

def rotate_scored_line90(line, image_shape, times=1):
    if times < 1: return line

    (x1, y1, x2, y2), score = line
    h, w = image_shape
    line = ((h - y1, x1, h - y2, x2), score)
    return rotate_scored_line90(line, (w, h), times - 1)


if __name__ == '__main__':
    img = np.array(Image.open('sudoku.jpg'))
    gray_img = get_grayscale_img(img)
    print(gray_img.shape)
    # hough_img = fast_hough_transform(gray_img)
    lines = get_lines(gray_img)
    new_img = draw_lines(img, lines)
    plt.imshow((gray_img), cmap='gray')
    plt.show()
    plt.imshow(new_img, cmap='gray')
    plt.show()  