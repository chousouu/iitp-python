import numpy as np
from pathlib import Path
from PIL.Image import Image as PIL_Image
from PIL import Image, ImageDraw
import cv2 as cv
from scipy import ndimage
import matplotlib.pyplot as plt

class HoughTransform():
    def __init__(self, image):
        image = handle_img_type(image)
        self.image = handle_img_type(image)
        self.grayscale = self._get_grayscale_img()
        self.shape = self.grayscale.shape
        self.lines = None
        self.hough_image = None

    def fast_hough_transform(self, image=None):
        if image is None:
            image = self.image
        imgh, imgw = image.shape
        hough_levels = imgw.bit_length() + 1
        
        hough_h = imgh
        hough_w = 2 ** (hough_levels - int(imgw == (2**hough_levels)))
        
        hough = np.zeros((hough_h, hough_w))
        hough[:imgh, :imgw] = self.grayscale
        
        for level in range(1, hough_levels):
            block_size = 2 ** level 
            new_hough = np.zeros((hough_h, hough_w))
            for x in range(block_size):
                shift = -((x + 1) // 2)
                lpos = x // 2
                rpos = (block_size + x) // 2
                left_values = hough[:, lpos::block_size]
                right_values = np.roll(hough[:, rpos::block_size], shift, axis=0)
                new_hough[:, x::block_size] = left_values + right_values
            
            hough = new_hough
        
        result = hough[:imgh, :imgw]
        self.hough_image = result
        return MinMaxImg(result)
    
    def get_lines(self):
        lines = []
        for rotate_times in range(1, 4):
            rotated_image = np.rot90(self.grayscale, k=rotate_times)
            hough_image = self.fast_hough_transform(rotated_image)

            lines_tmp = self._get_horiz_line(hough_image)
            for line in lines_tmp: #rotate lines back to original img
                lines.append(_rotate_scored_line90(line, self.shape, rotate_times)) 
        
        lines.sort(key=lambda line: line[1], reverse=True) #sort by score
        lines = [line[0] for line in lines] #remove scores
        self.lines = lines
        
        return lines
    
    def draw_lines(self):
        image = Image.fromarray(self.image)
        draw = ImageDraw.Draw(image)
        
        if self.lines is None:
            self.lines = self.get_lines()

        for line in self.lines:
            draw.line(line, fill='red')
        return image

    def _get_grayscale_img(self):
        image_t = Image.fromarray(self.image)
        gray_img = np.array(image_t.convert(mode='L'))

        return np.abs(cv.Laplacian(gray_img, cv.CV_64F))

    def _get_horiz_line(self, hough_image, min_cluster_size=1e-4, 
                       max_clusters=None, threshold=0.99):
        width = hough_image.shape[1]
        quantile_value = float(np.quantile(hough_image.squeeze(), threshold))
        houghmap = hough_image >= quantile_value
        
        clustermap, num_ids = ndimage.label(houghmap, structure=np.ones((3, 3)))
        clusters = [np.where(clustermap==v) for v in range(num_ids)]
        clusters.sort(key=lambda c: hough_image[c].mean(), reverse=True)
        clusters = clusters[:max_clusters]
        clusters = clusters[:-1]

        if min_cluster_size < 1:
            min_cluster_size *= hough_image.size    
        clusters = [c for c in clusters if len(c[0]) >= min_cluster_size]

        centroids = [_get_centroids(hough_image, cluster) for cluster in clusters]
        scores = [hough_image[cluster].mean() for cluster in clusters]

        lines = []
        for centroid, score in zip(centroids, scores):
            x, y = centroid
            lines.append(((0, x, width, x + y), score))
        return lines

def handle_img_type(img):
    if isinstance(img, PIL_Image):
        img = np.array(img)
    elif isinstance(img, np.ndarray):
        img = img 
    else:
        img = Image.open(img)
    return np.array(img)

def MinMaxImg(img):
    return (img - img.min()) / (img.max() - img.min())


def _get_centroids(image, cluster):
    points = np.array(list(zip(*cluster)))
    centroids = np.average(points, weights=image[cluster], axis=0)
    return centroids

def _rotate_scored_line90(line, image_shape, times=1):
    if times < 1: return line

    (x1, y1, x2, y2), score = line
    h, w = image_shape
    line = ((h - y1, x1, h - y2, x2), score)
    return _rotate_scored_line90(line, (w, h), times - 1)


if __name__ == '__main__':
    img_fht = HoughTransform('sudoku.jpg')
    plt.imshow(img_fht.draw_lines(), cmap='gray')
    plt.show()  
