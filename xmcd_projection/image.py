
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian
import numpy as np


def get_blurred_image(img, sigma=4):
    if img.shape[1] == 4:
        img = rgb2gray(rgba2rgb(img))
    elif img.shape[1] == 3:
        img = rgb2gray(img)
    return gaussian(img, sigma=sigma)


def img2uint(img):
    return (img * 255).astype(np.uint8)
