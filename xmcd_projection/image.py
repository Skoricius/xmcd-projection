
from skimage.color import rgb2gray, rgba2rgb
from skimage.filters import gaussian
import numpy as np


def get_blurred_image(img, sigma=4):
    """Applies the gaussian blur to the given image.

    Args:
        img ((n,m,4), (n,m,3) array): Colour image.
        sigma (int, optional): Gaussian sigma. Defaults to 4.

    Returns:
        (n,m) array: Blurred image
    """
    if img.shape[1] == 4:
        img = rgb2gray(rgba2rgb(img))
    elif img.shape[1] == 3:
        img = rgb2gray(img)
    return gaussian(img, sigma=sigma)


def img2uint(img):
    """Tranforms the image from float to uint8 type.

    Args:
        img (array): Image

    Returns:
        array: Uint8 image
    """
    return (img * 255).astype(np.uint8)
