from skimage.color.colorconv import rgb2gray, rgba2rgb
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np


def get_xmcd_color(xmcd, vmin=-1, vmax=1):
    """Gets the color of the xmcd vector in grayscale"""
    # get the colormap
    norm = colors.Normalize(vmin=vmin * 0.95, vmax=vmax * 0.95)
    cmap = cm.binary
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    xmcd_color = cmap_fun.to_rgba(xmcd)
    background_color = cmap_fun.to_rgba(0)

    return xmcd_color, background_color


def get_struct_face_mag_color(struct, magnetisation, cmap_name='seismic'):
    face_magnetisation = np.mean(
        [magnetisation[struct.faces[:, i], :] for i in range(3)], axis=0)
    mag_colors = magnetisation_to_color(
        face_magnetisation, cmap_name=cmap_name)
    return mag_colors


def magnetisation_to_color(magnetisation, cmap_name='seismic', direction=[0, 0, 1]):
    # get the colormap
    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = getattr(cm, cmap_name)
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    direction = np.array(direction) / np.linalg.norm(direction)
    mag_comp = magnetisation.dot(direction)
    mag_color = cmap_fun.to_rgba(mag_comp)

    return np.array(mag_color)
