from skimage.color.colorconv import rgb2gray, rgba2rgb
import matplotlib.cm as cm
from matplotlib import colors
import numpy as np

# TODO: check the caption


def get_xmcd_color(xmcd, vmin=None, vmax=None):
    """Gets the color of the xmcd vector in grayscale. 
    It scales the colours linearly to the range between vmin and vmax.
    If vmin(vmax) is None, use min (max) of xmcd.

    Args:
        xmcd ((n,) array): xmcd values to be normalized
        vmin (int, optional): Min of normalization. Defaults to None.
        vmax (int, optional): Max of normalization. Defaults to None.

    Returns:
        tule: xmcd_color ((n, 4) array); background_color ((4,) array)
    """
    if vmin is None:
        vmin = xmcd.min()
    if vmax is None:
        vmax = xmcd.max()
    # get the colormap
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.binary
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    xmcd_color = cmap_fun.to_rgba(xmcd)
    background_color = cmap_fun.to_rgba(0)

    return xmcd_color, background_color


def get_struct_face_mag_color(struct, magnetisation, cmap_name='seismic'):
    """Gets the colours of the structure faces based on the magnetisation using the given colormap.

    Args:
        struct (trimesh.Trimesh): STL file with n faces, m vertices
        magnetisation ((m,3) array): per vertex magnetisation
        cmap_name (str, optional): Matplotlib colormap specifier. Defaults to 'seismic'.

    Returns:
        (n,4) array: Colours of struct faces
    """
    face_magnetisation = np.mean(
        [magnetisation[struct.faces[:, i], :] for i in range(3)], axis=0)
    mag_colors = magnetisation_to_color(
        face_magnetisation, cmap_name=cmap_name)
    return mag_colors


def magnetisation_to_color(magnetisation, cmap_name='seismic', direction=[0, 0, 1]):
    """Assigns color to magnetisation component.

    Args:
        magnetisation ((n,3) array): Magnetisation values
        cmap_name (str, optional): Matplotlib colormap. Defaults to 'seismic'.
        direction (list, optional): Direction along which to project the magnetisation. Defaults to [0, 0, 1].

    Returns:
        (n,4) array: Colours corresponding to values of magnetisation along the projection direction.
    """
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
