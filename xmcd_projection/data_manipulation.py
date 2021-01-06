import matplotlib.cm as cm
from matplotlib import colors
import numpy as np


def project_points(pts, phi=0, theta=15, n=np.array([0, 0, 1])):
    """Projects the point along a vector with polar angles phi and theta on the plane with normal n"""
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)

    # xray direction
    p = np.array([np.sin(phi_r) * np.cos(theta_r), np.cos(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    # projection matrix
    P = np.eye(3) - np.outer(p, n) / np.dot(p, n)

    return np.dot(pts, P.T)


def calculate_xmcd(magnetisation, phi=0, theta=15):
    """Calculates the xmcd signal with light coming along a unit vector with polar angles phi and theta"""
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)
    # magnetisation should be normalised, but just make sure
    magnetisation /= np.linalg.norm(magnetisation, axis=1)[:, np.newaxis]

    # xray direction
    p = np.array([np.sin(phi_r) * np.cos(theta_r), np.cos(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    return np.dot(magnetisation, p)


def get_xmcd_color(xmcd):
    """Gets the color of the xmcd vector in grayscale"""
    # get the colormap
    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = cm.gray
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    xmcd_color = cmap_fun.to_rgba(xmcd)

    return xmcd_color


def project_structure(struct, phi=0, theta=15, n=np.array([0, 0, 1])):
    """Projects the structure along a vector with polar angles phi and theta to the plane with normal n. Returns a trimesh structure."""
    struct_projected = struct.copy()
    struct_projected.vertices = project_points(
        struct_projected.vertices, phi=phi, theta=theta, n=n)
    return struct_projected
