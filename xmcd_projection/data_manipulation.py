import matplotlib.cm as cm
from matplotlib import colors
import numpy as np
from collections import Counter


def project_points(pts, phi=90, theta=15, n=np.array([0, 0, 1])):
    """Projects the point along a vector with polar angles phi and theta on the plane with normal n"""
    p = get_projection_vector(phi, theta)
    return project_points_by_vector(pts, p, n=n)


def project_points_by_vector(pts, p, n=np.array([0, 0, 1])):
    # projection matrix
    P = np.eye(3) - np.outer(p, n) / np.dot(p, n)
    return np.dot(pts, P.T)


def get_projection_vector(phi, theta):
    """Gets the projection vector based on the phi angle in xy plane and theta inclination angle to the xy plane"""
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)

    # xray direction
    p = np.array([np.cos(phi_r) * np.cos(theta_r), np.sin(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    return p


def ground_structure(struct):
    struct.vertices[:, 2] -= struct.vertices[:, 2].min()
    return struct


def ground_mesh(mesh):
    mesh.points[:, 2] -= mesh.points[:, 2].min()
    return mesh


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


def get_struct_face_mag_color(struct, magnetisation):
    face_magnetisation = np.mean(
        [magnetisation[struct.faces[:, i], :] for i in range(3)], axis=0)
    mag_colors = magnetisation_to_color(face_magnetisation)
    return mag_colors


def magnetisation_to_color(magnetisation, cmap_name='gray'):
    # get the colormap
    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = getattr(cm, 'gray')
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    mag_color = cmap_fun.to_rgba(magnetisation[:, 2])

    return np.array(mag_color)


def project_structure(struct, phi=90, theta=15, n=np.array([0, 0, 1])):
    """Projects the structure along a vector with polar angles phi and theta to the plane with normal n. Returns a trimesh structure."""
    struct_projected = struct.copy()
    struct_projected.vertices = project_points(
        struct_projected.vertices, phi=phi, theta=theta, n=n)
    return struct_projected


def get_faces(tetra):
    # get all the faces
    faces = np.array([np.delete(tr, i) for tr in tetra for i in range(4)])
    faces_sets = [frozenset(fc) for fc in faces]
    return faces_sets


def get_edge_faces(tetra):
    # edge faces are faces that appear in only one tetrahedron
    faces_sets = get_faces(tetra)
    edge_faces = []
    for item, count in Counter(faces_sets).most_common():
        if count == 1:
            edge_faces.append(item)
    return edge_faces


def get_edge_points(edge_faces):
    return set.union(*[set(ef) for ef in edge_faces])
