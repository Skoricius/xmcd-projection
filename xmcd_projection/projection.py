import numpy as np


def project_points(pts, p, n=[0, 0, 1], x0=[0, 0, 0]):
    """Projects the points along a vector p to a surface with normal n going through point x0"""
    n = np.array(n)
    # projection matrix
    P = np.eye(3) - np.outer(p, n) / np.dot(p, n)
    return np.dot(pts, P.T) + np.dot(n, np.array(x0)) / np.dot(n, p) * p


def get_projection_vector(phi, theta):
    """Gets the projection vector based on the phi angle in xy plane and theta inclination angle to the xy plane (spherical polars)

    Args:
        phi (float): Phi angle in degrees.
        theta (float): Theta angle in degrees.

    Returns:
        (3,) array: Projection vector.
    """
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)

    # xray direction
    p = np.array([np.cos(phi_r) * np.cos(theta_r), np.sin(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    return p


def project_structure(struct, p, n=[0, 0, 1], x0=None):
    """Projects the structure along a vector p to the plane with normal n containing the point x0. If x0 is None, take the structure point closest to the plane along p. Returns a trimesh structure.

    Args:
        struct (object): Any object containing property vertices which is (n,3) array
        p ((3,) array): Projection direction.
        n ((3,) array, optional): Normal to the surface. Defaults to [0, 0, 1].
        x0 ((3,), optional): Point on the surface. Defaults to None.

    Returns:
        object: Same as struct with vertices being the projected points.
    """
    struct_projected = struct.copy()
    n = np.array(n)
    points = np.array(struct.vertices)
    if x0 is None:
        x0 = np.min(points.dot(n)) * n
    struct_projected.vertices = project_points(
        points, p, n=n, x0=x0)
    # get rid of the triangles which get compressed to lines
    tri = struct_projected.triangles
    tri_areas = np.linalg.norm(np.cross(
        tri[:, 0, :] - tri[:, 1, :], tri[:, 0, :] - tri[:, 2, :], axis=1), axis=1)
    struct_projected.faces = struct_projected.faces[tri_areas > 1e-13]
    return struct_projected
