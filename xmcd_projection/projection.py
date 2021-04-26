import numpy as np


def project_points(pts, p, n=[0, 0, 1], x0=[0, 0, 0]):
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


def project_structure(struct, p, n=[0, 0, 1]):
    """Projects the structure along a vector p to the plane with normal n containing the point x0. If x0 is None, take the point closest to the plane along p. Returns a trimesh structure."""
    struct_projected = struct.copy()
    n = np.array(n)
    points = np.array(struct.vertices)
    x0 = np.min(points.dot(n)) * n
    struct_projected.vertices = project_points(
        points, p, n=n, x0=x0)
    return struct_projected
