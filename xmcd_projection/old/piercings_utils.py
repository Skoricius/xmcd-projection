import numpy as np
from functools import partial
from numba import jit, njit, prange


# @jit(parallel=True, fastmath=True)
def get_points_piercings(points, p, triangles):
    piercings_list = list()
    func = partial(get_pt_ray_piercings,
                   p=p,
                   triangles=triangles)
    n_points = points.shape[0]

    piercings_list = [func(points[i, :]) for i in prange(n_points)]
    return piercings_list


@njit(fastmath=True, parallel=True)
def fast_norm_tri(tri):
    """Numba fast norm for triangles. np.linalg.norm is not fast enough!

    Args:
        tri ((n,3,3) array): Triangles.

    Returns:
        (n,3) array: Norm along third dimension
    """
    out = np.zeros((tri.shape[0], tri.shape[1]))
    for i in prange(tri.shape[0]):
        for j in prange(tri.shape[1]):
            for k in range(tri.shape[2]):
                out[i, j] += tri[i, j, k]**2
    return np.sqrt(out)


@njit(fastmath=True, parallel=True)
def fast_norm_two(x):
    """Numba fast norm for a list of vectors. np.linalg.norm is not fast enough!

    Args:
        x ((n,m) array): array of vectors.

    Returns:
        (n,) array: Norm along second dimension
    """
    out = np.zeros(x.shape[0])
    for i in prange(x.shape[0]):
        for j in range(x.shape[1]):
            out[i] += x[i, j]**2
    return np.sqrt(out)


@jit(fastmath=True)
def triangle_area(pt1, pt2, pt3):
    """Area of a triangle defined by three points. This is for Nx3 points where there are N triangles"""
    return fast_norm_two(np.cross(pt1 - pt3, pt2 - pt3))


@jit(fastmath=True)
def origin_triangle_area(pt1, pt2):
    """Area of a triangle defined by two points and origin"""
    return fast_norm_two(np.cross(pt1, pt2))


@jit(fastmath=True)
def origin_in_triangle(triangles):
    """Triangles defined as Nx3x3 array where the first axis is each triange, second axis the points in the triangle and the third axis their coordinates"""
    area_sums = origin_triangle_area(triangles[:, 0, :], triangles[:, 1, :]) + origin_triangle_area(
        triangles[:, 0, :], triangles[:, 2, :]) + origin_triangle_area(triangles[:, 1, :], triangles[:, 2, :])
    area_triangles = triangle_area(
        triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :])
    return np.isclose(area_sums, area_triangles)


@jit
def triangle_ray_intersection(p, x0, triangle):
    n = np.cross(triangle[:, 1, :] - triangle[:, 0, :],
                 triangle[:, 2, :] - triangle[:, 0, :])
    A = np.einsum('ij,ij->i', n, triangle[:, 0, :])
    c = np.cross(p, x0)
    return (np.outer(A, p) - np.cross(n, c)) / n.dot(p)[:, np.newaxis]


@jit(fastmath=True)
def get_potential_triangles_indx(pt, p, triangles, min_dist=10):
    dist = fast_norm_tri(np.cross(
        p[np.newaxis, np.newaxis, :], triangles - pt[np.newaxis, np.newaxis, :]))
    possible_triangles_indx = np.min(dist, axis=1) < min_dist
    return possible_triangles_indx

# @jit(parallel=True, fastmath=True)


def get_pt_ray_piercings(pt, p, triangles):
    """Gets the indices of tetrahedra pierced by the ray going through the point with unit direction p.
    Returns an array where the first row are the distances and the second tetrahedra they correspond to."""

    # first get only the triangles that are within a certain minimum distance from the ray. This is important for speedup
    possible_triangles_indx = get_potential_triangles_indx(pt, p, triangles)
    possible_triangles = triangles[possible_triangles_indx, :, :]
    # get the intersections of the ray with the triangle planes
    points_on_planes = triangle_ray_intersection(p, pt, possible_triangles)

    # check if the intersection is within the triangle
    pierced_triangles_indx = possible_triangles_indx.copy()
    possible_triangles_pierced_indx = origin_in_triangle(
        possible_triangles - points_on_planes[:, np.newaxis, :])
    pierced_triangles_indx[possible_triangles_indx] = possible_triangles_pierced_indx
    # reshape to be grouped per tetrahedra
    pierced_triangles_indx_resh = pierced_triangles_indx.reshape(-1, 4)
    n_piercings_per_tetra = pierced_triangles_indx_resh.sum(axis=1)
    # if just going through a point or touching an edge, will get 1
    pierced_triangles_indx_resh[n_piercings_per_tetra == 1, :] = False
    n_piercings_per_tetra[n_piercings_per_tetra == 1] = 0
    pierced_tetra_indx = n_piercings_per_tetra.astype(bool)
    # print('sec1: ', time.time() - t0)
    # t0 = time.time()
    if not np.all(n_piercings_per_tetra[pierced_tetra_indx] == 2):
        # One of the rays pierces the vertex! Chances of this happening should be infenitesimal, but can happen at large number of triangles because the intersection method can allow some edge states
        # TODO: deal with this better
        # print('Error in piercing')
        # for now, easiest way to deal with this is to add a small displacement to the point and try again
        pt += 1e-5 * np.random.rand()
        return get_pt_ray_piercings(pt, p, triangles)
    pierced_triangles_indx = pierced_triangles_indx_resh.flatten()
    # get the lenght of the pierce for each tetrahedron
    pierced_points = points_on_planes[pierced_triangles_indx[possible_triangles_indx], :]
    distances = fast_norm_two(pierced_points[::2] - pierced_points[1::2])
    # print('sec2: ', time.time() - t0)

    return distances, np.nonzero(pierced_tetra_indx)[0]
