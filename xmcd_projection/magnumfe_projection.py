from numba.npyufunc import parallel
from xmcd_projection.data_manipulation import get_edge_faces, get_faces_from_tetra, project_points, project_points_by_vector, get_projection_vector
import numpy as np
from .data_import import define_trimesh, get_mesh_data
from functools import partial
from tqdm import tqdm
from numba import jit, njit, prange
import time

# need to define some fast norms for performance. np.linalg.norm has a large overhead


@njit(fastmath=True, parallel=True)
def fast_norm_tri(tri):
    out = np.zeros((tri.shape[0], tri.shape[1]))
    for i in prange(tri.shape[0]):
        for j in prange(tri.shape[1]):
            for k in range(tri.shape[2]):
                out[i, j] += tri[i, j, k]**2
    return np.sqrt(out)


# need to define some fast norms for performance. np.linalg.norm has a large overhead
@njit(fastmath=True, parallel=True)
def fast_norm_two(x):
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


def get_triangles_from_faces(points, faces):
    return np.moveaxis(np.stack([points[faces[:, i], :] for i in range(3)]), 0, 1)


def get_triangles_from_tetra(points, tetra):
    faces = np.array(get_faces_from_tetra(tetra))
    return np.moveaxis(np.stack([points[faces[:, i], :] for i in range(3)]), 0, 1)


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
        pt += 1e-3 * np.random.rand()
        return get_pt_ray_piercings(pt, p, triangles)
    pierced_triangles_indx = pierced_triangles_indx_resh.flatten()
    # get the lenght of the pierce for each tetrahedron
    pierced_points = points_on_planes[pierced_triangles_indx[possible_triangles_indx], :]
    distances = fast_norm_two(pierced_points[::2] - pierced_points[1::2])
    # print('sec2: ', time.time() - t0)

    return distances, np.nonzero(pierced_tetra_indx)[0]


def get_xmcd_from_piercings(p, tetra_magnetisation, distances, pierced_tetra_nums):
    # extract the magnetisation of tetrahedra tetrahedra that are pierced by the beam
    pierced_tetra_magnetisation = tetra_magnetisation[pierced_tetra_nums, :]
    # get the total xmcd contribution
    xmcd = np.sum(pierced_tetra_magnetisation.dot(p) * distances)
    return xmcd


def get_xmcd_from_piercings_list(tetra, magnetisation, piercings_list, p=None):
    if p is None:
        # get the beam direction
        p = get_projection_vector(90, 15)
    # get the mesh data
    tetra_magnetisation = get_tetra_magnetisation(tetra, magnetisation)
    xmcd = np.array([get_xmcd_from_piercings(p, tetra_magnetisation, dist, nums)
                     for dist, nums in piercings_list])
    return xmcd


@jit(parallel=True)
def get_tetra_magnetisation(tetra, magnetisation):
    return np.mean(
        [magnetisation[tetra[:, i], :] for i in range(4)], axis=0)


@jit(parallel=True, fastmath=True)
def get_all_piercings(points, p, triangles):
    piercings_list = list()
    func = partial(get_pt_ray_piercings,
                   p=p,
                   triangles=triangles)
    n_points = points.shape[0]
    # for i in range(n_points):
    #     piercings_list.append(func(points[i, :]))
    #     print('step')

    piercings_list = [func(points[i, :]) for i in prange(n_points)]
    return piercings_list


def get_projection_with_piercings(mesh, p=None):
    # get the beam direction
    if p is None:
        p = get_projection_vector(90, 15)
    else:
        p /= np.linalg.norm(p)
    # get the mesh data
    points, faces, tetra = get_mesh_data(mesh)
    # get the triangles defining the mesh
    triangles = get_triangles_from_faces(points, faces)

    # project the points by the beam.
    projected_points = project_points_by_vector(points, p)
    # get only the edge faces for projection
    edge_faces = get_edge_faces(tetra)
    edge_faces = np.array([list(ef) for ef in edge_faces])
    # find the centroid of each of the edge faces
    projected_edge_triangles = get_triangles_from_faces(
        projected_points, edge_faces)
    projected_centroids = np.mean(projected_edge_triangles, axis=1)
    # add a small shift to the centroids to make sure that there is no weird numerical errors
    projected_centroids += 1e-3 * np.random.rand(*projected_centroids.shape)
    # get all the piercing data
    piercings_list = get_all_piercings(projected_centroids, p, triangles)

    struct_projected = define_trimesh(projected_points, edge_faces)
    return struct_projected, piercings_list, p
