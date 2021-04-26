import numpy as np
from .projection import project_structure
from numba import njit
from trimesh.ray.ray_triangle import ray_triangle_id


class RayTracing():
    def __init__(self, mesh, p, n=[0, 0, 1]) -> None:
        self.mesh = mesh
        self.p = p
        self.n = n

        self._struct = None
        self._piercings = None
        self._struct_projected = None

    @property
    def piercings(self):
        if self._piercings is None:
            self.get_piercings()
        return self._piercings

    @property
    def struct(self):
        if self._struct is None:
            self._struct = self.mesh.get_bounding_struct()
        return self._struct

    @property
    def struct_projected(self):
        if self._struct_projected is None:
            self._struct_projected = project_structure(
                self.struct, self.p, n=self.n)
        return self._struct_projected

    def get_piercings(self):
        """Gets the piercings of rays with direction self.p from a mesh to the screen with normal self.n.

        Returns:
            list of tuples of arrays (lengths (n,), indices(n,)): for each triangle in the projected structure, lengths of the segment of the ray going through the tetrahedron with the given index.
        """
        # get all the piercing data
        points = self.struct_projected.triangles_center
        self._piercings = get_points_piercings(
            points, self.p, self.mesh.triangles)

    def get_xmcd(self, magnetisation):
        # get the mesh data
        tetra_magnetisation = self.get_tetra_magnetisation(
            self.mesh.tetra, magnetisation)
        xmcd = np.array([np.sum(tetra_magnetisation[nums, :].dot(self.p) * dist)
                         for dist, nums in self.piercings])
        return xmcd

    @staticmethod
    def get_tetra_magnetisation(tetra, magnetisation):
        return np.mean(
            [magnetisation[tetra[:, i], :] for i in range(4)], axis=0)


def get_points_piercings(ray_origins, p, triangles):
    n = ray_origins.shape[0]
    ray_directions = np.repeat(p[None, :], n, axis=0)
    index_triangle, index_ray, locations = ray_triangle_id(
        triangles, ray_origins, ray_directions)
    piercings_list = [
        get_piercings_frompt_lengths(
            locations[index_ray == i],
            index_triangle[index_ray == i] // 4)
        for i in range(n)]
    return piercings_list


@njit(fastmath=True)
def get_piercings_frompt_lengths(locations, intersected_tetrahedra_indx):
    intersected_tetrahedra_indx_unique = np.unique(intersected_tetrahedra_indx)
    intersected_tetrahedra_lengths = np.zeros(
        intersected_tetrahedra_indx_unique.size)
    for i, idx in enumerate(intersected_tetrahedra_indx_unique):
        pts = locations[intersected_tetrahedra_indx == idx]
        if pts.shape[0] != 2:
            raise Exception(
                'Wrong number of intersections! Ensure that tetrahedra are valid and that the projection plane does not intersect the structure.')
        intersected_tetrahedra_lengths[i] = np.linalg.norm(
            pts[0, :] - pts[1, :])
    return (intersected_tetrahedra_lengths, intersected_tetrahedra_indx_unique)
