from warnings import warn
import numpy as np
from .projection import project_structure
from numba import njit
from trimesh.ray.ray_triangle import ray_triangle_id
from trimesh import triangles as triangles_mod
from tqdm import tqdm


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


# TODO: this could be sped up using pyembree. See: https://trimsh.org/trimesh.ray.ray_pyembree.html
# I was not able to install it on windows and this is fast enough for my structures.
def get_points_piercings(ray_origins, p, triangles):
    triangles_normal = triangles_mod.normals(triangles)[0]
    tree = triangles_mod.bounds_tree(triangles)

    pnew = p[np.newaxis, :]
    tol = 1e-3

    def ray_piercing_fun(orig): return ray_triangle_id(
        triangles, orig[np.newaxis, :], pnew, triangles_normal=triangles_normal, tree=tree)

    def get_piercings_item(orig):
        ray_ids, _, locs = ray_piercing_fun(orig)
        try:
            return get_piercings_frompt_lengths(locs, ray_ids // 4)
        except ValueError as e:
            warn(
                str(e) + ' If this happens rarely, it could be a numerical artefact.')
            # running this again, if it crashes a second time, it's not an artefact!
            ray_ids, _, locs = ray_piercing_fun(orig + tol)
            return get_piercings_frompt_lengths(locs, ray_ids // 4)

    # ray_ids_generator = (ray_id_fun(orig) for orig in ray_origins)
    piercings_list = [get_piercings_item(orig) for orig in tqdm(ray_origins)]
    return piercings_list


@njit(fastmath=True)
def get_piercings_frompt_lengths(locations, intersected_tetrahedra_indx):
    intersected_tetrahedra_indx_unique = np.unique(intersected_tetrahedra_indx)
    intersected_tetrahedra_lengths = np.zeros(
        intersected_tetrahedra_indx_unique.size)
    for i, idx in enumerate(intersected_tetrahedra_indx_unique):
        pts = locations[intersected_tetrahedra_indx == idx]
        if pts.shape[0] != 2:
            raise ValueError(
                'Wrong number of intersections! Ensure that tetrahedra are valid and that the projection plane does not intersect the structure.')
        intersected_tetrahedra_lengths[i] = np.linalg.norm(
            pts[0, :] - pts[1, :])
    return (intersected_tetrahedra_lengths, intersected_tetrahedra_indx_unique)
