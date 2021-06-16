from warnings import warn
import numpy as np
from .projection import project_structure
from numba import njit
# from trimesh.ray.ray_triangle import ray_triangle_id
from trimesh import triangles as triangles_mod
from tqdm import tqdm

# tol.zero = 1e-12
# print(tol.zero)


class RayTracing():
    """Class that contains all the data required for raytracing.
    """

    def __init__(self, mesh, p, n=[0, 0, 1], x0=[0, 0, 0], tol=1e-5):
        """Initializarion

        Args:
            mesh (Mesh)
            p ((3,) array): Beam direction vector.
            n ((3,), optional): Normal to the projection plane. Defaults to [0, 0, 1].
            x0 ((3,), optional): Point on the projection plane. Defaults to [0, 0, 0].
            tol (float, optional): Tolerance for numerical errors. Defaults to 1e-5.
        """
        self.mesh = mesh
        self.p = np.array(p)
        self.n = np.array(n) / np.linalg.norm(n)
        self.x0 = np.array(x0)
        self.tol = tol

        self._struct = None
        self._piercings = None
        self._struct_projected = None

    @property
    def piercings(self):
        """List of tuples of two arrays for each face of the projected structure. 
        Second array are the indices of the intersected tetrahedra and the first are the lengths of the intersections.

        Returns:
            list of (2,) tuple
        """
        if self._piercings is None:
            self.get_piercings()
        return self._piercings

    @property
    def struct(self):
        """trimesh.Trimesh structure made from the triangles at the outside edge of the mesh
        """
        if self._struct is None:
            self._struct = self.mesh.get_bounding_struct()
        return self._struct

    @property
    def struct_projected(self):
        """trimesh.Trimesh structure made by projecting struct on the plan along the beam direction
        """
        if self._struct_projected is None:
            self._struct_projected = project_structure(
                self.struct, self.p, n=self.n, x0=self.x0)
        return self._struct_projected

    def get_piercings(self):
        """Gets the piercings of rays with direction self.p from a mesh to the screen with normal self.n.

        Returns:
            list of tuples of arrays (lengths (n,), indices(n,)): for each triangle in the projected structure, lengths of the segment of the ray going through the tetrahedron with the given index.
        """
        # get all the piercing data
        # use inscribed circle centre or each triangle as the ray origin
        triangles = self.struct_projected.triangles
        a = np.linalg.norm(triangles[:, 1, :] -
                           triangles[:, 2, :], axis=1)[:, None]
        b = np.linalg.norm(triangles[:, 2, :] -
                           triangles[:, 0, :], axis=1)[:, None]
        c = np.linalg.norm(triangles[:, 1, :] -
                           triangles[:, 0, :], axis=1)[:, None]
        ray_origins = (a * triangles[:, 0, :] + b * triangles[:,
                                                              1, :] + c * triangles[:, 2, :]) / (a + b + c)
        # ray_origins = self.struct_projected.triangles_center
        # get the ray piercings
        self._piercings = get_points_piercings(
            ray_origins, self.p, self.mesh.triangles, tol=self.tol)

    def get_xmcd(self, magnetisation):
        """Gets the xmcd data based on the per-vertex magnetization of the mesh.

        Args:
            magnetisation ((n,3) array): magnetization

        Returns:
            (m,) array: XMCD value for each face of the projected structure.
        """
        # get the mesh data
        tetra_magnetisation = self.get_tetra_magnetisation(
            self.mesh.tetra, magnetisation)
        # integrate over the intersected tetrahedra
        xmcd = np.array([np.sum(tetra_magnetisation[nums, :].dot(self.p) * dist)
                         for dist, nums in self.piercings])
        return xmcd

    @staticmethod
    def get_tetra_magnetisation(tetra, magnetisation):
        """Gets the magnetization of tetrahedra from per-vertex magnetisation
        """
        return np.mean(
            [magnetisation[tetra[:, i], :] for i in range(4)], axis=0)


# TODO: this could be sped up using pyembree. See: https://trimsh.org/trimesh.ray.ray_pyembree.html
# I was not able to install it on windows and this is fast enough for my structures.
def get_points_piercings(ray_origins, p, triangles, tol=1e-3):
    """Gets the ray piercings of triangles for each of the ray_origins along the vector p. 
    Returns the piercings as a list of tuples of two arrays for each of the ray origins. 
    Second array are the indices of the intersected tetrahedra and the first are the lengths of the intersections.


    Args:
        ray_origins ((n,3) array): Origins of the rays
        p ((3,) array): Ray vector
        triangles ((n,3,3) array): Triangles

    Returns:
        list: piercings list
    """
    triangles_normal = triangles_mod.normals(triangles)[0]
    tree = triangles_mod.bounds_tree(triangles)

    pnew = p[np.newaxis, :]

    def ray_piercing_fun(orig): return ray_triangle_id(
        triangles, orig[np.newaxis, :], pnew, triangles_normal=triangles_normal, tree=tree)

    def get_piercings_item(orig):
        tri_id, _, locs = ray_piercing_fun(orig)
        try:
            return get_piercings_frompt_lengths(locs, tri_id // 4)
        except ValueError as e:
            warn(
                str(e) + ' If this happens rarely, it could be a numerical artefact.')
            # running this again with tiny random movement, if it crashes a second time, it's not an artefact!
            move_dir = np.random.rand(3)
            move_dir /= np.linalg.norm(move_dir)
            tri_id, _, locs = ray_piercing_fun(orig + tol * move_dir)
            return get_piercings_frompt_lengths(locs, tri_id // 4)

    # ray_ids_generator = (ray_id_fun(orig) for orig in ray_origins)
    piercings_list = [get_piercings_item(orig) for orig in tqdm(ray_origins)]
    return piercings_list


@njit(fastmath=True)
def get_piercings_frompt_lengths(locations, intersected_tetrahedra_indx):
    """Gets the lengths and the unique index of intersected tetrahedra.
    I.e. from the locations of intersections and the indices of intersected tetrahedra, get the tetrahedra that were pierced twice and the length of the intersection segment.
    """
    intersected_tetrahedra_indx_unique = np.unique(intersected_tetrahedra_indx)
    intersected_tetrahedra_lengths = np.zeros(
        intersected_tetrahedra_indx_unique.size)
    for i, idx in enumerate(intersected_tetrahedra_indx_unique):
        pts = locations[intersected_tetrahedra_indx == idx]
        if pts.shape[0] != 2:
            # this could be improved. Instead of giving an error, it should deal with the numerical artefacts
            # continue
            raise ValueError(
                'Wrong number of intersections! Ensure that tetrahedra are valid and that the projection plane does not intersect the structure.')
        intersected_tetrahedra_lengths[i] = np.linalg.norm(
            pts[0, :] - pts[1, :])
    return (intersected_tetrahedra_lengths, intersected_tetrahedra_indx_unique)


# THIS IS COPIED FROM TRIMESH LIBRARY, BUT I NEEDED TO EDIT THE LAST BIT WHERE THEY FILTER OUT FOR THE SIDE OF THE PLANE.
# WE DO NOT CARE ABOUT THAT AND IT CAN INTRODUCE ERRORS.
# ALSO, I ADDED THE CHECK FOR TRIANGLES NORMALS
# ====================

from trimesh import intersections
from trimesh import triangles as triangles_mod
from trimesh.constants import tol
from trimesh.ray.ray_triangle import ray_triangle_candidates, ray_bounds


def ray_triangle_id(
        triangles,
        ray_origins,
        ray_directions,
        triangles_normal=None,
        tree=None):
    """
    Find the intersections between a group of triangles and rays
    Parameters
    -------------
    triangles : (n, 3, 3) float
      Triangles in space
    ray_origins : (m, 3) float
      Ray origin points
    ray_directions : (m, 3) float
      Ray direction vectors
    triangles_normal : (n, 3) float
      Normal vector of triangles, optional
    tree : rtree.Index
      Rtree object holding triangle bounds
    Returns
    -----------
    index_triangle : (h,) int
      Index of triangles hit
    index_ray : (h,) int
      Index of ray that hit triangle
    locations : (h, 3) float
      Position of intersection in space
    """
    triangles = np.asanyarray(triangles, dtype=np.float64)
    ray_origins = np.asanyarray(ray_origins, dtype=np.float64)
    ray_directions = np.asanyarray(ray_directions, dtype=np.float64)

    # if we didn't get passed an r-tree for the bounds of each
    # triangle create one here
    if tree is None:
        tree = triangles_mod.bounds_tree(triangles)

    # find the list of likely triangles and which ray they
    # correspond with, via rtree queries
    ray_candidates, ray_id = ray_triangle_candidates(
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        tree=tree)

    # get subsets which are corresponding rays and triangles
    # (c,3,3) triangle candidates
    triangle_candidates = triangles[ray_candidates]
    # (c,3) origins and vectors for the rays
    line_origins = ray_origins[ray_id]
    line_directions = ray_directions[ray_id]

    # get the plane origins and normals from the triangle candidates
    plane_origins = triangle_candidates[:, 0, :]
    if triangles_normal is None:
        plane_normals, triangle_ok = triangles_mod.normals(
            triangle_candidates)
        if not triangle_ok.all():
            raise ValueError('Invalid triangles!')
    else:
        plane_normals = triangles_normal[ray_candidates]

    # find the intersection location of the rays with the planes
    location, valid = intersections.planes_lines(
        plane_origins=plane_origins,
        plane_normals=plane_normals,
        line_origins=line_origins,
        line_directions=line_directions)

    if (len(triangle_candidates) == 0 or
            not valid.any()):
        # we got no hits so return early with empty array
        return (np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.float64))

    # find the barycentric coordinates of each plane intersection on the
    # triangle candidates
    barycentric = triangles_mod.points_to_barycentric(
        triangle_candidates[valid], location)

    # the plane intersection is inside the triangle if all barycentric
    # coordinates are between 0.0 and 1.0
    hit = np.logical_and(
        (barycentric > -tol.zero).all(axis=1),
        (barycentric < (1 + tol.zero)).all(axis=1))

    # the result index of the triangle is a candidate with a valid
    # plane intersection and a triangle which contains the plane
    # intersection point
    index_tri = ray_candidates[valid][hit]
    # the ray index is a subset with a valid plane intersection and
    # contained by a triangle
    index_ray = ray_id[valid][hit]
    # locations are already valid plane intersections, just mask by hits
    location = location[hit]

    return index_tri, index_ray, location
