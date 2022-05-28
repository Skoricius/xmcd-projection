import os
from collections import Counter

import meshio
import numpy as np
import trimesh
from cached_property import cached_property
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


class Mesh:
    """Object for handling the GMSH .msh files information
    """

    def __init__(self, points, cells, parts=0):
        """Initializes the onbject

        Args:
            points ((n,3) array): points of the mesh
            cells (list of cell blocks): Cells (see meshio library)
            parts (int, optional): Which parts of the structure to take into account. Defaults to 0.
        """
        self.points = points
        self.cells = cells
        self._parts = parts

    @classmethod
    def from_file(cls, file_path, parts=0, scale=1):
        """Create the object from file_path. Supports .msh files and .vtu files as exported by mumax.

        Args:
            file_path (str)
            parts (int, optional): Which parts to use. Defaults to 0.
            scale (float, optional): Scale for the coordinates. Defaults to 1.

        Returns:
            Mesh
        """

        ext = os.path.splitext(file_path)[-1]
        if ext == ".msh":
            msh = meshio.read(file_path)
            points = msh.points * scale
            cells = msh.cells
        elif ext == ".vtu":
            msh0 = meshio.read(file_path)
            # construct tetrahedra from cubes
            n_cubes = msh0.cells[0].data.shape[0]
            cube_to_tetra_idx = np.array([
                [0, 3, 1, 4],
                [2, 3, 1, 6],
                [5, 6, 4, 1],
                [7, 4, 6, 3],
                [4, 6, 3, 1]])
            tetra = np.vstack([msh0.cells[0].data[i, :][cube_to_tetra_idx[j, :]]
                               for i in range(n_cubes) for j in range(5)])
            points = msh0.points * scale
            cells = [meshio.CellBlock(type="tetra", data=tetra), ]
        else:
            raise ValueError("File format not recognized!")
        return cls(points, cells, parts=parts)

    @cached_property
    def tetra(self):
        """Tetrahedra of the mesh.

        Returns:
            (n,4) array of point indices.
        """
        tetra_list = [cb.data for cb in self.cells if cb.type == 'tetra']
        if isinstance(self._parts, (list, tuple)):
            return np.vstack([tetra_list[i] for i in self._parts])
        else:
            return tetra_list[self._parts]

    @cached_property
    def faces(self):
        """Faces of the mesh

        Returns:
            (n,3) array
        """
        faces = Mesh.get_faces_from_tetra(self.tetra)
        return faces

    @cached_property
    def triangles(self):
        """Triangles representing tetrahedra faces.

        Returns:
            (n,3,3) array: Points of triangles.
        """
        return np.moveaxis(np.stack([self.points[self.faces[:, i], :] for i in range(3)]), 0, 1)

    @staticmethod
    def get_faces_from_tetra(tetra):
        selector = np.arange(4)
        # get the faces and reshuffle so that the same tetra faces are in groups of 4
        faces = np.array([tetra[:, np.delete(selector, i)] for i in range(4)])
        faces = np.swapaxes(faces, 0, 1).reshape(-1, 3)
        return faces

    @cached_property
    def edge_faces(self):
        """Gets the faces of the tetra that are on the outside of the mesh

        Returns:
            (m,3) array: Faces on the edge of the mesh 
        """
        # edge faces are faces that appear in only one tetrahedron
        faces_sets = [frozenset(fc) for fc in self.faces]
        edge_faces = []
        for item, count in Counter(faces_sets).most_common():
            if count == 1:
                edge_faces.append(item)
        return np.array([list(it) for it in edge_faces])

    @cached_property
    def edge_points_indices(self):
        """Indices of the points on the edges of the mesh
        """
        return set.union(*[set(ef) for ef in self.edge_faces])

    def get_bounding_struct(self, fix=False):
        """STL file consisting of the faces on the edge of the mesh. If fix=True, uses trimesh to fix the normals of the faces.
        """
        struct = trimesh.Trimesh(vertices=self.points,
                                 faces=self.edge_faces, process=False)
        if fix:
            trimesh.repair.fix_winding(struct)
            trimesh.repair.fix_inversion(struct)
        return struct

    def get_shuffle_indx(self, points):
        """Finds the indices such that points[shuffle_indx, :] = self.points.

        This is only necessary when due to the file containing multiple parts, 
        these get mixed up by the magnetization export when compared to the original mesh.
        These indices can be used to reshuffle the magnetisation:
        magnetisation = magnetisation[shuffle_indx, :]

        Args:
            points ((n, 3) array): Points to be shuffled.

        Returns:
            (n,) array: Indices of the shuffled array
        """
        kd = KDTree(points)
        _, shuffle_indx = kd.query(self.points, eps=1e-2, p=1)
        return shuffle_indx

    def translate(self, v):
        """Translate the structure by v

        Parameters
        ----------
        v : ((3,) array)
            Translation vector
        """
        self.points += v

    def rotate(self, v):
        """Rotates the structure by rotation vector

        Parameters
        ----------
        v : ((3,) array)
            Rotation vector. See scipy rotvec.
        """
        r = Rotation.from_rotvec(v)
        self.points = self.points.dot(r.as_matrix().T)
