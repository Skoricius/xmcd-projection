import meshio
import numpy as np
from cached_property import cached_property
import trimesh
from collections import Counter
from scipy.spatial import KDTree


class Mesh:
    def __init__(self, points, cells, parts=0) -> None:
        self.points = points
        self.cells = cells
        self._parts = parts

    @classmethod
    def from_file(cls, file_path, parts=0):
        msh = meshio.read(file_path)
        return cls(msh.points, msh.cells, parts=parts)

    @cached_property
    def tetra(self):
        tetra_list = [cb.data for cb in self.cells if cb.type == 'tetra']
        if isinstance(self._parts, (list, tuple)):
            return np.vstack([tetra_list[i] for i in self._parts])
        else:
            return tetra_list[self._parts]

    @cached_property
    def faces(self):
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
        return set.union(*[set(ef) for ef in self.edge_faces])

    def get_bounding_struct(self):
        """STL file consisting of the faces on the edge of the mesh
        """
        return trimesh.Trimesh(vertices=self.points, faces=self.edge_faces, process=False)

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