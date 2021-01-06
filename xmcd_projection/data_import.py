import numpy as np
import pandas as pd
import meshio
import trimesh


def get_mesh_data_from_file(file_path):
    mesh = meshio.read(file_path)
    tetra = [cb.data for cb in mesh.cells if cb.type == 'tetra'][0]
    points = mesh.points
    faces = np.array([np.delete(tr, i) for tr in tetra for i in range(4)])
    return points, faces, tetra


def get_mesh_data(mesh):
    tetra = [cb.data for cb in mesh.cells if cb.type == 'tetra'][0]
    points = mesh.points
    faces = np.array([np.delete(tr, i) for tr in tetra for i in range(4)])
    return points, faces, tetra


def get_magnumfe_magnetisation(file_path):
    data = pd.read_csv(file_path)
    magnetisation = data.loc[:, ['m:0', 'm:1', 'm:2']].to_numpy()
    points = data.loc[:, ['Points:0', 'Points:1', 'Points:2']].to_numpy()

    return magnetisation, points


def define_trimesh(points, faces):
    return trimesh.Trimesh(vertices=points, faces=faces, process=False)
