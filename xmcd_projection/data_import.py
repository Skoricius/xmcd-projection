import numpy as np
import pandas as pd
import meshio
import trimesh
import pickle


def get_mesh(file_path):
    return meshio.read(file_path)


def get_mesh_data_from_file(file_path):
    mesh = get_mesh(file_path)
    return get_mesh_data(mesh)


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


def get_struct_from_mesh(mesh):
    points, faces, _ = get_mesh_data(mesh)
    return define_trimesh(points, faces)


def load_piercing_data(file_path):
    data_loaded = np.load(file_path, allow_pickle=True).item()
    struct = data_loaded['struct']
    struct_projected = data_loaded['struct_projected']
    piercings_list = data_loaded['piercings_list']
    p = data_loaded['p']
    mesh = data_loaded['mesh']
    return struct, struct_projected, piercings_list, p, mesh


def save_piercing_data(file_path, struct, struct_projected, piercings_list, p, mesh):
    data_to_save = {
        'struct': struct,
        'struct_projected': struct_projected,
        'piercings_list': piercings_list,
        'p': p,
        'mesh': mesh
    }
    np.save(file_path, data_to_save, allow_pickle=True)
