import numpy as np
import pandas as pd


def load_mesh_magnetisation(file_path):
    """Gets the mesh magnetisation from a csv file containing columns for points (named "Points:<0-2>") and magnetisation (named "m:<0-2>").

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: magnetisation ((n,3) array), points ((n,3), array)
    """
    data = pd.read_csv(file_path)
    magnetisation = data.loc[:, ['m:0', 'm:1', 'm:2']].to_numpy()
    if 'Points:0' in data:
        points = data.loc[:, ['Points:0', 'Points:1', 'Points:2']].to_numpy()
    else:
        points = data.loc[:, ['Coordinates:0',
                              'Coordinates:1', 'Coordinates:2']].to_numpy()

    return magnetisation, points


def load_piercing_data(file_path):
    """Loads all the data about piercing of the structure by light rays.

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: [description]
    """
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
