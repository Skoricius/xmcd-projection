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
