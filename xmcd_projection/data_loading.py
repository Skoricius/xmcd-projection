import numpy as np
import pandas as pd
from pandas.core.base import DataError


def load_mesh_magnetisation(file_path, scale=1):
    """Gets the mesh magnetisation from a csv file containing columns for points (named "Points:<0-2>") and magnetisation (named "m:<0-2>").

    Args:
        file_path (str): Path to the file.

    Returns:
        tuple: magnetisation ((n,3) array), points ((n,3), array)
    """
    data = pd.read_csv(file_path)
    if 'm:0' in data:
        magnetisation = data.loc[:, ['m:0', 'm:1', 'm:2']].to_numpy()
    elif 'velocity_vectors:0' in data:
        magnetisation = data.loc[:, [
            'velocity_vectors:0', 'velocity_vectors:1', 'velocity_vectors:2']].to_numpy()
    else:
        raise DataError("Unrecognized format!")

    if 'Points:0' in data:
        points = data.loc[:, ['Points:0', 'Points:1', 'Points:2']].to_numpy()
    elif 'Coordinates:0' in data:
        points = data.loc[:, ['Coordinates:0',
                              'Coordinates:1', 'Coordinates:2']].to_numpy()
    else:
        raise DataError("Unrecognized format!")

    return magnetisation, points * scale
