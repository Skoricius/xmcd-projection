# XMCD Projection

[![Documentation Status](https://readthedocs.org/projects/xmcd-projection/badge/?version=latest)](https://xmcd-projection.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/374098368.svg)](https://zenodo.org/badge/latestdoi/374098368)

Projection simulation of the PEEM structures

## Installation

The project is available on PyPi: 
```
pip install xmcd-projection
```

In order to run the example notebooks, jupyter also needs to be installed: 
```
pip install jupyter
```

## Usage
See the [docs](https://xmcd-projection.readthedocs.io/en/latest/?badge=latest) for details and examples: 

If running in Jupyter, make sure you use qt gui for the visualizer to work:
```python
%gui qt
%matplotlib qt
```


### From GMSH .msh file and magnetisation in .csv exported from Magnumfe data


To do this, have .msh file exported by GMSH and the magnetisation data by importing the data in Paraview and going to Save Data... and exporting to .csv. See `examples/example_script.py`:
```python 
    from xmcd_projection import *

    msh_file = "example_mesh.msh"
    mag_file = "mag_data.csv"

    # get the mesh
    msh = Mesh.from_file(msh_file)
    # get the projection vector
    p = get_projection_vector(90, 16)

    # prepare raytracing object
    raytr = RayTracing(msh, p)
    raytr.get_piercings()
    struct = raytr.struct
    struct_projected = raytr.struct_projected

    # load magnetization and make sure the indices are not shuffled
    magnetisation, mag_points = load_mesh_magnetisation(mag_file)
    shuffle_indx = msh.get_shuffle_indx(mag_points)
    magnetisation = magnetisation[shuffle_indx, :]

    # get the colours and xmcd values
    xmcd_value = raytr.get_xmcd(magnetisation)
    mag_colors = get_struct_face_mag_color(struct, magnetisation)

    # define the visualizer parameters and show
    azi = 90
    center_struct = [0, 0, 0]
    dist_struct = 2e4
    center_peem = [0, -1000, 0]
    dist_peem = 2e5

    vis = MeshVisualizer(struct, struct_projected,
                        projected_xmcd=xmcd_value, struct_colors=mag_colors)
    vis.show(azi=azi, center=center_peem, dist=dist_peem)
    vis.start()
```

### From .vtk generated from Mumax data
Import .vtk file in Paraview, go to Saving Data... and export to .vtu format for the mesh, and .csv format for the magnetization. Ensure that you are only exporting the part with nonzero magnetization. This can be achieved by setting a threshold in Paraview. Then, it is very similar to Magnum.fe, only care needs to be taken to set the appropriate scale. See `examples/mumax_example_script.py`:
```python
    from xmcd_projection import *

    msh_file = "mumax_mesh.vtu"
    mag_file = "mumax_mag.csv"
    # need to scale points to get them in nm
    scale = 1e9

    # get the mesh
    msh = Mesh.from_file(msh_file, scale=scale)
    print(msh.cells)
    # get the projection vector
    p = get_projection_vector(90, 16)

    # prepare raytracing object
    raytr = RayTracing(msh, p)
    raytr.get_piercings()
    struct = raytr.struct
    struct_projected = raytr.struct_projected

    # load magnetization and make sure the indices are not shuffled
    magnetisation, mag_points = load_mesh_magnetisation(mag_file, scale=scale)
    shuffle_indx = msh.get_shuffle_indx(mag_points)
    magnetisation = magnetisation[shuffle_indx, :]

    # get the colours and xmcd values
    xmcd_value = raytr.get_xmcd(magnetisation)
    mag_colors = get_struct_face_mag_color(struct, magnetisation)

    # define the visualizer parameters and show
    azi = 90
    center_struct = [0, 0, 0]
    dist_struct = 1e4
    center_peem = [100, -200, 0]
    dist_peem = 8e4

    vis = MeshVisualizer(struct, struct_projected,
                        projected_xmcd=xmcd_value, struct_colors=mag_colors)
    vis.show(azi=azi, center=center_peem, dist=dist_peem)
    vis.start()
```

### From STL file and magnetisation as numpy array
This is older version of the library, so might not work as well. Here just in case it is needed in the future
```python
%gui qt
%matplotlib qt
from  xmcd_projection import *
from xmcd_projection.stl_visualisation import *
import trimesh
structure_file = r"testing\SOL6-1-3.stl"
magnetisation_file = r"testing\SOL6-1-3_uniform_mag.p"

struct = trimesh.load(structure_file)
magnetisation = np.zeros(struct.vertices.shape)
magnetisation[:, 1] = 1

# define projection vector
p = get_projection_vector(90, 15)

# create visualisation
v = Visualizer(struct, magnetisation, p)
v.generate_view()
v.show()
v.save_render('test1.png')
```
