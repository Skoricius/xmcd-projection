# XMCD Projection

Projection simulation of the PEEM structures

## Usage
If running in Jupyter, make sure you use qt gui for the visualizer to work:
```python
%gui qt
%matplotlib qt
```



### From GMSH .msh file and magnetisation in .csv exported from Magnumfe data
To do this, have .msh file exported by GMSH and the magnetisation data by importing the data in Paraview and going to Save Data... and exporting to .csv
```python 
%gui qt
%matplotlib qt
from xmcd_projection import *
import numpy as np
msh_file = r"testing\spiral_det_in_50w30t.msh"
magnetisation_file = r"testing\data.30.csv"

# get the mesh and magnetisation
msh = Mesh.from_file(msh_file)
magnetisation, mag_points = load_mesh_magnetisation(magnetisation_file)
# define the projection direction
p = get_projection_vector(90, 16)
# calculate raytracings. Note: this might take a while, but can be saved for multiple uses on the same mesh file
raytr = RayTracing(msh, p)
raytr.get_piercings()
np.save("raytracing.npy", raytr, allow_pickle=True)

# get the xmcd and magnetisation colors
xmcd_value = raytr.get_xmcd(magnetisation)
mag_colors = get_struct_face_mag_color(raytr.struct, magnetisation)

# visualize the result
vis = MeshVisualizer(raytr.struct, raytr.struct_projected, projected_xmcd=xmcd_value, struct_colors=mag_colors)
vis.show()
```

### From STL file and magnetisation as numpy array
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



## TO DO
* make sure you are only using struct.points
* update README (magnumfe projection part)