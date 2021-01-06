# XMCD Projection

Projection simulation of the PEEM structures

## Usage
If running in Jupyter, make sure you use qt gui:
```python
%gui qt
%matplotlib qt
```

### From STL file and magnetisation as numpy array
```python
from xmcd_projection import *
import trimesh
structure_file = 'SOL6-1-3.stl'
magnetisation_file = 'SOL6-1-3_uniform_mag.p'

struct = trimesh.load(structure_file)
magnetisation = np.zeros(struct.vertices.shape)
magnetisation[:, 1] = 1

v = Visualizer(struct, magnetisation)
v.generate_view()
v.save_render('test1.png')
```


### From GMSH .msh file and magnetisation in .csv exported from Magnumfe data
To do this, have .msh file exported by GMSH and the magnetisation data by importing the data in Paraview and going to Save Data... and exporting to .csv
```python 
from xmcd_projection import *
structure_file = r"C:\Users\lukas\OneDrive - University of Cambridge\PhD\Simulations\spirals\models\spiral_det_in_50w30t.msh"
magnetisation_file = r"F:\PhD_data\Simulations\spiral_automotion_simulations\data\basic_automotion\spiral_det_in_50w30t_005\data\data.30.csv"

# import the structure and magnetisation data
points, faces, tetra = get_mesh_data_from_file(structure_file)
magnetisation, points = get_magnumfe_magnetisation(magnetisation_file)
# define the trimesh from structure data
struct = define_trimesh(points, faces)

# show the visualiser (look at xmcd_visualisation file for details)
vis = Visualizer(struct, magnetisation)

# define the viewing angles (here all default)
vis.set_camera(azi=30, dist=1e5, center=[0, -700, 0])
# show the rendering
vis.show()
```