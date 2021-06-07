# XMCD Projection

Projection simulation of the PEEM structures

## Usage
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
