from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
import numpy as np
import sys
import trimesh
import matplotlib.cm as cm
from matplotlib import colors
import pickle
from pyqtgraph import Vector


pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'w')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def show_structure_points(struct, step=1, ax=None, rescale=True):
    """Shows points of the given trimesh structure on matplotlib plot

        Args:
            struct (trimesh): structure to show
            step: each how many points to show (if the structure has a lot of points, it is useful to only show say every 5th to speed up the plotting)
            ax: axes on which to plot. If none new axes are created
            rescale: if the axes should be rescaled to the new structure
        Returns:
            ax
            """
    if ax is None:
        # Create a new plot
        fig = plt.figure()
        ax = fig.add_subplot((111), projection='3d', proj_type = 'ortho')
        
    pts = struct.vertices
    ax.scatter(pts[::step, 0], pts[::step, 1],pts[::step, 2])

    if rescale:
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0

        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    return ax

def show_magnetisation(struct, magnetisation, step=1, ax=None, rescale=True, length=100):
    """Shows vectors of magnetisation of the given trimesh structure on matplotlib quiver plot

        Args:
            struct (trimesh): structure to show
            magnetisation: per vertex magnetisation vector (Nx3)
            step: each how many points to show (if the structure has a lot of points, it is useful to only show say every 5th to speed up the plotting)
            ax: axes on which to plot. If none new axes are created
            rescale: if the axes should be rescaled to the new structure
        Returns:
            ax
            """
    if ax is None:
        # Create a new plot
        fig = plt.figure()
        ax = fig.add_subplot((111), projection='3d')
        
    pts = struct.vertices
    x, y, z = pts[::step, 0], pts[::step, 1], pts[::step, 2]
    u, v, w = magnetisation[::step, 0], magnetisation[::step, 1], magnetisation[::step, 2]
    
    ax.quiver(x, y, z, u, v, w, normalize=True, length=length, arrow_length_ratio=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    
    return ax


def project_points(pts, phi=0, theta=15, n=np.array([0, 0, 1])):
    """Projects the point along a vector with polar angles phi and theta on the plane with normal n"""
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)

    # xray direction
    p = np.array([np.sin(phi_r) * np.cos(theta_r), np.cos(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    # projection matrix
    P = np.eye(3) - np.outer(p, n) / np.dot(p, n)

    return np.dot(pts, P.T)


def calculate_xmcd(magnetisation, phi=0, theta=15):  
    """Calculates the xmcd signal with light coming along a unit vector with polar angles phi and theta"""
    # define the projection geometry
    theta_r = np.deg2rad(theta)
    phi_r = np.deg2rad(phi)
    # magnetisation should be normalised, but just make sure
    magnetisation /= np.linalg.norm(magnetisation, axis=1)[:, np.newaxis]

    # xray direction
    p = np.array([np.sin(phi_r) * np.cos(theta_r), np.cos(phi_r)
                  * np.cos(theta_r), np.sin(theta_r)])
    return np.dot(magnetisation, p)

def get_xmcd_color(xmcd):
    """Gets the color of the xmcd vector in grayscale"""
    # get the colormap
    norm = colors.Normalize(vmin=-1, vmax=1)
    cmap = cm.gray
    # get the mapping function for colors
    cmap_fun = cm.ScalarMappable(norm=norm, cmap=cmap)
    # convert xmcd to color
    xmcd_color = cmap_fun.to_rgba(xmcd)

    return xmcd_color

def project_structure(struct, phi=0, theta=15, n=np.array([0, 0, 1])):
    """Projects the structure along a vector with polar angles phi and theta to the plane with normal n. Returns a trimesh structure."""
    struct_projected = struct.copy()
    struct_projected.vertices = project_points(
        struct_projected.vertices, phi=phi, theta=theta, n=n)
    return struct_projected


class Visualizer(object):
    """Object for visualising the xmcd projection"""
    def __init__(self, struct, magnetisation):

        # add the file attributes
        self.struct = struct
        self.magnetisation = magnetisation
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        self.app = app

        # create the view
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(0.5)
        self.view.opts['distance'] = 10000
        self.view.setGeometry(100, 100, 1000, 1000)

        # generate the view
        self.generate_view()

    def update_data(self, struct, magnetisation):
        # add the file attributes
        self.structure_file = struct
        self.magnetisation = magnetisation


    def generate_view(self):
        # remove all items
        while len(self.view.items) != 0:
            self.view.removeItem(self.view.items[0])
        # struct = trimesh.load(self.structure_file)
        struct = self.struct.copy()
        magnetisation = self.magnetisation
        # ground the structure
        struct.vertices -= np.min(struct.vertices, axis=0)[np.newaxis, :]
        # with open(self.magnetisation_file, 'rb') as f:
        #     magnetisation = pickle.load(f)
        # calculate the xmcd based on the magnetisation
        xmcd = calculate_xmcd(magnetisation)
        # get the xmcd color
        xmcd_color = get_xmcd_color(xmcd)
        xmcd_color_inv = get_xmcd_color(-xmcd)

        # get the projected structure
        struct_projected = project_structure(struct)

        # associate colors with faces
        face_colors = xmcd_color[struct.faces[:, 0], :]
        face_colors_inv = xmcd_color_inv[struct.faces[:, 0], :]

        # create meshes
        self.meshdata = gl.MeshData(vertexes=struct.vertices, faces=struct.faces,
                                    faceColors=face_colors)
        self.meshdata_projected = gl.MeshData(vertexes=struct_projected.vertices,
                                              faces=struct_projected.faces,
                                              faceColors=face_colors_inv)

        self.mesh = gl.GLMeshItem(meshdata=self.meshdata, smooth=False,
                                  drawFaces=True, drawEdges=False,
                                  shader='balloon')
        self.mesh_projected = gl.GLMeshItem(meshdata=self.meshdata_projected, smooth=False,
                                            drawFaces=True, drawEdges=False,
                                            shader='balloon')
        self.view.addItem(self.mesh_projected)
        self.view.addItem(self.mesh)


    def show(self):
        self.view.show()

    def set_camera(self, ele=90, azi=30, dist=5e5, fov=1, center=[0, -3000, 0]):
        self.view.opts['elevation'] = ele
        self.view.opts['azimuth'] = azi
        self.view.opts['distance'] = dist
        self.view.opts['fov'] = fov
        self.view.opts['center'] = Vector(center[0], center[1], center[2])

    def save_render(self, filename, size=(1024, 1024)):
        img = self.view.renderToArray(size)
        saved = pg.makeQImage(img).save(filename)

    def start(self):
        self.view.show()
        self.view.raise_()
        # del app
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()


# Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    structure_file = 'SOL6-1-3.stl'
    magnetisation_file = 'SOL6-1-3_uniform_mag.p'


    struct = trimesh.load(structure_file)
    with open(magnetisation_file, 'rb') as f:
        magnetisation = pickle.load(f)

    v = Visualizer(struct, magnetisation)
    v.generate_view()
    v.save_render('test1.png')
    # v.animation()
