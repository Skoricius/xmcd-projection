# This visualizer and functions here need to be updated. It is currently discontinued because mesh_visualisation is better and more accurate, but this is faster and simpler
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph import Vector
import numpy as np
import sys
from .projection import project_structure
from .color import get_xmcd_color

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'w')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer(object):
    """Object for visualising the xmcd projection"""

    def __init__(self, struct, magnetisation, p):

        # add the file attributes
        self.struct = struct
        self.p = p
        # ensure magnetisation normalized
        magnetisation /= np.linalg.norm(magnetisation, axis=1)[:, np.newaxis]
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
        struct = self.struct.copy()
        magnetisation = self.magnetisation
        # ground the structure
        struct.vertices -= np.min(struct.vertices, axis=0)[np.newaxis, :]
        # calculate the xmcd based on the magnetisation
        xmcd = np.dot(magnetisation, self.p)
        # get the xmcd color
        xmcd_color, _ = get_xmcd_color(xmcd)
        xmcd_color_inv, _ = get_xmcd_color(-xmcd)

        # get the projected structure
        struct_projected = project_structure(struct, self.p)

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
        ax = fig.add_subplot((111), projection='3d', proj_type='ortho')

    pts = struct.vertices
    ax.scatter(pts[::step, 0], pts[::step, 1], pts[::step, 2])

    if rescale:
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        max_range = np.array(
            [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0

        mid_x = (X.max() + X.min()) * 0.5
        mid_y = (Y.max() + Y.min()) * 0.5
        mid_z = (Z.max() + Z.min()) * 0.5
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
    u, v, w = magnetisation[::step,
                            0], magnetisation[::step, 1], magnetisation[::step, 2]

    ax.quiver(x, y, z, u, v, w, normalize=True,
              length=length, arrow_length_ratio=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    return ax
