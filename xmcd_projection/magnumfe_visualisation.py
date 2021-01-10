
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph import Vector
import numpy as np
import sys
from .data_manipulation import *


class MagnumfeVisualizer(object):
    """Object for visualising the xmcd projection"""

    def __init__(self, struct, projected_struct, projected_xmcd=None, struct_colors=None):
        # add the file attributes
        self.struct = struct
        self.projected_struct = projected_struct
        if projected_xmcd is None:
            self.xmcd_color = np.zeros(
                (self.projected_struct.faces.shape[0], 4))
        else:
            self.projected_xmcd = projected_xmcd / \
                np.abs(projected_xmcd).max()
            # get the xmcd color
            self.xmcd_color = get_xmcd_color(self.projected_xmcd)
        self.struct_colors = np.zeros(
            (self.struct.faces.shape[0], 4)) if struct_colors is None else struct_colors

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

        # create meshes
        self.meshdata = gl.MeshData(vertexes=self.struct.vertices, faces=self.struct.faces,
                                    faceColors=self.struct_colors)
        self.meshdata_projected = gl.MeshData(vertexes=self.projected_struct.vertices,
                                              faces=self.projected_struct.faces,
                                              faceColors=self.xmcd_color)

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
        self.view.repaint()

    def save_render(self, filename, size=(1024, 1024)):
        img = self.view.renderToArray(size)
        saved = pg.makeQImage(img).save(filename)

    def start(self):
        self.view.show()
        self.view.raise_()
        # del app
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()
