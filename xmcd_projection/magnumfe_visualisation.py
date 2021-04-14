
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
        # prepare the background color variable
        self.background_color = 0.5
        if projected_xmcd is None:
            self.xmcd_color = np.zeros(
                (self.projected_struct.faces.shape[0], 4))
        else:
            self.projected_xmcd = projected_xmcd
            # get the xmcd color
            self.update_xmcd_color()
        self.struct_colors = np.zeros(
            (self.struct.faces.shape[0], 4)) if struct_colors is None else struct_colors

        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        self.app = app

        # create the view
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self.background_color)
        self.view.opts['distance'] = 10000
        self.view.setGeometry(100, 100, 1000, 1000)

        # generate the view
        self.generate_view()

    def update_xmcd_color(self):
        mn, mx = self.projected_xmcd.min(), self.projected_xmcd.max()
        # get the xmcd color
        self.xmcd_color, background_color = get_xmcd_color(
            self.projected_xmcd, vmin=mn, vmax=mx)
        self.background_color = background_color[0]

    def update_colors(self, projected_xmcd, struct_colors):
        self.projected_xmcd = projected_xmcd
        self.update_xmcd_color()
        # set the colors
        self.struct_colors = struct_colors
        self.meshdata.setFaceColors(self.struct_colors)
        self.meshdata_projected.setFaceColors(self.xmcd_color)

        self.update_view()

    def update_view(self):
        self.mesh.meshDataChanged()
        self.mesh_projected.meshDataChanged()
        self.view.setBackgroundColor(self.background_color)
        self.view.repaint()

    def hide_projection(self):
        self.projected_xmcd *= 0
        self.update_xmcd_color()
        self.update_view()

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
        self.set_camera(azi=None, center=self.get_structs_center())

    def show(self):
        self.view.show()

    def get_structs_center(self):
        all_pts = np.vstack(
            (self.struct.vertices, self.projected_struct.vertices))
        return -(all_pts.max(axis=0) - all_pts.min(axis=0)) / 2

    def set_camera(self, ele=90, azi=None, dist=5e5, fov=1, center=None):
        if ele is not None:
            self.view.opts['elevation'] = ele
        if azi is not None:
            self.view.opts['azimuth'] = azi
        self.view.opts['distance'] = dist
        self.view.opts['fov'] = fov
        if center is not None:
            self.view.opts['center'] = Vector(center[0], center[1], center[2])
        self.view.repaint()

    def save_render(self, filename, size=(1024, 1024)):
        img = self.get_view_image(size=size)
        pg.makeQImage(img).save(filename)

    def get_view_image(self, size=(1024, 1024)):
        return self.view.renderToArray(size)

    def start(self):
        self.view.show()
        self.view.raise_()
        # del app
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()
