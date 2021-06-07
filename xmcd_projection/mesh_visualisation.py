from .image import get_blurred_image
from .color import *
from pyqtgraph.Qt import QtCore, QtWidgets
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph import Vector
import numpy as np
import sys


class PyQtVisualizer():
    """Parent class for the visualizers using PyQt5.
    """

    def __init__(self, dist=10000, background_color=0.5):
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        self.app = app

        # create the view
        self.background_color = background_color
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor(self.background_color)
        self.view.setGeometry(100, 100, 1000, 1000)
        self.view.opts['distance'] = dist

    def show(self, **kwargs):
        """Show the visualization. All kwags are passed as camera parameters.
        """
        self.view.show()
        self.set_camera(**kwargs)

    def set_camera(self, ele=None, azi=None, dist=None, fov=None, center=None):
        """Sets the camera angle and position. Any kwag that is not passed is kept as is.

        Args:
            ele (float, optional): Elevation angle. Defaults to None.
            azi (float, optional): Azimuthal angle in deg. Defaults to None.
            dist (float, optional): Distance. Defaults to None.
            fov (float, optional): fov angle. Defaults to None.
            center ((3,) array, optional): Center of view location. Defaults to None.
        """
        if ele is not None:
            self.view.opts['elevation'] = ele
        if azi is not None:
            self.view.opts['azimuth'] = azi
        if dist is not None:
            self.view.opts['distance'] = dist
        if fov is not None:
            self.view.opts['fov'] = fov
        if center is not None:
            self.view.opts['center'] = Vector(center[0], center[1], center[2])
        self.view.repaint()

    def save_render(self, filename, size=(1024, 1024)):
        """Saves the render to filename with the givern resolution.
        """
        img = self.get_view_image(size=size)
        pg.makeQImage(img).save(filename)

    def get_view_image(self, size=(1024, 1024)):
        """Gets the image from the view.
        """
        return self.view.renderToArray(size)

    def get_image_np(self, size=(1024, 1024)):
        """Gets the image in the form plottable by matplotlib"""
        img = self.get_view_image()
        img = np.swapaxes(img, 0, 1)
        return img

    def start(self):
        """Starts the display and pyqt interactivity
        """
        self.view.show()
        self.view.raise_()
        # del app
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            self.app.exec_()


class MeshVisualizer(PyQtVisualizer):
    """Object for visualising the xmcd projection"""

    def __init__(self, struct, projected_struct, projected_xmcd=None, struct_colors=None):
        """Initializes the class

        Args:
            struct (trimesh.Trimesh)
            projected_struct (trimesh.Trimesh)
            projected_xmcd ((n,) array, optional): Values of the projected xmcd corresponding to the faces of the projected structure. Defaults to None.
            struct_colors ((m,4), optional): Colours of the structure magnetization. Defaults to None.
        """
        super().__init__()
        # add the file attributes
        self.struct = struct
        self.projected_struct = projected_struct

        if projected_xmcd is None:
            self.xmcd_color = np.zeros(
                (self.projected_struct.faces.shape[0], 4))
        else:
            self.projected_xmcd = projected_xmcd
            # get the xmcd color
            self.update_xmcd_color()
        self.struct_colors = np.zeros(
            (self.struct.faces.shape[0], 4)) if struct_colors is None else struct_colors

        # generate the view
        self.generate_view()

    def get_structs_center(self):
        """Gets the centre of the structure and the projected structure

        Returns:
            (n,3) array: Centre.
        """
        v1 = self.struct.vertices[self.struct.faces].reshape(-1, 3)
        v2 = self.projected_struct.vertices[self.projected_struct.faces].reshape(
            -1, 3)
        all_pts = np.vstack((v1, v2))
        return (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2

    def update_xmcd_color(self):
        """Updates internally the colours of the xmcd and the background.
        """
        mn, mx = self.projected_xmcd.min(), self.projected_xmcd.max()
        # get the xmcd color
        self.xmcd_color, background_color = get_xmcd_color(
            self.projected_xmcd, vmin=mn, vmax=mx)
        self.background_color = background_color[0]

    def update_colors(self, projected_xmcd, struct_colors):
        """Updates the colours of the xmcd and the structure. Repaints the view.

        Args:
            projected_xmcd ((n,) array): XMCD values of the projected structure faces.
            struct_colors ((m,4) array): Colurs of the structure faces.
        """
        self.projected_xmcd = projected_xmcd
        self.update_xmcd_color()
        # set the colors
        self.struct_colors = struct_colors

        self.meshdata.setFaceColors(self.struct_colors)
        self.meshdata_projected.setFaceColors(self.xmcd_color)
        self.mesh.meshDataChanged()
        self.mesh_projected.meshDataChanged()
        self.view.repaint()

    def view_projection(self, **kwargs):
        """Removes the display of the structure and focuses only on the XMCD.
        """
        if self.mesh_projected not in self.view.items:
            self.view.addItem(self.mesh_projected)
        if self.mesh in self.view.items:
            self.view.removeItem(self.mesh)
        self.view.setBackgroundColor(self.background_color)
        self.set_camera(**kwargs)

    def view_struct(self, **kwargs):
        """Removes the display of the projection and focuses only on the structure
        """
        if self.mesh_projected in self.view.items:
            self.view.removeItem(self.mesh_projected)
        if self.mesh not in self.view.items:
            self.view.addItem(self.mesh)
        self.view.setBackgroundColor(1.0)
        self.set_camera(**kwargs)

    def view_both(self, **kwargs):
        """View both the projection and the structure in the same render.
        """
        if self.mesh_projected not in self.view.items:
            self.view.addItem(self.mesh_projected)
        if self.mesh not in self.view.items:
            self.view.addItem(self.mesh)
        self.view.setBackgroundColor(self.background_color)
        self.set_camera(**kwargs)

    def generate_view(self):
        """Generates the view data and sets the initial camera position.
        """
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
        self.view_both()
        self.set_camera(
            azi=None, center=self.get_structs_center(), ele=90, fov=1)

    def get_blurred_image(self, sigma=4, desired_background=None):
        """Applies a Gaussian blur to the image to make it correspond to the actual measurements more"""

        img = self.get_image_np()
        if desired_background is not None:
            img = rgb2gray(rgba2rgb(img))
            background = self.background_color
            if desired_background >= background:
                new1 = background / desired_background
                img[img > new1] = new1
                img /= new1
            elif desired_background < background:
                new0 = (background - desired_background) / \
                    (1 - desired_background)
                img[img < new0] = new0
                img -= new0
                img /= 1 - new0

        return get_blurred_image(img, sigma=sigma)


class NoProjectionVisualizer(PyQtVisualizer):
    """Object for visualising only the structure. Useful when don't have the xmcd information and just want to see the structure magnetisation."""

    def __init__(self, struct, struct_colors=None):
        super().__init__(background_color=1.0)
        # add the file attributes
        self.struct = struct
        self.struct_colors = np.zeros(
            (self.struct.faces.shape[0], 4)) if struct_colors is None else struct_colors

        # generate the view
        self.generate_view()

    def update_colors(self, struct_colors):
        """Updates the structure colours

        Args:
            struct_colors ((n,4) array)
        """
        # set the colors
        self.struct_colors = struct_colors

        self.meshdata.setFaceColors(self.struct_colors)
        self.mesh.meshDataChanged()
        self.view.repaint()

    def generate_view(self):
        """Generates the view data and sets the initial camera position.
        """
        # remove all items
        while len(self.view.items) != 0:
            self.view.removeItem(self.view.items[0])
        # struct = trimesh.load(self.structure_file)

        # create meshes
        self.meshdata = gl.MeshData(vertexes=self.struct.vertices, faces=self.struct.faces,
                                    faceColors=self.struct_colors)

        self.mesh = gl.GLMeshItem(meshdata=self.meshdata, smooth=False,
                                  drawFaces=True, drawEdges=False,
                                  shader='balloon')
        self.view_struct()
        self.set_camera(
            azi=None, center=self.get_structs_center(), ele=90, fov=1)

    def get_structs_center(self):
        """Gets the centre of the structure.

        Returns:
            (n,3) array: Centre.
        """
        all_pts = self.struct.vertices[self.struct.faces].reshape(-1, 3)
        return (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
