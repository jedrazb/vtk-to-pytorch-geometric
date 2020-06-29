
import numpy as np
import pyvista as pv


def create_data_list():
    '''
    Function to create a list of 3d data objects along with some metadata. The output
    of this function should be of a shape:

    {
        path: (string): absolute path to the shape
        y: (number): label
        ... (other kwargs will be passed to internal representation as well)
    }

    '''

    # data_list = []
    # data = {'path': ..., 'y': 42, ...}
    # data_list.append(data)
    # return data_list

    print('Make sure to implement create_data_list function.')
    raise NotImplementedError


def read_PolyData(filename):
    """
    Read a vtk polydata file and computes the normal at each point.    
    filename: e.g., 'Filename.vtk'
    """
    return pv.PolyData(filename)


def PolyDataToNumpy(polydata):
    """
    Processes a pyvista polydata structure into vertices, triangles with deep copy.
    To avoid deep copy and keep a link between numpy arrays and polydata points/faces, use the built-in functionalities.
    For points, the getter and setter:
        polydata.points
    For faces, also the getter and setter
        polydata.faces
    Note that faces doesn't assume that all faces are triangles. It is a more generic array that looks like
    [n1,v1_{1},...,v1_{n1}, n2,v2_{1},...,v2_{n2}, etc.]
    where ni is the number of vertices on face i, and vi_{k} is the vertex index for the kth vertex on face i.
    Our code however assumes we have triangles, and converts faces to a "tris" array of shape
    [[v1_{1},v1_{2},v1_{3}],
     [v2_{1},v2_{2},v2_{3}],
     etc.]
    input: a pv.PolyData
    """

    numpy_verts = np.copy(polydata.points).reshape(-1, 3)

    numpy_tris = np.copy(polydata.faces)
    if (numpy_tris.size) != (polydata.n_faces*4):
        raise ValueError(
            "Function only supports Polydata with triangle faces.")

    numpy_tris = numpy_tris.reshape(-1, 4)
    numpy_tris = numpy_tris[:, 1:]

    return numpy_verts, numpy_tris
