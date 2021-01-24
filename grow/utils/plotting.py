import pyvista as pv
import vtk
import numpy as np


def plot_voxels(position_tuples, values):
    x = np.array(position_tuples)
    scalars = np.array(values)
    n = x.shape[0]

    # Add each of the voxel's vertrices.
    # Lower level.
    p1, p2, p3, p4 = (
        x.copy(),
        x.copy(),
        x.copy(),
        x.copy(),
    )
    p2[:, 0] += 1
    p3[:, 0] += 1
    p3[:, 1] += 1
    p4[:, 1] += 1

    # Upper level.
    p5 = x.copy()
    p5[:, 2] += 1
    p6, p7, p8 = (
        p5.copy(),
        p5.copy(),
        p5.copy(),
    )
    p6[:, 0] += 1
    p7[:, 0] += 1
    p7[:, 1] += 1
    p8[:, 1] += 1

    # Weave the eight coordinates of the voxel together.
    num_vertices = n * 8
    points = np.empty((num_vertices, 3), dtype=x.dtype)
    points[0::8, :] = p1
    points[1::8, :] = p2 
    points[2::8, :] = p3
    points[3::8, :] = p4 
    points[4::8, :] = p5
    points[5::8, :] = p6
    points[6::8, :] = p7
    points[7::8, :] = p8

    # Prepare the vtk grid object and plot.
    cells_hex = np.arange(num_vertices).reshape((n, 8))
    grid = pv.UnstructuredGrid({vtk.VTK_HEXAHEDRON: cells_hex}, points)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        grid,
        show_edges=True,
        scalars=scalars,
        cmap="cool",
    )
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img
