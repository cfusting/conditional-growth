from grow.utils.plotting import plot_voxels
from PIL import Image


def test_plot_voxels_single():
    lower_left = [
        (0, 0, 0) 
    ]
    values = [1]

    X = plot_voxels(lower_left, values)
    print(X.shape)
    assert X is not None
    img = Image.fromarray(X)
    img.save("/tmp/voxel.png")
