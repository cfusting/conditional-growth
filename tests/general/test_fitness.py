from grow.utils.fitness import (
    max_z,
    table,
    get_min_max_z,
    get_num_at_z,
    get_convex_hull_area,
    get_convex_hull_volume,
    max_hull_volume_min_density,
    get_stability,
    has_fallen,
    get_height_from_floor,
)
import pytest
from grow.entities.growth_function import GrowthFunction
import numpy as np
import math
from numpy.testing import assert_almost_equal

def test_get_convex_hull_area_none():
    x = ()
    assert get_convex_hull_area(x) == 0


def test_get_convex_hull_area_single():
    x = ((0, 4),)
    assert get_convex_hull_area(x) == 1


def test_get_convex_hull_area_two():
    x = (
        (0, 0),
        (0, 4),
    )
    assert get_convex_hull_area(x) == 5


def test_get_convex_hull_area_three():
    x = (
        (0, 0),
        (0, 4),
        (4, 0),
    )
    assert get_convex_hull_area(x) == 17


def test_get_convex_hull_area_four():
    x = (
        (0, 0),
        (0, 4),
        (4, 0),
        (4, 4),
    )
    assert math.isclose(get_convex_hull_area(x), 25, rel_tol=0.01) is True


def test_min_max_z():
    x = (
        (0, 0, -10.3),
        (4, 7, 9),
        (0, 0, 0.5),
    )
    min_z, max_z = get_min_max_z(x)
    assert min_z == -10.3
    assert max_z == 9


def test_max_z_single_coordinate():
    x = ((0, 0, 0),)
    assert max_z(x) == 0


def test_max_z_multiple_coordinates():
    x = (
        (0, 0, 0),
        (4, 7, 9),
        (0, 0, 0.5),
    )
    assert max_z(x) == 9


def test_get_num_at_z():
    x = (
        (0, 0, 0),
        (0, 0, 0),
        (4, 7, 9),
        (0, 0, 0.5),
    )

    at, not_at = get_num_at_z(x, 3)
    assert at == 0
    assert not_at == 4

    at, not_at = get_num_at_z(x, 9)
    assert at == 1
    assert not_at == 3

    at, not_at = get_num_at_z(x, 0)
    assert at == 3
    assert not_at == 1


def test_get_stability():
    x = [
        (0, 4, 0),
        (0, 0, 0),
    ]
    assert get_stability(x, 0) == 0
    assert get_stability(x, 1) == 5

    x.append((2, 3, 3))
    assert get_stability(x, 3) == 5

    x.append((2, 6, 9))
    x.append((2, 3, 9))
    assert get_stability(x, 9) == 6

    x.append((-0.5, 4.5, 7))
    x.append((-0.5, -0.5, 7))
    x.append((4.5, -0.5, 7))
    x.append((4.5, 4.5, 7))
    assert get_stability(x, 9) == 6 + 36


def test_stupid_table():
    x = [
        (0, 0, 0),
    ]
    assert table(x) == 1

    x.append((0, 0, 1))
    assert table(x) == 2

    x.append((0, 0, 2))
    assert table(x) == 3

    x.append((1, 0, 2))
    x.append((2, 0, 2))
    assert table(x) == 3 * 3


def test_coffee_table():
    x = (
        # Three levels of legs.
        # Area of 16 each
        (0, 0, 0),
        (3, 0, 0),
        (0, 3, 0),
        (3, 3, 0),
        (0, 0, 1),
        (3, 0, 1),
        (0, 3, 1),
        (3, 3, 1),
        (0, 0, 2),
        (3, 0, 2),
        (0, 3, 2),
        (3, 3, 2),
        # Surface.
        (0, 0, 3),
        (0, 1, 3),
        (0, 2, 3),
        (0, 3, 3),
        (1, 0, 3),
        (1, 1, 3),
        (1, 2, 3),
        (1, 3, 3),
        (2, 0, 3),
        (2, 1, 3),
        (2, 2, 3),
        (2, 3, 3),
        (3, 0, 3),
        (3, 1, 3),
        (3, 2, 3),
        (3, 3, 3),
    )

    assert table(x) == 4 * 16 * ((3 * 16) / (3 * 4))


def test_has_fallen():
    x = [
        [0, 0, 0],
        [1, 0, 0],
    ]
    # Fall right.
    # Bottom left corner goes up one, right one.
    y = [
        [1, 0, 1],
        [2, 0, 1],
    ]

    assert has_fallen(x, x) == False
    assert has_fallen(x, y) == True


def test_has_fallen_and_max_z_example():
    x = (
        np.array(
            [
                [0.030000, 0.020000, 0.010000],
                [0.020000, 0.030000, 0.010000],
                [0.030000, 0.030000, 0.010000],
                [0.030000, 0.040000, 0.010000],
                [0.020000, 0.020000, 0.020000],
                [0.030000, 0.020000, 0.020000],
                [0.010000, 0.030000, 0.020000],
                [0.020000, 0.030000, 0.020000],
                [0.030000, 0.030000, 0.020000],
                [0.040000, 0.030000, 0.020000],
                [0.020000, 0.040000, 0.020000],
                [0.030000, 0.040000, 0.020000],
                [0.030000, 0.010000, 0.030000],
                [0.030000, 0.020000, 0.030000],
                [0.020000, 0.030000, 0.030000],
                [0.030000, 0.030000, 0.030000],
                [0.020000, 0.040000, 0.030000],
                [0.030000, 0.040000, 0.030000],
            ]
        )
        / 0.01
    )
    y = (
        np.array(
            [
                [0.029912, 0.020071, -0.000042],
                [0.019936, 0.030082, -0.000070],
                [0.029912, 0.030064, -0.000033],
                [0.029904, 0.040058, -0.000032],
                [0.019840, 0.020061, 0.009774],
                [0.029837, 0.020061, 0.009892],
                [0.009847, 0.030066, 0.009618],
                [0.019846, 0.030058, 0.009823],
                [0.029842, 0.030054, 0.009912],
                [0.039843, 0.030051, 0.009899],
                [0.019849, 0.040052, 0.009804],
                [0.029842, 0.040053, 0.009920],
                [0.029735, 0.010024, 0.019686],
                [0.029740, 0.020023, 0.019848],
                [0.019738, 0.030048, 0.019795],
                [0.029745, 0.030036, 0.019894],
                [0.019744, 0.040052, 0.019793],
                [0.029746, 0.040043, 0.019901],
            ]
        )
        / 0.01
    )

    assert has_fallen(x, y) == False
    assert math.isclose(max_z(y), 1.9901, rel_tol=0.01) is True

    def test_should_fall_example():
        x = (
            np.array(
                [0.010000, 0.010000, 0.000000],
                [0.000000, 0.010000, 0.010000],
                [0.010000, 0.010000, 0.010000],
                [0.010000, 0.020000, 0.010000],
                [0.010000, 0.010000, 0.020000],
            )
            / 0.01
        )
        y = (
            np.array(
                [0.010120, 0.009879, -0.000010],
                [-0.003517, 0.013550, -0.000020],
                [0.004345, 0.015657, 0.005742],
                [0.006456, 0.023518, -0.000020],
                [-0.001465, 0.021471, 0.011418],
            )
            / 0.01
        )

        assert has_fallen(x, y) is True


@pytest.fixture
def growth_function():
    return GrowthFunction(
        materials=(0, 1),
        max_voxels=6,
        search_radius=3,
        axiom_material=1,
        num_timestep_features=1,
        max_steps=10,
    )


def get_configuration_index(c, g):
    c_i = list(g.configuration_map.keys())[list(g.configuration_map.values()).index(c)]
    return c_i


def test_volume_none():
    assert get_convex_hull_volume([]) == 0


def test_volume_one():
    assert get_convex_hull_volume([[1, 1, 1]]) == 1


def test_volume_two():
    assert get_convex_hull_volume([[1, 1, 1], [2, 1, 1]]) == 2


def test_volume_two_apart():
    assert_almost_equal(3, get_convex_hull_volume([[1, 1, 1], [3, 1, 1]]))
    

def test_volume_eight_apart():
    points = [
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [2, 2, 0],
        [0, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
        [2, 2, 1],
    ]
    assert_almost_equal(18, get_convex_hull_volume(points))

    
def test_hull_density_none():
    assert max_hull_volume_min_density([]) == 0


def test_hull_density_one():
    assert max_hull_volume_min_density([[1, 1, 1]]) == 1


def test_hull_density_two():
    assert max_hull_volume_min_density([[1, 1, 1], [2, 1, 1]]) == 1


def test_hull_density_two_apart():
    assert_almost_equal(3 / 2, max_hull_volume_min_density([[1, 1, 1], [3, 1, 1]]))
    

def test_hull_density_eight_apart():
    points = [
        [0, 0, 0],
        [2, 0, 0],
        [0, 2, 0],
        [2, 2, 0],
        [0, 0, 1],
        [2, 0, 1],
        [0, 2, 1],
        [2, 2, 1],
    ]
    assert_almost_equal(18 / 8, max_hull_volume_min_density(points))

    
def test_get_height_from_floor():
    y = get_height_from_floor(np.zeros((0, 0, 0)), [1], floor_index=0)
    assert y == 0

    y = get_height_from_floor(np.zeros((1, 1, 1)), [1], floor_index=0)
    assert y == 1

    y = get_height_from_floor(np.ones((1, 1, 1)), [1], floor_index=0)
    assert y == 1

    X = np.zeros((3, 3, 3))
    X[0, :, 0] = np.array([[1, 1, 1]])
    y = get_height_from_floor(X, [1], floor_index=0)
    assert y == 3

    assert get_height_from_floor(X, [1], floor_index=1) == 2
    assert get_height_from_floor(X, [1], floor_index=2) == 1

    X[0, 1, 0] = 0
    y = get_height_from_floor(X, [1], floor_index=0)
    assert y == 1

    X[1, 0, 0] = 1
    y = get_height_from_floor(X, [1], floor_index=0)
    assert y == 1

    X[2, 0, 0] = 1
    y = get_height_from_floor(X, [1], floor_index=0)
    assert y == 1

    X[2, 1, 0] = 1
    y = get_height_from_floor(X, [1], floor_index=0)
    assert y == 2

    assert get_height_from_floor(X, [1], floor_index=1) == 1
    assert get_height_from_floor(X, [1], floor_index=2) == 1
