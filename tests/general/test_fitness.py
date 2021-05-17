from grow.utils.fitness import (
    max_z,
    table,
    get_min_max_z,
    get_num_at_z,
    get_convex_hull_area,
    get_stability,
    has_fallen,
    twist,
)
import pytest
from grow.entities.growth_function import GrowthFunction
import numpy as np
import math


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


def test_twist_three_voxels(growth_function):
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_y"),), growth_function)
    growth_function.step(configuration_index)
    assert twist(growth_function.axiom) == 1


def test_twist_three_voxels_straight(growth_function):
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    assert twist(growth_function.axiom) == 0


def test_twist_four_voxels(growth_function):
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_y"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    assert twist(growth_function.axiom) == 2


def test_twist_four_voxels_one_twist(growth_function):
    configuration_index = get_configuration_index(((1, "positive_x"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_y"),), growth_function)
    growth_function.step(configuration_index)
    configuration_index = get_configuration_index(((1, "positive_y"),), growth_function)
    growth_function.step(configuration_index)
    assert twist(growth_function.axiom) == 1

