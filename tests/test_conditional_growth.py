import pytest
from grow.entities.growth_function import GrowthFunction
from numpy.testing import assert_equal
import numpy as np


def get_configuration_index(c, g):
    c_i = list(g.configuration_map.keys())[list(g.configuration_map.values()).index(c)]
    return c_i


def get_search_area_volume(search_radius):
    """
    Example if search radius is 3 and we are voxel v.
    * indicates searched area in 2d (assume z is
    the same expanding upward).

    y-axis is search radius * 2 + 1.
    x-axis includes v and is thus search radius + 1.

    000****
    000****
    000v***
    000****
    000****

    """
    extent = (search_radius * 2) + 1
    return (extent ** 2) * (search_radius + 1)


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


def test_max_length(growth_function):
    assert growth_function.max_length == 22


def test_axiom_coordinate(growth_function):
    assert growth_function.axiom_coordinate == 11


def test_single_voxel_features(growth_function):
    volume = get_search_area_volume(growth_function.search_radius)

    features = []
    # Six faces to check.
    for _ in range(6):
        # Three material types.
        # All but one 0.
        # One 1.
        features.extend([(volume - 1) / volume, 1 / volume])

    # Relative coordinate of voxel.
    # Axiom is stored explicitly.
    features.extend(
        [
            growth_function.axiom_coordinate / growth_function.max_length
            for _ in range(3)
        ]
    )


def test_single_voxel_tensor(growth_function):
    X = np.zeros(
        (
            growth_function.max_length,
            growth_function.max_length,
            growth_function.max_length,
        )
    )
    X[
        growth_function.axiom_coordinate,
        growth_function.axiom_coordinate,
        growth_function.axiom_coordinate,
    ] = growth_function.axiom_material
    assert_equal(growth_function.X, X)
