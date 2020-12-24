from grow.entities.conditional_growth_genome import ConditionalGrowthGenome
from numpy.testing import assert_array_equal
import numpy as np


def print_info():
    g = ConditionalGrowthGenome()
    for k in g.configuration_map:
        print(f"k: {k} | v: {g.configuration_map[k]}")

    c = ((2, "positive_x"),)
    c_i = list(g.configuration_map.keys())[list(g.configuration_map.values()).index(c)]
    print(c_i)


def get_configuration_index(c, g):
    c_i = list(g.configuration_map.keys())[list(g.configuration_map.values()).index(c)]
    return c_i


def check_state(b, v, l, s, g):
    assert g.building() is b
    assert g.num_voxels == v
    assert g.max_level == l
    assert g.steps == s


def check_representations(X, A, g):
    assert_array_equal(X, g.get_local_voxel_representation())
    assert_array_equal(A, g.to_tensor())


def get_search_area_volume(g):
    extent = (g.search_radius * 2) + 1
    return (extent ** 2) * (g.search_radius + 1)


def test_single_voxel():
    g = ConditionalGrowthGenome()

    check_state(True, 1, 0, 0, g)

    s = get_search_area_volume(g)
    features = []
    for _ in range(6):
        features.extend([(s - 1) / s, 1 / s, 0])
    check_representations(np.array(features), np.array([[[1.0]]]), g)


def test_single_addition():
    g = ConditionalGrowthGenome()

    i = get_configuration_index(((2, "positive_x"),), g)
    g.step(i)

    check_state(True, 2, 1, 1, g)

    s = get_search_area_volume(g)
    features = []
    features.extend([(s - 2) / s, 1 / s, 1 / s])
    features.extend([(s - 1) / s, 0, 1 / s])
    for _ in range(4):
        features.extend([(s - 2) / s, 1 / s, 1 / s])
    A = np.zeros((3, 3, 3))
    A[1, 1, 1] = 1
    A[2, 1, 1] = 2
    check_representations(np.array(features), A, g)


def test_blank_addition():
    g = ConditionalGrowthGenome()

    i = get_configuration_index(None, g)
    g.step(i)

    check_state(False, 1, 0, 1, g)

    check_representations(np.array([0 for _ in range(18)]), np.array([[[1.0]]]), g)


# def test_add_multiple():
#     g = ConditionalGrowthGenome()
#     check_state(True, 1, 0, 0, g)
#     check_representations(np.array([0 for _ in range(18)]), np.array([[[1.0]]]), g)
#     i = get_configuration_index(((1, "negative_z"), (2, "positive_y")), g)
#     g.step(i)
#     check_state(True, 3, 1, 1, g)
#     A = np.zeros((3, 3, 3))
#     A[1, 1, 1] = 1.0
#     A[1, 1, 0] = 1.0
#     A[1, 2, 1] = 2.0
#     check_representations(np.array([0 for _ in range(15)] + [0, 0.5, 0.5]), A, g)
#     i = get_configuration_index(None, g)
#     g.step(i)
#     check_state(True, 3, 1, 2, g)
#     check_representations(
#         np.array([0 for _ in range(6)] + [0, 1.0, 0] + [0 for _ in range(9)]), A, g
#     )
