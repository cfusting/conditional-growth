from grow.utils.fitness import (
    max_z,
    table,
    get_min_max_z,
    get_num_at_z,
    get_convex_hull_area,
    get_stability,
)
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
    assert get_convex_hull_area(x) == 25


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
    assert at == 2
    assert not_at == 2


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
