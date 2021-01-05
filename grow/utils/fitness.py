import numpy as np
from scipy.spatial import ConvexHull


def max_z(initial_positions, final_positions):
    if has_fallen(initial_positions, final_positions):
        return 0
    _, max_z = get_min_max_z(final_positions)
    return max_z


def table(initial_positions, final_positions):
    if has_fallen(initial_positions, final_positions):
        return 0
    _, max_z = get_min_max_z(initial_positions)
    num_at_z, num_not_at_z = get_num_at_z(initial_positions, max_z)
    # Account for 0 being height with one voxel.
    height = max_z + 1
    surface = num_at_z
    excess = num_not_at_z or 1
    stability = get_stability(initial_positions, max_z) or 1
    return height * surface * (stability / excess)


def get_min_max_z(final_positions):
    max_z = -np.inf
    min_z = np.inf
    for p in final_positions:
        if p[2] > max_z:
            max_z = p[2]
        if p[2] < min_z:
            min_z = p[2]
    return min_z, max_z


def get_num_at_z(x, z):
    at_z = 0
    not_at_z = 0
    for p in x:
        if np.floor(p[2]) == np.floor(z):
            at_z += 1
        else:
            not_at_z += 1
    return at_z, not_at_z


def get_convex_hull_area(x):
    q = set()
    for p in x:
        q.add(p)
        q.add((p[0] + 1, p[1]))
        q.add((p[0], p[1] + 1))
        q.add((p[0] + 1, p[1] + 1))
    x = list(q)

    if len(x) == 0:
        return 0
    if len(x) == 1:
        return 0
    if len(x) == 2:
        return np.sqrt((x[0][0] - x[0][1])**2 + (x[1][0] - x[1][1])**2)

    return ConvexHull(x).volume


def get_stability(x, max_z):
    descending_positions = sorted(
        x, key=lambda p: p[2], reverse=True
    )
    # Top layer is not considered.
    while np.floor(descending_positions[0][2]) == np.floor(max_z):
        descending_positions.pop(0)
        if len(descending_positions) == 0:
            return 0

    stability = 0
    # Calculate the convex hull for each layer.
    z = descending_positions[0][2]
    current_layer = []
    for p in descending_positions:
        if np.floor(p[2]) == np.floor(z):
            current_layer.append((p[0], p[1]))
        else:
            stability += get_convex_hull_area(current_layer)
            current_layer = []
            current_layer.append((p[0], p[1]))
            z = p[2]
    stability += get_convex_hull_area(current_layer)

    return stability


def has_fallen(initial_positions, final_positions, threshold=0.25):
    """Have the x or y axes moved more than half a voxel?"""

    X = np.array(initial_positions)[:, :2]
    Y = np.array(final_positions)[:, :2]
    difference = np.abs(X - Y)
    return np.any(difference >= threshold)
