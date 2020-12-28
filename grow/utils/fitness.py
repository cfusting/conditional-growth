import numpy as np
from scipy.spatial import ConvexHull


def max_z(x):
    _, max_z = get_min_max_z(x)
    return max_z


def table(x):
    _, max_z = get_min_max_z(x)
    num_at_z, num_not_at_z = get_num_at_z(x, max_z)
    surface = num_at_z
    excess = num_not_at_z
    stability = get_stability(x, max_z)
    print(max_z)
    print(surface)
    if stability == 0:
        return max_z * surface
    if excess == 0:
        return max_z * surface * stability
    else:
        return max_z * surface * (stability / excess)


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
        if p[2] == z:
            at_z += 1
        else:
            not_at_z += 1
    return at_z, not_at_z


def get_convex_hull_area(x):
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
    while descending_positions[0][2] == max_z:
        descending_positions.pop(0)
        if len(descending_positions) == 0:
            return 0

    stability = 0
    # Calculate the convex hull for each layer.
    z = descending_positions[0][2]
    current_layer = []
    for p in descending_positions:
        if p[2] == z:
            current_layer.append((p[0], p[1]))
        else:
            stability += get_convex_hull_area(current_layer)
            current_layer = []
            current_layer.append((p[0], p[1]))
            z = p[2]
    stability += get_convex_hull_area(current_layer)

    return stability

