import numpy as np
from collections import deque
from scipy.spatial import ConvexHull
from grow.utils.plotting import get_vertices_of_voxel


def max_z(final_positions):
    _, max_z = get_min_max_z(final_positions)
    return max_z


def table(initial_positions):
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


def prepare_points_for_convex_hull(x):
    q = set()
    for p in x:
        q.add(p)
        q.add((p[0] + 1, p[1]))
        q.add((p[0], p[1] + 1))
        q.add((p[0] + 1, p[1] + 1))
    x = list(q)
    return x


def get_convex_hull_area(x):
    x = prepare_points_for_convex_hull(x)    
    if len(x) == 0:
        return 0
    if len(x) == 1:
        return 0
    if len(x) == 2:
        return np.sqrt((x[0][0] - x[0][1])**2 + (x[1][0] - x[1][1])**2)

    return ConvexHull(x).volume


def get_convex_hull_volume(x):
    if len(x) == 0:
        return 0
    x, _, _ = get_vertices_of_voxel(x)
    return ConvexHull(x).volume


def twist(axiom):
    """Reward a creature that twists continuously.

    This function adds one to the reward every time three voxels 
    make an L-shaped twist with only two connections.

    Examples in two dimensions:

        **  and  *
        *        **

    """
    reward = 0
    voxels = deque([axiom])    
    visited = set()

    def add_voxel(voxel):
        if voxel not in visited:
            voxels.appendleft(voxel)
            visited.add(voxel)

    while len(voxels) > 0:

        current_voxel = voxels.pop()
        count = 0
        num_x = 0
        num_y = 0
        num_z = 0

        if current_voxel.positive_x:
            count += 1
            num_x += 1
            add_voxel(current_voxel.positive_x)
        if current_voxel.negative_x:
            count += 1
            num_x += 1
            add_voxel(current_voxel.negative_x)
        if current_voxel.positive_y:
            count += 1
            num_y += 1
            add_voxel(current_voxel.positive_y)
        if current_voxel.negative_y:
            count += 1
            num_y += 1
            add_voxel(current_voxel.negative_y)
        if current_voxel.positive_z:
            count += 1
            num_z += 1
            add_voxel(current_voxel.positive_z)
        if current_voxel.negative_z:
            count += 1
            num_z += 1
            add_voxel(current_voxel.negative_z)

        if count == 2 and num_x < 2 and num_y < 2 and num_z < 2:
            reward += 1

    return reward


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
    X = np.array(initial_positions)[:, :2]
    Y = np.array(final_positions)[:, :2]
    difference = np.abs(X - Y)
    return np.any(difference >= threshold)


def distance_traveled(initial_positions, final_positions):
    X = np.array(initial_positions)[:, :2]
    Y = np.array(final_positions)[:, :2]
    return np.linalg.norm(X - Y).max()


def max_volume(X):
    s = get_surface_area(X)
    v = get_volume(X)
    if v == 0 or s == 0:
        return 0
    return v / s


def max_surface_area(X):
    s = get_surface_area(X)
    v = get_volume(X)
    if v == 0 or s == 0:
        return 0
    return s / v


def get_volume(X):
    return np.sum(X != 0)


def get_surface_area(X):
    m = X.shape[0]
    n = X.shape[1]
    z = X.shape[2]
    
    surfaces = 0
    for i in range(m):
        for j in range(n):
            for k in range(z):
                # If the cell is empty check for surfaces surrounding it.
                if X[i, j, k] == 0:
                    if i > 0 and X[i - 1, j, k] != 0:
                        surfaces += 1
                    if i < m - 1 and X[i + 1, j, k] != 0:
                        surfaces += 1
                    if j > 0 and X[i, j - 1, k] != 0:
                        surfaces += 1
                    if j < n - 1 and X[i, j + 1, k] != 0:
                        surfaces += 1
                    if k > 0 and X[i, j, k - 1] != 0:
                        surfaces += 1
                    if k < z - 1 and X[i, j, k + 1] != 0:
                        surfaces += 1
    return surfaces
