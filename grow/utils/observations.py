import numpy as np


def get_voxel_material_proportions(X, x, y, z, materials):
    proportions = []  # Ordered by -x, +x, -y, ...

    def check_for_zero(x, y):
        if x == 0 or y == 0:
            return 0, 0
        return x, y
    # x axis positive.
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[x:, :, :] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    # x axis negative.
    v = min(x + 1, X.shape[0])
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[:v, :, :] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    # y axis positive.
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[:, y:, :] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    # x axis negative.
    v = min(y + 1, X.shape[1])
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[:, :v, :] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    # z axis positive.
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[:, :, z:] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    # z axis negative.
    v = min(z + 1, X.shape[2])
    material_totals = []
    for m in materials:
        material_totals.append(np.sum(X[:, :, :v] == m))
    for i in range(len(materials)):
        total = np.sum(material_totals)
        if total == 0:
            proportions.append(0)
        else:
            proportions.append(material_totals[i] / total)

    return proportions
