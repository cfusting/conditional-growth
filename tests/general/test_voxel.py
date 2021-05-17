from grow.entities.voxel import Voxel


def test_equality():
    v1 = Voxel(1, 1, 1, 1)
    v2 = Voxel(1, 2, 1, 1)
    assert v1 == v1
    assert v1 != v2


def test_hashing():
    v1 = Voxel(1, 1, 1, 1)
    v2 = Voxel(1, 2, 1, 1)
    collection = set()
    collection.add(v1)
    assert v1 in collection

    assert v2 not in collection
    collection.add(v2)
    assert v2 in collection



