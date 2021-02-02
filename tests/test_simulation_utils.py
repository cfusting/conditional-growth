from grow.utils.simulation import write_configs_to_base


def test_write_configs_to_base():
    write_configs_to_base("/root/conditional_growth/tests/data/base.vxa", 1000, 10, 3, 1)
