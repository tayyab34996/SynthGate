from importlib.util import find_spec


def test_synthgate_package_is_importable() -> None:
    assert find_spec("synthgate") is not None
