def test_import():
    import jackknify

    assert hasattr(jackknify, "__version__")


def test_version_is_string():
    from jackknify import __version__

    assert isinstance(__version__, str)
