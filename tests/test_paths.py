import pytest

from jennit.paths import sanitize_rel_path


def test_basic():
    assert sanitize_rel_path("") == ""
    assert sanitize_rel_path(None) == ""
    assert sanitize_rel_path("a/b/c.jpg") == "a/b/c.jpg"
    assert sanitize_rel_path("  a//b/  ") == "a/b"
    assert sanitize_rel_path("./a/./b.jpg") == "a/b.jpg"
    assert sanitize_rel_path("a\\b.jpg") == "a/b.jpg"


def test_rejected():
    bad = [
        "/abs",
        "a/../b",
        "..",
        "a/../../etc",
        "a/b\0c.jpg",
        "x" * 600,
        "a/b/" + "y" * 300,
    ]
    for p in bad:
        with pytest.raises(ValueError):
            sanitize_rel_path(p)
