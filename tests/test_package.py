from __future__ import annotations

from hf_readmit import __version__


def test_package_version() -> None:
    """Smoke test that the package imports and exposes a version."""
    assert __version__ == "0.1.0"
