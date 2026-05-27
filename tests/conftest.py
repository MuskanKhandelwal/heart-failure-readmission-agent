from __future__ import annotations

import pytest

from hf_readmit.config import settings


@pytest.fixture(scope="session")
def app_settings():
    """Shared application settings fixture for tests."""
    return settings
