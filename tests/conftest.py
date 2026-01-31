"""Shared pytest fixtures for WordcountClerkBot tests."""

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_site():
    """Create a mock pywikibot APISite."""
    site = MagicMock()
    site.hostname.return_value = "en.wikipedia.org"
    site.apipath.return_value = "/w/api.php"
    return site


@pytest.fixture
def mock_cache():
    """Create and install a mock WordCountCache."""
    with patch("bot.CACHE") as cache:
        cache.get.return_value = None  # Cache miss by default
        cache.get_rendered_html.return_value = None  # HTML cache miss
        yield cache


@pytest.fixture
def api_render_mock():
    """Fixture to mock _api_render and return controlled HTML."""
    with patch("bot._api_render") as mock:
        yield mock
