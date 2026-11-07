"""Incomplete data must never become authoritative output or cached counts."""
import hashlib
import pickle
from unittest.mock import patch
from types import SimpleNamespace

import pytest
import bot


@pytest.mark.parametrize("failed_page", [
    bot.CFG.ae_page, bot.CFG.arca_page, bot.CFG.arc_page,
    bot.CFG.open_cases_page,
    "Wikipedia:Arbitration/Requests/Case/Example",
    "Wikipedia:Arbitration/Requests/Case/Example/Evidence",
])
def test_failed_source_prevents_publication(mock_site, monkeypatch, failed_page):
    def fetch(site, title):
        if title == failed_page:
            raise OSError("temporary failure")
        if title == bot.CFG.open_cases_page:
            return "{{ArbComOpenTasks/line|mode=case|name=Example}}"
        return ""
    monkeypatch.setattr(bot, "fetch_page", fetch)
    monkeypatch.setattr(bot, "prepare_run", lambda site: SimpleNamespace(texts=None, refresh=False))
    with patch.object(bot, "publish_changes") as publish:
        with pytest.raises(RuntimeError):
            bot.run_logic(mock_site)
        publish.assert_not_called()


def test_genuinely_empty_sources_are_valid(mock_site, monkeypatch):
    monkeypatch.setattr(bot, "fetch_page", lambda *args: "")
    data = bot.collect_all_data(mock_site)
    assert data.boards == {"ARCA": [], "AE": [], "ARC": []}
    assert data.evidence == {}


def test_failed_render_retries_after_restart(mock_site, tmp_path, monkeypatch):
    cache = bot.WordCountCache(str(tmp_path))
    monkeypatch.setattr(bot, "CACHE", cache)
    with patch.object(bot.pwb_api, "Request") as request:
        request.return_value.submit.side_effect = [
            OSError("outage"), {"parse": {"text": {"*": "<p>One two three.</p>"}}}
        ]
        with pytest.raises(OSError):
            bot.visible_word_count(mock_site, "{{Example}}")
        assert cache._cache == {} and cache._mem_cache_html == {}
        cache.flush()
        monkeypatch.setattr(bot, "CACHE", bot.WordCountCache(str(tmp_path)))
        assert bot.visible_word_count(mock_site, "{{Example}}") == 3
        assert bot.rendered_word_count(mock_site, "{{Example}}") == 3
        assert request.return_value.submit.call_count == 2


@pytest.mark.parametrize("response", [{}, {"parse": {}}, {"parse": {"text": {"*": None}}}])
def test_malformed_render_is_not_cached(mock_site, tmp_path, monkeypatch, response):
    cache = bot.WordCountCache(str(tmp_path))
    monkeypatch.setattr(bot, "CACHE", cache)
    with patch.object(bot.pwb_api, "Request") as request:
        request.return_value.submit.return_value = response
        with pytest.raises(RuntimeError):
            bot.visible_word_count(mock_site, "some text")
    assert cache._cache == {} and cache._mem_cache_html == {}


def test_valid_empty_render_counts_zero(mock_site, tmp_path, monkeypatch):
    monkeypatch.setattr(bot, "CACHE", bot.WordCountCache(str(tmp_path)))
    with patch.object(bot.pwb_api, "Request") as request:
        request.return_value.submit.return_value = {"parse": {"text": {"*": ""}}}
        assert bot.visible_word_count(mock_site, "{{Invisible}}") == 0


def test_legacy_cache_is_not_reused(mock_site, tmp_path, monkeypatch):
    old_key = hashlib.sha1(b"en.wikipedia.org|/w/api.php||{{Example}}").hexdigest()
    (tmp_path / "wordcount_cache.pkl").write_bytes(pickle.dumps({old_key: (999, 999)}))
    monkeypatch.setattr(bot, "CACHE", bot.WordCountCache(str(tmp_path)))
    with patch.object(bot, "_api_render", return_value="<p>One.</p>"):
        assert bot.visible_word_count(mock_site, "{{Example}}") == 1
