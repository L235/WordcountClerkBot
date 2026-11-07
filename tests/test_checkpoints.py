"""Revision checkpoints must survive no-ops, partial saves, and concurrent edits."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import bot
import wiki_state


@pytest.fixture
def wiki(monkeypatch, tmp_path, mock_site):
    monkeypatch.setattr(bot, "CFG", bot.BotConfig(state_dir=str(tmp_path)))
    monkeypatch.setattr(bot, "run_fingerprint", lambda site: "version-1")
    now = [10000.0]
    monkeypatch.setattr(bot.time, "time", lambda: now[0])
    titles = bot.source_titles([]) + [bot.CFG.target_page, bot.CFG.data_page, bot.CFG.extended_page]
    revisions = {title: i + 1 for i, title in enumerate(titles)}
    contents = {rev: "" for rev in revisions.values()}
    metadata_calls, content_calls = [], []
    def metadata(site, names):
        metadata_calls.append(names)
        return {name: revisions.get(name, 0) for name in names}
    def texts(site, ids):
        content_calls.append(dict(ids))
        return {name: contents[rev] if rev else "" for name, rev in ids.items()}
    monkeypatch.setattr(bot, "read_revisions", metadata)
    monkeypatch.setattr(bot, "read_contents", texts)
    monkeypatch.setattr(bot, "CACHE", None)
    saves = []
    fail_on = [None]
    def page(site, title):
        obj = MagicMock()
        def save(**kwargs):
            if title == fail_on[0]:
                raise OSError("save failed")
            revision = max(contents) + 1
            revisions[title] = revision
            contents[revision] = obj.text
            obj.latest_revision_id = revision
            saves.append(title)
            return True
        obj.save.side_effect = save
        return obj
    monkeypatch.setattr(bot.pywikibot, "Page", page)
    return SimpleNamespace(site=mock_site, revisions=revisions, contents=contents,
                           metadata=metadata_calls, reads=content_calls, saves=saves,
                           fail_on=fail_on, now=now)


def edit(wiki, title, text):
    revision = max(wiki.contents) + 1
    wiki.contents[revision] = text
    wiki.revisions[title] = revision
    return revision


def test_unchanged_poll_batches_metadata_without_reading_content(wiki):
    bot.run_logic(wiki.site)
    wiki.metadata.clear(); wiki.reads.clear(); wiki.saves.clear()
    bot.run_logic(wiki.site)
    assert len(wiki.metadata) == 1
    assert wiki.reads == [] and wiki.saves == []


def test_noop_edit_advances_checkpoint(wiki):
    bot.run_logic(wiki.site)
    new_revision = edit(wiki, bot.CFG.ae_page, "<!-- non-counted edit -->")
    wiki.saves.clear()
    bot.run_logic(wiki.site)
    assert wiki.saves == []
    assert wiki_state.load_checkpoint(bot.checkpoint_path())["sources"][bot.CFG.ae_page] == new_revision
    assert bot.prepare_run(wiki.site) is None


@pytest.mark.parametrize("failed_output", ["data_page", "extended_page"])
def test_partial_save_retried_even_after_restart(wiki, failed_output):
    wiki.fail_on[0] = getattr(bot.CFG, failed_output)
    with pytest.raises(OSError):
        bot.run_logic(wiki.site)
    assert bot.CFG.target_page in wiki.saves
    assert not bot.checkpoint_path().exists()
    wiki.fail_on[0] = None
    bot.run_logic(wiki.site)
    assert bot.prepare_run(wiki.site) is None
    assert wiki.saves.count(bot.CFG.target_page) == 1


def test_failed_save_does_not_advance_existing_checkpoint(wiki, monkeypatch):
    bot.run_logic(wiki.site)
    old = bot.checkpoint_path().read_bytes()
    edit(wiki, bot.CFG.ae_page, "new input")
    monkeypatch.setattr(bot, "generate_reports", lambda data: bot.ReportContent("new", "new", "new", "stats"))
    wiki.fail_on[0] = bot.CFG.extended_page
    with pytest.raises(OSError):
        bot.run_logic(wiki.site)
    assert bot.checkpoint_path().read_bytes() == old
    assert bot.prepare_run(wiki.site) is not None


def test_source_edited_during_publication_is_processed_next_time(wiki, monkeypatch):
    processed = wiki.revisions[bot.CFG.ae_page]
    real_publish = bot.publish_changes
    def publish(site, content, plan):
        edit(wiki, bot.CFG.ae_page, "changed while publishing")
        return real_publish(site, content, plan)
    monkeypatch.setattr(bot, "publish_changes", publish)
    bot.run_logic(wiki.site)
    assert wiki_state.load_checkpoint(bot.checkpoint_path())["sources"][bot.CFG.ae_page] == processed
    assert bot.prepare_run(wiki.site) is not None


def test_case_roots_watched_and_new_case_discovered(wiki):
    root = "Wikipedia:Arbitration/Requests/Case/Example"
    edit(wiki, bot.CFG.open_cases_page, "{{ArbComOpenTasks/line|mode=case|name=Example}}")
    edit(wiki, root, "=== Involved parties ===\n{{userlinks|Alice}}")
    edit(wiki, root + "/Evidence", "")
    bot.run_logic(wiki.site)
    assert bot.prepare_run(wiki.site) is None
    edit(wiki, root, "=== Involved parties ===\n{{userlinks|Bob}}")
    assert bot.prepare_run(wiki.site) is not None


@pytest.mark.parametrize("field", ["target_page", "data_page", "extended_page"])
def test_missing_output_repaired(wiki, field):
    bot.run_logic(wiki.site)
    title = getattr(bot.CFG, field)
    wiki.revisions[title] = 0
    bot.run_logic(wiki.site)
    assert wiki.revisions[title] != 0


def test_missing_source_never_clears_outputs(wiki):
    wiki.revisions[bot.CFG.ae_page] = 0
    with pytest.raises(RuntimeError, match="source"):
        bot.run_logic(wiki.site)
    assert wiki.saves == []


def test_render_refresh_is_not_postponed_by_source_edits(wiki, monkeypatch):
    cache = MagicMock()
    monkeypatch.setattr(bot, "CACHE", cache)
    bot.run_logic(wiki.site)
    wiki.now[0] += 100
    edit(wiki, bot.CFG.ae_page, "<!-- edit -->")
    bot.run_logic(wiki.site)
    assert cache.clear.call_count == 1
    wiki.now[0] += bot.CFG.render_refresh_interval
    bot.run_logic(wiki.site)
    assert cache.clear.call_count == 2
    assert bot.prepare_run(wiki.site) is None


def test_code_config_change_forces_reconciliation(wiki, monkeypatch):
    bot.run_logic(wiki.site)
    monkeypatch.setattr(bot, "run_fingerprint", lambda site: "version-2")
    plan = bot.prepare_run(wiki.site)
    assert plan is not None and plan.refresh


def test_malformed_checkpoint_starts_clean(wiki):
    bot.checkpoint_path().write_text('{"cases":42}')
    bot.run_logic(wiki.site)
    assert bot.prepare_run(wiki.site) is None


def test_old_revision_contents_are_read_even_if_current_page_changed(mock_site):
    mock_site.simple_request.return_value.submit.return_value = {"query": {"pages": [
        {"title": "Example", "revisions": [{"revid": 123, "slots": {"main": {"content": "old text"}}}]}
    ]}}
    assert wiki_state.read_contents(mock_site, {"Example": 123}) == {"Example": "old text"}
    assert mock_site.simple_request.call_args.kwargs["revids"] == "123"


def test_batched_revisions_normalize_titles_and_missing_pages(mock_site):
    mock_site.simple_request.return_value.submit.return_value = {"query": {
        "normalized": [{"from": "example_page", "to": "Example page"}],
        "pages": [{"title": "Example page", "revisions": [{"revid": 7}]},
                  {"title": "Missing", "missing": True}],
    }}
    assert wiki_state.read_revisions(mock_site, ["example_page", "Missing"]) == {"example_page": 7, "Missing": 0}


def test_missing_revision_content_fails(mock_site):
    mock_site.simple_request.return_value.submit.return_value = {"query": {"pages": []}}
    with pytest.raises(KeyError):
        wiki_state.read_contents(mock_site, {"Example": 123})
