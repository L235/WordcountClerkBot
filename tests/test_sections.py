"""Statement ownership includes children but never the next author's heading."""
import pytest
import bot


@pytest.fixture
def captured(monkeypatch):
    bodies = {}
    def make(self, user, body, anchor, limit=None):
        bodies[user] = body
        return bot.Statement(user, anchor, 1, 1, limit or 500)
    monkeypatch.setattr(bot.BaseParser, "_make_statement", make)
    return bodies


def test_evidence_does_not_include_next_heading(mock_site, captured):
    parser = bot.EvidenceParser(mock_site, "Example", [])
    parser.parse("== Evidence presented by Alice ==\nFirst.\n=== Details ===\nMore.\n== Evidence presented by Bob ==\nLast.")
    assert "Details" in captured["Alice"] and "More." in captured["Alice"]
    assert "Bob" not in captured["Alice"]
    assert captured["Bob"].strip() == "Last."


def test_placeholder_next_heading_is_not_evidence(mock_site, captured):
    bot.EvidenceParser(mock_site, "Example", []).parse(
        "== Evidence presented by Alice ==\nEvidence.\n== Evidence presented by {your user name} ==\nInstructions.")
    assert list(captured) == ["Alice"]
    assert captured["Alice"].strip() == "Evidence."


@pytest.mark.parametrize("level", [3, 4])
def test_ae_includes_child_subsections(mock_site, captured, level):
    h = "=" * level
    child = h + "="
    bot.AEParser(mock_site).parse(
        f"== Request ==\n{h} Statement by Alice {h}\nIntro.\n{child} Details {child}\nAdditional evidence.\n{h} Statement by Bob {h}\nOther.")
    assert "Additional evidence." in captured["Alice"]
    assert "Bob" not in captured["Alice"] and "Other." not in captured["Alice"]


def test_nested_author_not_double_counted(mock_site, captured):
    bot.AEParser(mock_site).parse("== Request ==\n=== Statement by Alice ===\nAlice.\n==== Statement by Bob ====\nBob.")
    assert "Bob" not in captured["Alice"]
    assert captured["Bob"].strip() == "Bob."


def test_parent_continues_after_nested_replies_without_inheriting_limits(mock_site, monkeypatch):
    bodies = []
    monkeypatch.setattr(bot, "visible_word_count", lambda site, body, title: bodies.append(body) or 1)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 1)
    parser = bot.AEParser(mock_site)
    parser.board_page = "Example"
    text = ("== Request ==\n=== Statement by Alice ===\nAlice intro.\n"
            "==== Statement by Bob ====\n{{ApprovedWordLimit|words=1000}}\nBob reply.\n"
            "===== Reply detail =====\nBob details.\n"
            "==== Additional evidence ====\nAlice continuation.\n"
            "==== Statement by Carol ====\nCarol reply.\n"
            "==== Final evidence ====\nAlice conclusion.\n")
    alice, bob, carol = parser.parse(text)[0].statements
    assert (alice.user, alice.limit) == ("Alice", 500)
    assert (bob.user, bob.limit) == ("Bob", 1000)
    assert carol.user == "Carol"
    assert all(part in bodies[0] for part in ("Alice intro.", "Alice continuation.", "Alice conclusion."))
    assert "Bob" not in bodies[0] and "Carol" not in bodies[0]
    assert "Bob reply." in bodies[1] and "Bob details." in bodies[1]
    assert "Alice" not in bodies[1] and "Carol" not in bodies[1]
    assert bodies[2].strip() == "Carol reply."


def test_request_field_includes_child_heading_but_not_next_field(mock_site):
    text = "== Case ==\n=== Request concerning Bob ===\n;Additional comments:\nFirst.\n==== Details ====\nMore.\n;Notification:\nExcluded.\n=== Discussion ===\nOther."
    request = bot.scan_sections(text)[1]
    body = bot.AEParser(mock_site)._collect_request_body(request, text)
    assert "More." in body
    assert "Excluded" not in body and "Other" not in body


def test_filing_excludes_nested_author_and_their_extension(mock_site, monkeypatch):
    bodies = []
    monkeypatch.setattr(bot, "visible_word_count", lambda site, body, title: bodies.append(body) or 1)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 1)
    parser = bot.AEParser(mock_site)
    parser.board_page = "Example"
    text = ("== Request ==\n=== Request concerning Bob ===\n"
            "; User who is submitting this request : {{userlinks|Alice}}\n"
            "; Additional comments:\nAlice evidence.\n"
            "==== Statement by Bob ====\n{{ApprovedWordLimit|words=1000}}\nBob reply.\n"
            "===== Reply details =====\nBob details.\n"
            "==== More filing evidence ====\nAlice continuation.\n"
            "=== Discussion concerning Bob ===\n")
    alice, bob = parser.parse(text)[0].statements
    assert (alice.user, alice.limit) == ("Alice", 500)
    assert (bob.user, bob.limit) == ("Bob", 1000)
    assert "Alice continuation." in bodies[0]
    assert "Bob" not in bodies[0] and "ApprovedWordLimit" not in bodies[0]
    assert "Bob reply." in bodies[1] and "Alice continuation." not in bodies[1]


@pytest.mark.parametrize("parser_cls", [bot.SimpleBoardParser, bot.AEParser])
def test_closed_requests_and_placeholder_statistics(mock_site, captured, parser_cls):
    tables = parser_cls(mock_site).parse(
        "== Closed case ==\n{{hat|Closed}}\n=== Statement by Alice ===\nText.\n{{hab}}\n"
        "== Open case ==\n=== Statement by (username) ===\n"
        "=== Statement by (involved editor 2) ===\n=== Statement by {other-editor} ===\n")
    assert tables[0].closed
    assert tables[1].statements == []
    assert list(captured) == ["Alice"]
    report = bot.generate_reports(bot.ParsedData({"AE": tables}, {}))
    assert "0 total statements" in report.stats_summary


def test_section_offsets_preserve_immediate_and_recursive_body():
    text = "== One ==\nIntro.\n=== Child ===\nChild.\n== Two ==\nEnd."
    sections = bot.scan_sections(text)
    assert sections[0].body(text).strip() == "Intro."
    assert "Child." in sections[0].subtree_body(text)
    assert "Two" not in sections[0].subtree_body(text)
    assert text[sections[2].heading_start:].startswith("== Two ==")


@pytest.mark.parametrize("parser_cls", [bot.SimpleBoardParser, bot.AEParser])
def test_similarly_named_template_does_not_close_thread(mock_site, captured, parser_cls):
    tables = parser_cls(mock_site).parse(
        "== Request ==\n{{hatnote|A note}}\n=== Statement by Alice ===\nEvidence.")
    assert not tables[0].closed
