"""Identity must be explicit, consistent with limits, and case-safe."""
import pytest
import bot


@pytest.mark.parametrize("header,body,expected", [
    ("HetmanTheResearcher", "Evidence about [[User:AndreJustAndre]].", "HetmanTheResearcher"),
    ("Alice", "[[User:Alicia]] wrote this.", "Alice"),
    ("Alice", "[[User:ALICE|ALICE]]", "Alice"),
    ("Vanamonde", "Signed [[User:Vanamonde93|Vanamonde]].", "Vanamonde93"),
    ("Alias", "[[User:First|Alias]] and [[User:Second|Alias]]", "Alias"),
    ("[[User:Actual|Alias]]", "Other [[User:Person]]", "Actual"),
    ("Alice (about [[User:Bob|Bob]])", "[[User:Alice|Alice]]", "Alice"),
    ("Alice (reply to [[User:Bob|Alice]])", "[[User:Alice|Alice]]", "Alice"),
    ("[[User:Actual|Alias]] (reply to [[User:Bob|Alias]])", "", "Actual"),
    ("[[User:Actual|Alias]] (about [[User:Bob|Bob]])", "", "Actual"),
    ("[[User:Alice (continued)|Alice (continued)]]", "", "Alice (continued)"),
    ("Alice (continued)", "[[User:Alice|Alice]]", "Alice"),
    ("Alice (continued)", "[[User:Alice (continued)|Alice]]", "Alice (continued)"),
    ("Example_User", "No links", "Example User"),
    ("Wrong", "{{Arbitration Committee word status|user=Actual}}", "Actual"),
])
def test_resolve_identity(header, body, expected):
    assert bot.resolve_username(header, body) == expected


@pytest.mark.parametrize("heading,body", [
    ("Alice (continued)", "{{ACWordStatus|user=Alice}}"),
    ("Alias", "{{Arbitration Committee word status|user=Alice}}"),
    ("[[User:Alice|Alias]]", "Evidence."),
    ("Alice (continued)", "[[User:Alice|Alice]]"),
])
def test_evidence_uses_resolved_author_for_limit(mock_site, monkeypatch, heading, body):
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 600)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 600)
    parser = bot.EvidenceParser(mock_site, "Example", ["Alice"])
    parser.board_page = "Example/Evidence"
    statement = parser.parse(f"== Evidence presented by {heading} ==\n{body}")[0].statements[0]
    assert statement.user == "Alice"
    assert statement.limit == 1000 and statement.status == bot.Status.OK


def test_explicit_nonparty_does_not_inherit_heading_party_limit(mock_site, monkeypatch):
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 600)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 600)
    parser = bot.EvidenceParser(mock_site, "Example", ["Alice"])
    parser.board_page = "Example/Evidence"
    statement = parser.parse("== Evidence presented by Alice ==\n{{ACWordStatus|user=Bob}}")[0].statements[0]
    assert statement.user == "Bob" and statement.limit == 500


def test_piped_heading_identity_is_preserved_on_simple_boards(mock_site, monkeypatch):
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 1)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 1)
    parser = bot.SimpleBoardParser(mock_site)
    parser.board_page = "Example"
    statement = parser.parse("== Request ==\n=== Statement by [[User:Actual|Alias]] ===\nEvidence.")[0].statements[0]
    assert statement.user == "Actual"


def test_canonical_username_data_also_accepts_underscore_form():
    statement = bot.Statement("Example User", "anchor", 12, 12, 500)
    data = bot.ParsedData({"AE": [bot.RequestTable("Request", "Request", [statement])]}, {})
    output = bot.assemble_data_template_from_data(data)
    assert "| Example_User\n         | Example User =" in output


@pytest.mark.parametrize("label", ["Bob", "Alice"])
def test_contextual_heading_link_does_not_grant_party_limit(mock_site, monkeypatch, label):
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 600)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 600)
    parser = bot.EvidenceParser(mock_site, "Example", ["Bob"])
    parser.board_page = "Example/Evidence"
    statement = parser.parse(f"== Evidence presented by Alice (about [[User:Bob|{label}]]) ==\nEvidence. [[User:Alice|Alice]]")[0].statements[0]
    assert statement.user == "Alice" and statement.limit == 500
    assert statement.status == bot.Status.OVER


@pytest.mark.parametrize("heading", [
    "<span>Statement by</span> [[User:Actual|Alias]]",
    "''Statement by'' [[User:Actual|Alias]]",
    "<span>Statement by [[User:Actual|Alias]]</span>",
    "Statement<!-- comment --> by [[User:Actual|Alias]]",
])
def test_formatted_statement_prefix_keeps_author_link(mock_site, monkeypatch, heading):
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 1)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 1)
    parser = bot.SimpleBoardParser(mock_site)
    parser.board_page = "Example"
    statement = parser.parse(f"== Request ==\n=== {heading} ===\nEvidence.")[0].statements[0]
    assert statement.user == "Actual"
