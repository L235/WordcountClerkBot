"""Status metadata, exclusions, and generated data share one template contract."""
from unittest.mock import patch

import mwparserfromhell as mw
import pytest
import pywikibot
import bot


@pytest.mark.parametrize("name", [
    "ACWordStatus", "Arbitration Committee word status",
    "arbitration_committee_word_status", "Template:ACWordStatus",
    " Template:Arbitration   Committee word status ",
])
def test_all_status_names_are_excluded_and_detected(mock_site, name):
    parser = bot.BaseParser(mock_site)
    parser.board_page = "Example"
    body = "Before {{" + name + "|user=Alice}} after."
    with patch.object(bot, "visible_word_count", return_value=2) as count, patch.object(bot, "rendered_word_count", return_value=2):
        statement = parser._make_statement("Other", body, "anchor")
    assert statement.user == "Alice"
    assert statement.has_status_template
    assert count.call_args.args[1] == "Before  after."


def test_nested_signature_removed_with_entire_limit_template():
    controls = bot.parse_statement_controls(
        "Before {{ApprovedWordLimit|words=1500|sig={{U|Clerk}} signature {{small|more text}}}} after.")
    assert controls.limit == 1500
    assert controls.body == "Before  after."


def test_nested_control_templates_are_removed_safely():
    controls = bot.parse_statement_controls("{{ACWordStatus|user=Alice|extra={{ApprovedWordLimit|words=1000}}}}")
    assert controls.body == "" and controls.limit == 1000


def test_comments_nowiki_and_similarly_named_templates_are_not_controls():
    text = "<!-- {{ACWordStatus|user=Wrong}} --> <nowiki>{{ApprovedWordLimit|words=9000}}</nowiki> {{ACWordStatusExample|user=Other}}"
    controls = bot.parse_statement_controls(text)
    assert controls.body == text
    assert controls.user is None and controls.limit is None and not controls.has_status


@pytest.mark.parametrize("words", ["1000junk", "-10", "0", "invalid"])
def test_invalid_limit_does_not_override_default(mock_site, words):
    assert bot.BaseParser(mock_site)._extract_limit("{{ApprovedWordLimit|words=" + words + "}}", 500) == 500


@pytest.mark.parametrize("label", ["AE", "ARC", "ARCA"])
def test_full_board_title_is_switch_alias(label):
    statement = bot.Statement("Alice", "anchor", 123, 123, 500)
    data = bot.ParsedData({label: [bot.RequestTable("Request", "Request", [statement])]}, {})
    text = bot.assemble_data_template_from_data(data)
    switch = mw.parse(text).filter_templates(recursive=False)[0]
    full_title = bot.get_board_parsers()[label][0]
    params = list(switch.params)
    alias_index = next(i for i, param in enumerate(params) if not param.showkey and str(param.value).strip() == full_title)
    assert str(params[alias_index + 1].name).strip() == label
    assert "| words       = 123" in str(params[alias_index + 1].value)


@pytest.mark.integration
@pytest.mark.parametrize("label", ["AE", "ARC", "ARCA"])
def test_mediawiki_resolves_short_full_and_default_page_keys(label):
    """Expand generated data in memory; never edit a wiki page."""
    statement = bot.Statement("Alice", "anchor", 123, 123, 500)
    data = bot.ParsedData({label: [bot.RequestTable("Request", "Request", [statement])]}, {})
    text = bot.assemble_data_template_from_data(data)
    text = text.replace("{{{section}}}", "Request").replace("{{{user}}}", "Alice").replace("{{{type}}}", "words")
    title = bot.get_board_parsers()[label][0]
    examples = [text.replace("{{{page}}}", key) for key in (label, title, "{{FULLPAGENAME}}")]
    result = pywikibot.Site("en", "wikipedia").simple_request(
        action="expandtemplates", text=";".join(examples), title=title, prop="wikitext",
    ).submit()
    assert result["expandtemplates"]["wikitext"].replace(" ", "").replace("\n", "") == "123;123;123"
