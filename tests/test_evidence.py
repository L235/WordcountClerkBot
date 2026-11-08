"""Regression coverage for evidence party limits (no Wikipedia requests)."""

import pytest

import bot


@pytest.fixture
def parse_evidence(mock_site, monkeypatch):
    monkeypatch.setattr(bot, "CFG", bot.BotConfig())
    monkeypatch.setattr(bot, "visible_word_count", lambda *args: 103)
    monkeypatch.setattr(bot, "rendered_word_count", lambda *args: 103)

    def parse(party, heading, extra="", template="userlinks"):
        case = "=== Involved parties ===\n* {{" + template + "|1=" + party + "}}\n"
        monkeypatch.setattr(bot, "fetch_page", lambda *args: case)
        parser = bot.EvidenceParser(mock_site, "Example")
        parser.board_page = "Wikipedia:Arbitration/Requests/Case/Example/Evidence"
        text = f"== Evidence presented by {heading} ==\n{extra}\nEvidence text."
        return parser.parse(text)

    return parse


@pytest.mark.parametrize("template", ["admin", "adminlinks", "userlinks"])
@pytest.mark.parametrize(
    "party,heading,expected",
    [
        ("wbm1058", "Wbm1058", 1000),
        ("Wbm1058", "wbm1058", 1000),
        ("Wbm1058", "Wbm1058", 1000),
        ("wbm1058", "wbm1058", 1000),
        ("Example_User", "Example User", 1000),
        ("Example User", "example_User", 1000),
        (" example__User ", "Example User", 1000),
        ("NinjaRobotPirate", "ninjaRobotPirate", 1000),
        ("NinjaRobotPirate", "Ninjarobotpirate", 500),
        ("OtherUser", "Wbm1058", 500),
        ("", "Wbm1058", 500),
    ],
)
def test_party_limit(parse_evidence, template, party, heading, expected):
    statement = parse_evidence(party, heading, template=template)[0].statements[0]
    assert statement.limit == expected


@pytest.mark.parametrize("party", ["wbm1058", "SomeoneElse"])
def test_approved_limit_takes_precedence(parse_evidence, party):
    statement = parse_evidence(
        party, "Wbm1058", "{{ApprovedWordLimit|words=1500}}"
    )[0].statements[0]
    assert statement.limit == 1500


def test_correct_limit_reaches_status_template_data(parse_evidence):
    tables = parse_evidence(
        "wbm1058", "Wbm1058", "{{ACWordStatus|user=Wbm1058}}"
    )
    statement = tables[0].statements[0]
    assert statement.user == "Wbm1058"
    assert statement.has_status_template
    assert statement.anchor == "Evidence_presented_by_Wbm1058"
    data = bot.ParsedData(boards={}, evidence={"Example": tables})
    output = bot.assemble_data_template_from_data(data)
    assert "| Wbm1058 =" in output
    assert "| limit       = 1000" in output


def test_airline_destinations_party_list(mock_site, monkeypatch):
    # Involved parties at revision 1383099992 (the case revision in effect
    # when evidence revision 1383740179 was posted on 4 November 2026).
    case = """== Case information ==
=== Involved parties ===
*{{userlinks|10mmsocket}}
*{{userlinks|Danners430}}
*{{userlinks|Electricmemory}}
*{{userlinks|Longhuasz}}
*{{userlinks|Norco3921}}
*{{userlinks|Reywas92}}
*{{userlinks|Suaimhneas26}}
*{{adminlinks|Wbm1058}}
*{{userlinks|Vanguard10}}
===Prior dispute resolution===
{{userlinks|NotAParty}}
"""
    monkeypatch.setattr(bot, "fetch_page", lambda *args: case)
    parties = bot.extract_involved_parties(mock_site, "Airline destinations")
    assert len(parties) == 9
    assert "Wbm1058" in parties
    assert "NotAParty" not in parties
