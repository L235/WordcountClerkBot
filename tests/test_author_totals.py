"""Multiple sections must share a single per-author allowance and output total."""
import pytest
import bot


def table():
    return bot.RequestTable("Request", "Request", [
        bot.Statement("Alice", "first", 600, 620, 500),
        bot.Statement("alice", "continued", 10, 15, 500, True),
        bot.Statement("Bob", "bob", 3, 3, 500),
    ])


def test_all_outputs_agree_on_combined_count():
    request = table()
    data = bot.ParsedData({"AE": [request]}, {})
    output = bot.generate_reports(data)
    for report in (output.text, output.extended):
        assert report.count("[[User:Alice|Alice]]") == 1
        assert "610 || 635 || 500 || over" in report
        assert "#first|link 1]]" in report and "#continued|link 2]]" in report
    assert "| words       = 610" in output.data
    assert "| uncollapsed = 635" in output.data
    assert "| words       = 600" not in output.data
    assert "2 total statements" in output.stats_summary
    assert "1 statements over" in output.stats_summary
    # Aggregation must not mutate parser output or double-count on another render.
    assert request.statements[0].visible == 600
    assert bot.generate_reports(data) == output


def test_extension_applies_to_combined_total_without_adding_limits():
    request = table()
    request.statements[1].limit = 1000
    total = request.author_totals()[0]
    assert total.limit == 1000 and total.visible == 610
    assert total.status == bot.Status.OK and total.has_status_template


def test_different_requests_and_internal_capitalization_remain_separate():
    request = table()
    request.statements.append(bot.Statement("ALICE", "other-account", 20, 20, 500))
    assert [s.user for s in request.author_totals()] == ["Alice", "Bob", "ALICE"]
    other = bot.RequestTable("Other", "Other", [bot.Statement("Alice", "different", 7, 7, 500)])
    assert other.author_totals()[0].visible == 7


def test_evidence_data_uses_same_total():
    request = table()
    data = bot.ParsedData({}, {"Example": [request]})
    assert "| words       = 610" in bot.assemble_data_template_from_data(data)


def test_empty_user_does_not_enter_stats_or_data():
    request = bot.RequestTable("Request", "Request", [bot.Statement("", "placeholder", 0, 0, 500)])
    data = bot.ParsedData({"AE": [request]}, {})
    output = bot.generate_reports(data)
    assert "0 total statements" in output.stats_summary
    assert "| words" not in output.data


def test_single_statement_link_format_unchanged():
    statement = bot.Statement("Alice", "first", 1, 1, 500)
    assert statement.section_links("Page") == "[[Page#first|link]]"
