"""Equivalent CSS spelling must not change the visible word count."""
from unittest.mock import patch

import pytest
import bot


@pytest.mark.parametrize("style,expected", [
    ("display:none", 2),
    ("display: none", 2),
    ("DISPLAY : NONE !important", 2),
    ("display: /* comment */ none", 2),
    ("color: red; display:\nnone;", 2),
    ("text-decoration: line-through", 2),
    ("text-decoration: underline line-through solid red", 2),
    ("text-decoration-line : line-through !important", 2),
    ("display:none; display:inline", 5),
    ("display:none !important; display:inline", 2),
    ("display:none; display:inline !important", 5),
    ("text-decoration:line-through; text-decoration-line:none", 5),
    ("text-decoration-line:line-through; text-decoration:none", 5),
    ("text-decoration:none; text-decoration-line:line-through", 2),
    ("content: 'display:none'; color: red", 5),
    ("--example: display:none; color: red", 5),
    ("border: none; text-decoration: underline", 5),
    ("broken CSS; color: red", 5),
    ("display:none; display:banana", 2),
    ("display:none; display:banana !important", 2),
    ("display:none; display:block inline", 2),
    ("display:none; display:none block", 2),
    ("display:none; display:inline flex", 5),
    ("display:none; display:inline flow-root list-item", 5),
    ("text-decoration:line-through; text-decoration-line:banana", 2),
    ("text-decoration-line:underline line-through banana", 5),
    ("text-decoration:line-through banana", 5),
    ("text-decoration:line-through; text-decoration:banana !important", 2),
    ("text-decoration:line-through none", 5),
    ("text-decoration-line:line-through line-through", 5),
    ("text-decoration:line-through solid dotted", 5),
    ("text-decoration:line-through red blue", 5),
    ("text-decoration:line-through 1px 2px", 5),
    ("text-decoration-line:line-through red", 5),
    ("text-decoration:line-through wavy #f00 2px", 2),
    ("text-decoration:line-through rgb(255, 0, 0)", 2),
    ("text-decoration:line-through; text-decoration:underline solid red", 5),
    ("text-decoration:color()", 5),
    ("text-decoration:line-through color()", 5),
    ("text-decoration:line-through; text-decoration:color()", 2),
    ("text-decoration:line-through; text-decoration:color() !important", 2),
    ("text-decoration:line-through rgb()", 5),
    ("text-decoration:line-through color(banana)", 5),
])
def test_count_respects_inline_css(mock_site, mock_cache, style, expected):
    html = f'<p>Visible <span style="{style}">hidden words here</span> text.</p>'
    with patch.object(bot, "_api_render", return_value=html):
        visible, expanded = bot._count_rendered_visible(mock_site, "source")
    assert visible == expected
    assert expanded == 5


def test_nested_excluded_elements_are_safe(mock_site, mock_cache):
    html = '<p>Visible <span style="display: none">hidden <span style="display:none">nested</span></span> text.</p>'
    with patch.object(bot, "_api_render", return_value=html):
        assert bot.visible_word_count(mock_site, "source") == 2


@pytest.mark.parametrize("color,valid", [
    ("hsl(0,0,0)", False),
    ("hwb(0,0%,0%)", False),
    ("rgb(0,none,0)", False),
    ("rgb(0,0%,0)", False),
    ("rgb(0,0,0,none)", False),
    ("hsl(none,0%,0%)", False),
    ("rgb(0 0% 0)", True),
    ("rgb(0+0+0)", True),
    ("rgb(0-0-0)", True),
    ("rgb(0/**/0/**/0)", True),
    ("rgb(0%0%0%)", True),
    ("rgb(0 0 0 / none)", True),
    ("hsl(0DEG 0% 0%)", True),
    ("hsl(0 0% 0)", True),
    ("hsl(0deg,0%,0%,50%)", True),
    ("hwb(none 0% 0 / none)", True),
    ("lab(50% 0 none / none)", True),
    ("lch(50% 0 0TURN)", True),
    ("oklab(50% 0 0 / 50%)", True),
    ("oklch(50% 0 none)", True),
    ("color(srgb 0 0% none / none)", True),
    ("color(xyz-d50 0% 0 0)", True),
    ("lab(50%,0,0)", False),
    ("color(srgb 0,0,0)", False),
    ("rgb(0 0 0 none)", False),
    ("hsl(0px 0% 0%)", False),
    ("color()", False),
    ("color(banana 0 0 0)", False),
])
def test_literal_color_grammar_controls_strike_count(mock_site, mock_cache, color, valid):
    html = f'<p>Visible <span style="text-decoration:line-through {color}">three extra words</span> end.</p>'
    with patch.object(bot, "_api_render", return_value=html):
        assert bot.visible_word_count(mock_site, "source") == (2 if valid else 5)
    # Invalid declarations must not reset an earlier valid strike.
    assert bot.excluded_inline_style(f"text-decoration:line-through; text-decoration:{color}") is not valid
