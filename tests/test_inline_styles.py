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
