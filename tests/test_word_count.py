"""Tests for word counting functionality in bot.py.

Tests cover the _count_rendered_visible function and its various exclusion rules
for different types of content that should not be counted as visible words.
"""

import re
import pytest
from unittest.mock import MagicMock, patch

# Import the module under test
import bot


class TestWordRegex:
    """Tests for the WORD_RE pattern used for tokenization."""

    def test_simple_words(self):
        """Basic word matching."""
        assert bot.WORD_RE.findall("hello world") == ["hello", "world"]

    def test_hyphenated_words(self):
        """Hyphenated words count as single tokens."""
        assert bot.WORD_RE.findall("well-known fact") == ["well-known", "fact"]

    def test_contractions(self):
        """Contractions with apostrophes count as single tokens."""
        assert bot.WORD_RE.findall("don't won't can't") == ["don't", "won't", "can't"]

    def test_possessives(self):
        """Possessive forms count as single tokens."""
        assert bot.WORD_RE.findall("John's book") == ["John's", "book"]

    def test_numbers(self):
        """Numbers are counted as words."""
        assert bot.WORD_RE.findall("the year 2024") == ["the", "year", "2024"]

    def test_mixed_alphanumeric(self):
        """Mixed alphanumeric tokens are counted."""
        assert bot.WORD_RE.findall("version 2a is better") == [
            "version",
            "2a",
            "is",
            "better",
        ]


class TestCountRenderedVisible:
    """Tests for the _count_rendered_visible function."""

    @pytest.fixture(autouse=True)
    def setup_cache(self):
        """Set up a mock cache for each test."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.get_rendered_html.return_value = None
        with patch.object(bot, "CACHE", mock_cache):
            yield mock_cache

    @pytest.fixture
    def mock_site(self):
        """Create a mock pywikibot site."""
        site = MagicMock()
        site.hostname.return_value = "en.wikipedia.org"
        site.apipath.return_value = "/w/api.php"
        return site

    def test_simple_text(self, mock_site):
        """Plain text without any HTML produces correct counts."""
        html = "<p>Hello world this is a test.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 6, f"Expected 6 visible words, got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    def test_rendered_strips_all_html_tags(self, mock_site):
        """Rendered count strips HTML tags but counts all text."""
        html = "<p>One <strong>two</strong> <em>three</em></p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert rendered == 3, f"Expected 3 rendered words, got {rendered}"
            assert visible == 3, f"Expected 3 visible words, got {visible}"

    # Reference exclusion tests

    def test_excludes_ol_references(self, mock_site):
        """References in <ol class="references"> are excluded from visible count."""
        html = """
        <p>Main text has five words.</p>
        <ol class="references">
            <li>Reference one text here.</li>
            <li>Reference two text here.</li>
        </ol>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 5, f"Expected 5 visible words (Main text has five words), got {visible}"
            assert rendered == 13, f"Expected 13 rendered words (all including references), got {rendered}"

    def test_excludes_div_references(self, mock_site):
        """References in <div class="references"> are excluded from visible count."""
        html = """
        <p>Main content here.</p>
        <div class="references">
            <p>Reference text excluded.</p>
        </div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 3, f"Expected 3 visible words (Main content here), got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    def test_excludes_div_reflist(self, mock_site):
        """References in <div class="reflist"> are excluded from visible count."""
        html = """
        <p>The main statement.</p>
        <div class="reflist">
            <ol class="references">
                <li>A reference.</li>
            </ol>
        </div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 3, f"Expected 3 visible words (The main statement), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    # Hidden content tests

    def test_excludes_display_none(self, mock_site):
        """Elements with display:none are excluded from visible count."""
        html = """
        <p>Visible text here.</p>
        <div style="display:none">Hidden text excluded.</div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 3, f"Expected 3 visible words (Visible text here), got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    def test_excludes_display_none_case_insensitive(self, mock_site):
        """display:none matching is case-insensitive."""
        html = """
        <p>Visible words.</p>
        <div style="DISPLAY:NONE">Hidden words.</div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words, got {visible}"
            assert rendered == 4, f"Expected 4 rendered words, got {rendered}"

    def test_excludes_hidden_attribute(self, mock_site):
        """Elements with [hidden] attribute are excluded from visible count."""
        html = """
        <p>Shown text.</p>
        <p hidden>Hidden text.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Shown text), got {visible}"
            assert rendered == 4, f"Expected 4 rendered words, got {rendered}"

    def test_excludes_aria_hidden(self, mock_site):
        """Elements with aria-hidden='true' are excluded from visible count."""
        html = """
        <p>Main content.</p>
        <span aria-hidden='true'>Screen reader hidden.</span>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Main content), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    # Collapsed content tests

    def test_excludes_mw_collapsed(self, mock_site):
        """Elements with .mw-collapsed are excluded from visible count."""
        html = """
        <p>Before collapse.</p>
        <div class="mw-collapsed">Collapsed content here.</div>
        <p>After collapse.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 4, f"Expected 4 visible words (Before/After collapse), got {visible}"
            assert rendered == 7, f"Expected 7 rendered words, got {rendered}"

    def test_excludes_mw_collapsible_content(self, mock_site):
        """Elements with .mw-collapsible-content are excluded from visible count."""
        html = """
        <p>Normal text.</p>
        <div class="mw-collapsible-content">Hidden when collapsed.</div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Normal text), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    # Struck-through text tests

    def test_excludes_s_tag(self, mock_site):
        """<s> struck-through text is excluded from visible count."""
        html = "<p>Keep this <s>remove this</s> and this.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 4, f"Expected 4 visible words (Keep this and this), got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    def test_excludes_strike_tag(self, mock_site):
        """<strike> struck-through text is excluded from visible count."""
        html = "<p>Valid <strike>invalid</strike> text.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Valid text), got {visible}"
            assert rendered == 3, f"Expected 3 rendered words, got {rendered}"

    def test_excludes_del_tag(self, mock_site):
        """<del> deleted text is excluded from visible count."""
        html = "<p>Current <del>old version</del> text.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Current text), got {visible}"
            assert rendered == 4, f"Expected 4 rendered words, got {rendered}"

    def test_excludes_text_decoration_line_through(self, mock_site):
        """Text with text-decoration:line-through is excluded from visible count."""
        html = """
        <p>Normal <span style="text-decoration:line-through">struck</span> text.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Normal text), got {visible}"
            assert rendered == 3, f"Expected 3 rendered words, got {rendered}"

    # UI element tests

    def test_excludes_siteSub(self, mock_site):
        """div#siteSub is excluded from visible count."""
        html = """
        <div id="siteSub">From Wikipedia</div>
        <p>Article content here.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 3, f"Expected 3 visible words (Article content here), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    def test_excludes_contentSub(self, mock_site):
        """div#contentSub is excluded from visible count."""
        html = """
        <div id="contentSub">Revision info</div>
        <p>Main text.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Main text), got {visible}"
            assert rendered == 4, f"Expected 4 rendered words, got {rendered}"

    def test_excludes_jump_to_nav(self, mock_site):
        """div#jump-to-nav is excluded from visible count."""
        html = """
        <div id="jump-to-nav">Jump to navigation</div>
        <p>Content here.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Content here), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    def test_excludes_localcomments(self, mock_site):
        """span.localcomments is excluded from visible count."""
        html = """
        <p>Statement text.</p>
        <span class="localcomments">Local comment here.</span>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Statement text), got {visible}"
            assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    # Error message tests

    def test_excludes_error_class(self, mock_site):
        """Elements with .error class are excluded from visible count."""
        html = """
        <p>Normal content.</p>
        <span class="error">Template error message here.</span>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Normal content), got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    # Talk page quote template tests ({{tq}}, {{tqb}})

    def test_excludes_inline_quote_talk(self, mock_site):
        """{{tq}} template output (.inline-quote-talk) is excluded from visible count.

        This matches the actual MediaWiki rendering where {{tq|text}} produces
        <q class="inline-quote-talk">text</q>.
        """
        # This mimics the actual HTML output from MediaWiki for {{tq|quoted text}}
        html = """
        <p>Before quote <q class="inline-quote-talk">quoted text here</q> after quote.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 4, f"Expected 4 visible words (Before quote after quote), got {visible}"
            assert rendered == 7, f"Expected 7 rendered words, got {rendered}"

    def test_excludes_talkquote_class(self, mock_site):
        """{{tqb}} template output (.talkquote) is excluded from visible count.

        This matches the actual MediaWiki rendering where {{tqb|text}} produces
        <blockquote class="talkquote">text</blockquote>.
        """
        html = """
        <p>Before block.</p>
        <blockquote class="talkquote"><p>Block quoted text.</p></blockquote>
        <p>After block.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 4, f"Expected 4 visible words (Before/After block), got {visible}"
            assert rendered == 7, f"Expected 7 rendered words, got {rendered}"

    def test_excludes_inline_quote_talk_italic(self, mock_site):
        """{{tqi}} template output (.inline-quote-talk-italic) is handled.

        Note: The current code excludes .inline-quote-talk but not
        .inline-quote-talk-italic specifically. However, if the element
        also has .inline-quote-talk class, it will be excluded.
        """
        html = """
        <p>Normal <q class="inline-quote-talk inline-quote-talk-italic">italic quote</q> text.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Normal text), got {visible}"
            assert rendered == 4, f"Expected 4 rendered words, got {rendered}"

    # Timestamp removal tests

    def test_removes_timestamps(self, mock_site):
        """Wikipedia timestamps are removed from word count."""
        html = "<p>Comment text 12:34, 15 January 2024 (UTC) more text.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Timestamp tokens: "12:34", "15", "January", "2024", "UTC" = 5 tokens
            # After _TS_RE removal, we should have: "Comment text more text" = 4 words
            assert visible == 4, f"Expected 4 visible words (Comment text more text), got {visible}"

    def test_removes_multiple_timestamps(self, mock_site):
        """Multiple timestamps are all removed."""
        html = """
        <p>First 10:00, 1 March 2024 (UTC) second 23:59, 31 December 2024 (UTC) third.</p>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 3, f"Expected 3 visible words (First second third), got {visible}"

    # Edge cases

    def test_empty_wikitext(self, mock_site):
        """Empty wikitext produces zero counts."""
        html = ""
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 0, f"Expected 0 visible words for empty content, got {visible}"
            assert rendered == 0, f"Expected 0 rendered words for empty content, got {rendered}"

    def test_whitespace_only(self, mock_site):
        """Whitespace-only content produces zero counts."""
        html = "   \n\t\n   "
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 0, f"Expected 0 visible words for whitespace-only, got {visible}"
            assert rendered == 0, f"Expected 0 rendered words for whitespace-only, got {rendered}"

    def test_nested_exclusions(self, mock_site):
        """Nested excluded elements are properly handled."""
        html = """
        <p>Outer text.</p>
        <div class="mw-collapsed">
            <p>Collapsed <s>and struck</s> text.</p>
        </div>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            assert visible == 2, f"Expected 2 visible words (Outer text), got {visible}"
            assert rendered == 6, f"Expected 6 rendered words, got {rendered}"

    def test_multiple_exclusion_types(self, mock_site):
        """Multiple different exclusion types in same content."""
        html = """
        <p>Statement with <s>struck text</s> and a reference.</p>
        <ol class="references"><li>Ref content.</li></ol>
        <div style="display:none">Hidden content.</div>
        <q class="inline-quote-talk">Quoted text.</q>
        """
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible: "Statement with and a reference" = 5 words
            assert visible == 5, f"Expected 5 visible words, got {visible}"
            # Rendered: Statement with struck text and a reference Ref content Hidden content Quoted text = 13
            assert rendered == 13, f"Expected 13 rendered words, got {rendered}"

    def test_visible_tokens_filter_non_alphanumeric(self, mock_site):
        """Tokens without alphanumeric characters are filtered from visible count."""
        html = "<p>Real words -- and more.</p>"
        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # "--" should be filtered out, so visible = 4
            assert visible == 4, f"Expected 4 visible words (non-alphanumeric filtered), got {visible}"


class TestCaching:
    """Tests for the caching behavior of word count functions."""

    def test_cache_key_includes_title(self):
        """Cache key incorporates title for context-dependent rendering."""
        import hashlib

        host = "en.wikipedia.org"
        path = "/w/api.php"
        wikitext = "{{tq|test}}"

        key1 = hashlib.sha1(
            f"{host}|{path}|Wikipedia talk:Test|{wikitext}".encode()
        ).hexdigest()
        key2 = hashlib.sha1(
            f"{host}|{path}|Main Page|{wikitext}".encode()
        ).hexdigest()

        assert key1 != key2, "Different titles should produce different cache keys"

    def test_cache_key_without_title(self):
        """Cache key works when title is None."""
        import hashlib

        host = "en.wikipedia.org"
        path = "/w/api.php"
        wikitext = "simple text"

        key = hashlib.sha1(f"{host}|{path}||{wikitext}".encode()).hexdigest()
        assert len(key) == 40  # SHA1 hex digest length

    def test_uses_cached_value(self):
        """Cached values are returned without re-rendering."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = (42, 50)  # Cached (visible, rendered)

        mock_site = MagicMock()
        mock_site.hostname.return_value = "en.wikipedia.org"
        mock_site.apipath.return_value = "/w/api.php"

        with patch.object(bot, "CACHE", mock_cache):
            with patch.object(bot, "_api_render") as mock_render:
                visible, rendered = bot._count_rendered_visible(
                    mock_site, "any wikitext"
                )

                assert visible == 42, f"Expected cached visible value 42, got {visible}"
                assert rendered == 50, f"Expected cached rendered value 50, got {rendered}"
                mock_render.assert_not_called()

    def test_caches_computed_value(self):
        """Computed values are stored in cache."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # Cache miss

        mock_site = MagicMock()
        mock_site.hostname.return_value = "en.wikipedia.org"
        mock_site.apipath.return_value = "/w/api.php"

        html = "<p>Three word sentence.</p>"

        with patch.object(bot, "CACHE", mock_cache):
            with patch.object(bot, "_api_render", return_value=html):
                bot._count_rendered_visible(mock_site, "test wikitext")

                # Verify put was called with the computed values
                mock_cache.put.assert_called_once()
                call_args = mock_cache.put.call_args
                assert call_args[0][1] == (3, 3)  # (visible, rendered)


class TestHelperFunctions:
    """Tests for the visible_word_count and rendered_word_count helpers."""

    def test_visible_word_count_returns_visible_only(self):
        """visible_word_count returns only the visible count."""
        mock_site = MagicMock()
        mock_site.hostname.return_value = "en.wikipedia.org"
        mock_site.apipath.return_value = "/w/api.php"

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        html = "<p>Visible text <s>struck</s> here.</p>"

        with patch.object(bot, "CACHE", mock_cache):
            with patch.object(bot, "_api_render", return_value=html):
                result = bot.visible_word_count(mock_site, "test")
                assert result == 3  # "Visible text here"

    def test_rendered_word_count_returns_rendered_only(self):
        """rendered_word_count returns only the rendered count."""
        mock_site = MagicMock()
        mock_site.hostname.return_value = "en.wikipedia.org"
        mock_site.apipath.return_value = "/w/api.php"

        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        html = "<p>Visible text <s>struck</s> here.</p>"

        with patch.object(bot, "CACHE", mock_cache):
            with patch.object(bot, "_api_render", return_value=html):
                result = bot.rendered_word_count(mock_site, "test")
                assert result == 4  # All words including struck


class TestRealisticWikipediaHTML:
    """Tests using realistic HTML structures from actual Wikipedia rendering."""

    @pytest.fixture(autouse=True)
    def setup_cache(self):
        """Set up a mock cache for each test."""
        mock_cache = MagicMock()
        mock_cache.get.return_value = None
        mock_cache.get_rendered_html.return_value = None
        with patch.object(bot, "CACHE", mock_cache):
            yield

    @pytest.fixture
    def mock_site(self):
        """Create a mock pywikibot site."""
        site = MagicMock()
        site.hostname.return_value = "en.wikipedia.org"
        site.apipath.return_value = "/w/api.php"
        return site

    def test_real_tq_template_html(self, mock_site):
        """Test with actual HTML output from {{tq}} template."""
        # This is the actual structure MediaWiki produces for {{tq|text}}
        # Note: <style> tag content is included in rendered count (CSS class names, etc.)
        html = """<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">
        <p>This is a test. <style data-mw-deduplicate="TemplateStyles:r1248585704">
        .mw-parser-output .inline-quote-talk{font-family:Georgia,serif}</style>
        <q class="inline-quote-talk">This is quoted text that should be excluded.</q>
        And this is after.</p></div>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible: "This is a test And this is after" = 8 words
            # (quoted text excluded, CSS in style tag excluded since it has no alphanumeric-only tokens)
            assert visible == 8, f"Expected 8 visible words (quoted text excluded), got {visible}"
            # Rendered includes everything including CSS content tokens
            assert rendered == 21, f"Expected 21 rendered words (all content), got {rendered}"

    def test_real_tqb_template_html(self, mock_site):
        """Test with actual HTML output from {{tqb}} template."""
        html = """<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">
        <p>Before quote.</p>
        <blockquote class="talkquote"><p>This is a block quote.</p></blockquote>
        <p>After quote.</p></div>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible: "Before quote After quote" = 4 words
            assert visible == 4, f"Expected 4 visible words (block quote excluded), got {visible}"
            # Rendered: all 9 words
            assert rendered == 9, f"Expected 9 rendered words, got {rendered}"

    def test_real_reflist_html(self, mock_site):
        """Test with actual HTML output from {{reflist}} template."""
        html = """<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">
        <p>Main text with a reference.<sup class="reference"><a href="#cite_note-1">[1]</a></sup></p>
        <div class="mw-references-wrap"><ol class="references">
        <li id="cite_note-1"><span class="reference-text">Reference content here.</span></li>
        </ol></div></div>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible: "Main text with a reference" = 5 words (reference excluded)
            # Note: [1] inside sup.reference is kept but filtered as non-word
            assert visible == 5, f"Expected 5 visible words (reference excluded), got {visible}"
            # Rendered: Main text with a reference 1 Reference content here = 9
            assert rendered == 9, f"Expected 9 rendered words, got {rendered}"

    def test_real_collapse_template_html(self, mock_site):
        """Test with actual HTML output from {{collapse}} template."""
        html = """<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">
        <p>Normal text here.</p>
        <table class="mw-collapsible mw-collapsed">
        <tbody><tr><th>Extended content</th></tr>
        <tr><td><div>Collapsed content here.</div></td></tr></tbody></table>
        <p>More normal text.</p></div>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible: "Normal text here More normal text" = 6 words
            assert visible == 6, f"Expected 6 visible words (collapsed excluded), got {visible}"

    def test_complex_statement_with_multiple_exclusions(self, mock_site):
        """Test a realistic statement combining multiple exclusion types."""
        html = """<div class="mw-content-ltr mw-parser-output" lang="en" dir="ltr">
        <p>I am making a statement about policy. The template says
        <q class="inline-quote-talk">any admin may remove it</q> which is
        relevant.<sup class="reference"><a href="#cite_note-1">[1]</a></sup>
        <s>I previously said something wrong.</s></p>
        <div class="mw-references-wrap"><ol class="references">
        <li id="cite_note-1"><span class="reference-text">Policy page reference.</span></li>
        </ol></div></div>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # Visible excludes: quoted text, struck text, references
            # "I am making a statement about policy The template says which is relevant [1]"
            # The visible token filter will include "relevant.[1]" as one token with alphanumeric
            # = 13 tokens (including the ref marker attached to "relevant")
            assert visible == 13, f"Expected 13 visible words (multiple exclusions), got {visible}"

    def test_statement_with_timestamp_signature(self, mock_site):
        """Test that timestamps in signatures are properly removed."""
        html = """<p>I support this proposal for the following reasons.
        <a href="/wiki/User:Example">Example</a>
        (<a href="/wiki/User_talk:Example">talk</a>)
        14:30, 25 January 2024 (UTC)</p>"""

        with patch.object(bot, "_api_render", return_value=html):
            visible, rendered = bot._count_rendered_visible(mock_site, "ignored")
            # After timestamp removal: "I support this proposal for the following reasons Example talk"
            # Note: "14:30", "25", "January", "2024", "(UTC)" are part of timestamp pattern
            assert visible == 10, f"Expected 10 visible words (timestamp removed), got {visible}"
