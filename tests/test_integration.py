"""Integration tests that use real Wikipedia API calls.

These tests verify that:
1. Templates render with expected CSS classes
2. The word counting logic works correctly with real MediaWiki HTML output
3. Our selectors match actual Wikipedia markup

Mark these tests with @pytest.mark.integration to run separately from unit tests.
Run with: pytest -m integration
"""

import time
from pathlib import Path

import pytest
import pywikibot
from pywikibot.site import APISite

# Import the module under test
import bot

# Delay between API calls to avoid rate limiting (in seconds)
API_DELAY = 0.5

# Load saved page fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> str:
    """Load a wikitext fixture file."""
    return (FIXTURES_DIR / name).read_text()


@pytest.fixture(scope="module")
def site() -> APISite:
    """Create a real pywikibot site connection.

    This fixture is module-scoped to reuse the connection across tests.
    Sets a descriptive user agent to identify test traffic.
    """
    pywikibot.config.user_agent_description = (
        "WordcountClerkBot test suite (https://github.com/L235/WordcountClerkBot)"
    )
    return pywikibot.Site("en", "wikipedia")


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    """Initialize the cache for integration tests.

    Uses a temporary directory for the cache to avoid conflicts and ensure
    cross-platform compatibility. Restores the original cache after tests.
    """
    old_cache = bot.CACHE
    tmp_dir = tmp_path_factory.mktemp("wordcount_cache")
    bot.CACHE = bot.WordCountCache(state_dir=str(tmp_dir))
    yield bot.CACHE
    bot.CACHE = old_cache


@pytest.fixture(autouse=True)
def api_rate_limit():
    """Add a small delay after each test to avoid Wikipedia API rate limiting."""
    yield
    time.sleep(API_DELAY)


@pytest.mark.integration
class TestTemplateRendering:
    """Tests that verify MediaWiki templates render with expected CSS classes."""

    def test_tq_template_produces_inline_quote_talk_class(self, site):
        """Verify {{tq}} template renders with .inline-quote-talk class."""
        result = site.simple_request(
            action="parse",
            text="{{tq|quoted text}}",
            contentmodel="wikitext",
            prop="text",
            title="Wikipedia talk:Test",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert "inline-quote-talk" in html, "{{tq}} should produce .inline-quote-talk class"
        assert "quoted text" in html, "{{tq}} should preserve the quoted content"

    def test_tqb_template_produces_talkquote_class(self, site):
        """Verify {{tqb}} template renders with .talkquote class."""
        result = site.simple_request(
            action="parse",
            text="{{tqb|block quoted text}}",
            contentmodel="wikitext",
            prop="text",
            title="Wikipedia talk:Test",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert "talkquote" in html, "{{tqb}} should produce .talkquote class"
        assert "block quoted text" in html, "{{tqb}} should preserve the quoted content"

    def test_reflist_produces_references_class(self, site):
        """Verify {{reflist}} renders with .references class."""
        result = site.simple_request(
            action="parse",
            text="Text<ref>Reference</ref>\n{{reflist}}",
            contentmodel="wikitext",
            prop="text",
            title="Test page",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert 'class="references"' in html, "{{reflist}} should produce .references class"

    def test_collapse_produces_mw_collapsed_class(self, site):
        """Verify {{collapse}} renders with .mw-collapsed class."""
        result = site.simple_request(
            action="parse",
            text="{{collapse|hidden content}}",
            contentmodel="wikitext",
            prop="text",
            title="Test page",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert "mw-collapsed" in html, "{{collapse}} should produce .mw-collapsed class"

    def test_struck_tags_render_correctly(self, site):
        """Verify <s>, <strike>, and <del> tags render properly."""
        result = site.simple_request(
            action="parse",
            text="<s>struck</s> <strike>strike</strike> <del>deleted</del>",
            contentmodel="wikitext",
            prop="text",
            title="Test page",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert "<s>" in html or "<s " in html, "s tag should be preserved"
        assert "<del>" in html or "<del " in html, "del tag should be preserved"


@pytest.mark.integration
class TestWordCountingWithRealAPI:
    """Tests that verify word counting with real MediaWiki API responses."""

    def test_tq_excluded_from_visible_count(self, site, cache):
        """{{tq}} quoted text should be excluded from visible word count."""
        wikitext = "Normal text {{tq|quoted text}} more normal."

        visible, rendered = bot._count_rendered_visible(
            site, wikitext, title="Wikipedia talk:Test"
        )

        # Visible: "Normal text more normal" = 4 visible words
        # (quoted text excluded via .inline-quote-talk class)
        assert visible == 4, f"Expected 4 visible words, got {visible}"
        # Rendered includes all text plus CSS from <style> tags embedded by templates
        # We just verify the visible count is correct and rendered >= visible
        assert rendered >= visible, "Rendered should be >= visible"

    def test_tqb_excluded_from_visible_count(self, site, cache):
        """{{tqb}} block quoted text should be excluded from visible word count."""
        wikitext = "Before quote.\n{{tqb|Block quote content here.}}\nAfter quote."

        visible, rendered = bot._count_rendered_visible(
            site, wikitext, title="Wikipedia talk:Test"
        )

        # Visible: "Before quote After quote" = 4 visible words
        # (block quoted text excluded via .talkquote class)
        assert visible == 4, f"Expected 4 visible words, got {visible}"
        # Rendered includes all text plus CSS from <style> tags
        assert rendered >= visible, "Rendered should be >= visible"

    def test_references_excluded_from_visible_count(self, site, cache):
        """Reference content should be excluded from visible word count."""
        wikitext = "Main text here.<ref>Reference content excluded.</ref>\n\n{{reflist}}"

        visible, rendered = bot._count_rendered_visible(site, wikitext, title="Test")

        # "Main text here" = 3 visible words
        assert visible == 3, f"Expected 3 visible words, got {visible}"
        # Rendered includes reference content
        assert rendered >= 3, f"Rendered should include reference content, got {rendered}"

    def test_struck_text_excluded_from_visible_count(self, site, cache):
        """Struck-through text should be excluded from visible word count."""
        wikitext = "Keep this <s>remove this</s> text."

        visible, rendered = bot._count_rendered_visible(site, wikitext, title="Test")

        # "Keep this text" = 3 visible words
        assert visible == 3, f"Expected 3 visible words, got {visible}"
        assert rendered == 5, f"Expected 5 rendered words, got {rendered}"

    def test_collapsed_content_excluded_from_visible_count(self, site, cache):
        """Collapsed content should be excluded from visible word count."""
        wikitext = "Before collapse.\n{{collapse|Hidden collapsed content.}}\nAfter collapse."

        visible, rendered = bot._count_rendered_visible(site, wikitext, title="Test")

        # Visible should exclude the collapsed content
        # "Before collapse After collapse" = 4 words
        # But note: {{collapse}} also adds "Extended content" header which may be counted
        assert visible <= 6, f"Expected visible <= 6, got {visible}"
        assert rendered > visible, "Rendered should include more than visible"

    def test_multiple_exclusions_combined(self, site, cache):
        """Multiple exclusion types in one statement should all be applied."""
        wikitext = """I am stating my position. The policy says {{tq|admins may act}} which supports this.<ref>Policy reference.</ref>
<s>I previously said something incorrect.</s>

{{reflist}}"""

        visible, rendered = bot._count_rendered_visible(
            site, wikitext, title="Wikipedia talk:Test"
        )

        # Visible should exclude: {{tq}} content, reference, struck text
        # "I am stating my position The policy says which supports this"
        # = 11 words approximately
        assert visible < rendered, "Visible should be less than rendered"
        assert visible >= 8, f"Expected visible >= 8, got {visible}"
        assert visible <= 13, f"Expected visible <= 13, got {visible}"


@pytest.mark.integration
class TestTitleContextMatters:
    """Tests that verify title context affects template rendering."""

    def test_tq_renders_correctly_on_talk_page(self, site, cache):
        """{{tq}} should render correctly when given a talk page title."""
        wikitext = "{{tq|test quote}}"

        # Use pywikibot's simple_request to check raw HTML
        result = site.simple_request(
            action="parse",
            text=wikitext,
            contentmodel="wikitext",
            prop="text",
            title="Wikipedia talk:Sandbox",
        ).submit()

        html = result["parse"]["text"]["*"]

        # Should have the proper class, not an error
        assert "inline-quote-talk" in html, "{{tq}} on talk page should produce .inline-quote-talk class"
        assert "error" not in html.lower() or "class=\"error\"" not in html, "{{tq}} on talk page should not produce an error"

    def test_tq_on_article_page_produces_error(self, site, cache):
        """{{tq}} on an article page produces an error message instead of quoted text.

        Note: This test documents the behavior that led to issue #32.
        The {{tq}} template is designed for talk/project pages only.
        When used on article pages, it shows an error message.
        This is why we pass the title parameter to _count_rendered_visible.
        """
        wikitext = "{{tq|test quote}}"

        # Render with article namespace title (not talk/project page)
        result = site.simple_request(
            action="parse",
            text=wikitext,
            contentmodel="wikitext",
            prop="text",
            title="Main Page",  # Article namespace, not talk
        ).submit()

        html = result["parse"]["text"]["*"]

        # On article pages, {{tq}} produces an error message
        assert "error" in html.lower(), "{{tq}} on article pages should show an error"
        # The quoted text is NOT rendered - instead we get the error message
        assert "only for quoting in talk and project pages" in html, "{{tq}} error message should explain it's for talk/project pages only"


@pytest.mark.integration
class TestAEPageFixture:
    """Tests using a saved snapshot of the AE page (captured 2026-01-31).

    These tests use a complete snapshot of the Wikipedia:Arbitration/Requests/Enforcement
    page to verify word counting works correctly with real page content, without
    depending on the live page having any specific content.
    """

    @pytest.fixture(scope="class")
    def ae_page_wikitext(self):
        """Load the AE page snapshot fixture."""
        return load_fixture("ae_page_snapshot.txt")

    def test_ae_fixture_has_content(self, ae_page_wikitext):
        """Verify the AE fixture loaded correctly."""
        assert len(ae_page_wikitext) > 10000, "AE fixture should have substantial content"

    def test_ae_page_word_count(self, site, cache, ae_page_wikitext):
        """Test word counting on the full AE page snapshot."""
        visible, rendered = bot._count_rendered_visible(
            site, ae_page_wikitext, title="Wikipedia:Arbitration/Requests/Enforcement"
        )

        # Sanity checks for a full page
        assert visible > 100, f"Expected visible > 100, got {visible}"
        assert rendered > visible, "Rendered should be >= visible"
        # Visible should be less due to exclusions (references, collapsed, etc.)
        assert visible < rendered, "Visible should exclude some content"

    def test_ae_page_renders_without_error(self, site, ae_page_wikitext):
        """Verify the AE page fixture renders successfully via the API."""
        result = site.simple_request(
            action="parse",
            text=ae_page_wikitext,
            contentmodel="wikitext",
            prop="text",
            title="Wikipedia:Arbitration/Requests/Enforcement",
        ).submit()

        html = result["parse"]["text"]["*"]
        assert len(html) > 1000, "Rendered HTML should have content"
        # Should not have parser errors
        assert "mw-parser-output" in html, "Rendered HTML should have mw-parser-output wrapper"

    def test_ae_page_tq_exclusion(self, site, cache, ae_page_wikitext):
        """Verify {{tq}} templates in AE page are excluded from visible count.

        The AE page snapshot contains {{tq}} templates. This test verifies
        that the .inline-quote-talk class is present in rendered output,
        confirming the exclusion logic will apply.
        """
        result = site.simple_request(
            action="parse",
            text=ae_page_wikitext,
            contentmodel="wikitext",
            prop="text",
            title="Wikipedia:Arbitration/Requests/Enforcement",
        ).submit()

        html = result["parse"]["text"]["*"]
        # The snapshot contains {{tq}} which should render with this class
        assert "inline-quote-talk" in html, "AE page should contain {{tq}} content"
