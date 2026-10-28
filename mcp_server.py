"""mcp_server.py - the news tools, offered over MCP (like a shop with 8 tools on the shelf).

The app starts this file by itself when it needs news (stdio transport), so you rarely run it.
To try it by hand:   uv run python main.py check-tools

IMPORTANT: over stdio, stdout is the conversation channel. Never print() in this file.

Layout of this file:  feeds -> small helpers -> functions that fetch -> the 8 MCP tools at the bottom.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import sys
from datetime import UTC, datetime

import feedparser
import httpx
import trafilatura
from mcp.server.fastmcp import FastMCP

from core import is_safe_url

logging.basicConfig(stream=sys.stderr, level=logging.WARNING)
log = logging.getLogger("daily_agent.tools")

# RSS feeds: (source name, address). A dead feed is skipped, so one bad link never breaks the paper.
# If a source stays silent, fix its address here. Names must match SOURCE_TRUST in core.py.
FEEDS = {
    "world": [
        ("BBC News", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
        ("NPR", "https://feeds.npr.org/1004/rss.xml"),
        ("The Hindu", "https://www.thehindu.com/news/international/feeder/default.rss"),
    ],
    "science": [
        ("ScienceDaily", "https://www.sciencedaily.com/rss/top/science.xml"),
        ("Phys.org", "https://phys.org/rss-feed/"),
        ("NASA", "https://www.nasa.gov/feed/"),
        ("Science News", "https://www.sciencenews.org/feed"),
    ],
    "wildlife": [
        ("Mongabay", "https://news.mongabay.com/feed/"),
        ("ScienceDaily", "https://www.sciencedaily.com/rss/plants_animals.xml"),
    ],
    "ai": [
        ("OpenAI", "https://openai.com/news/rss.xml"),
        ("Google DeepMind", "https://deepmind.google/blog/rss.xml"),
        ("Hugging Face", "https://huggingface.co/blog/feed.xml"),
        ("MIT Technology Review", "https://www.technologyreview.com/topic/artificial-intelligence/feed"),
        ("The Verge", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ],
}

AI_WORDS = re.compile(
    r"\b(ai|a\.i\.|llm|llms|gpt|openai|anthropic|claude|gemini|llama|mistral|deepmind|"
    r"neural|diffusion|machine learning|language model)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------- small helpers
def _http() -> httpx.AsyncClient:
    """One place that creates the web client (the tests swap it for a fake one)."""
    return httpx.AsyncClient(headers={"User-Agent": "daily-agent/0.1 (learning project)"}, timeout=15, follow_redirects=True)


def _plain(text: str, limit: int = 500) -> str:
    """Remove HTML tags and extra spaces, then cut to a maximum length."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text or ""))
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _news_item(title, url, source, category, published=None, snippet="", image_url=None) -> dict:
    """Every tool returns items in this one shape (it matches the Article class in core.py)."""
    return {
        "id": hashlib.sha1(url.encode("utf-8")).hexdigest()[:10],  # a short unique id made from the url
        "title": title, "url": url, "source": source, "category": category,
        "published": published, "snippet": snippet, "image_url": image_url,
    }


def _iso(time_tuple) -> str | None:
    """feedparser gives dates as tuples; turn them into an ISO text like 2026-10-27T08:00:00+00:00."""
    if not time_tuple:
        return None
    try:
        return datetime(*time_tuple[:6], tzinfo=UTC).isoformat()
    except (TypeError, ValueError):
        return None


def _find_image(entry) -> str | None:
    """Look for a picture inside an RSS entry: media tags, then attachments, then an <img> in the text."""
    for key in ("media_content", "media_thumbnail"):
        for media in entry.get(key) or []:
            if media.get("url"):
                return media["url"]
    for link in list(entry.get("links") or []) + list(entry.get("enclosures") or []):
        if str(link.get("type", "")).startswith("image") and link.get("href"):
            return link["href"]
    match = re.search(r'<img[^>]+src="([^"]+)"', entry.get("summary", "") or "")
    return match.group(1) if match else None


# ---------------------------------------------------------------- functions that fetch
async def _read_feed(feed_url: str, source: str, category: str, limit: int = 6) -> list[dict]:
    """Download one RSS feed and return its newest entries (an empty list if anything goes wrong)."""
    if not is_safe_url(feed_url):
        return []
    try:
        async with _http() as client:
            response = await client.get(feed_url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        log.warning("feed %s failed: %s", source, error)
        return []

    feed = await asyncio.to_thread(feedparser.parse, response.content)
    items = []
    for entry in feed.entries[:limit]:
        url, title = entry.get("link"), _plain(entry.get("title", ""), 200)
        if url and title:
            items.append(
                _news_item(
                    title, url, source, category,
                    published=_iso(entry.get("published_parsed") or entry.get("updated_parsed")),
                    snippet=_plain(entry.get("summary", "")),
                    image_url=_find_image(entry),
                )
            )
    return items


async def _read_group(group: str, category: str, limit: int) -> list[dict]:
    """Read every feed of one group (e.g. 'world') at the same time and join the results."""
    results = await asyncio.gather(*(_read_feed(url, name, category, limit) for name, url in FEEDS[group]))
    return [item for result in results for item in result]


async def _hacker_news(n: int) -> list[dict]:
    """Top Hacker News stories whose title mentions AI."""
    base = "https://hacker-news.firebaseio.com/v0"
    async with _http() as client:
        try:
            ids = (await client.get(f"{base}/topstories.json")).json()[:60]
        except (httpx.HTTPError, ValueError) as error:
            log.warning("hacker news failed: %s", error)
            return []

        async def get_story(item_id):
            try:
                return (await client.get(f"{base}/item/{item_id}.json")).json()
            except (httpx.HTTPError, ValueError):
                return None

        stories = await asyncio.gather(*(get_story(i) for i in ids))

    items = []
    for story in stories:
        if not story or story.get("type") != "story" or not AI_WORDS.search(story.get("title", "")):
            continue
        url = story.get("url") or f"https://news.ycombinator.com/item?id={story['id']}"
        published = datetime.fromtimestamp(story.get("time", 0), tz=UTC).isoformat()
        items.append(_news_item(_plain(story["title"], 200), url, "Hacker News", "ai",
                                published=published, snippet=f"{story.get('score', 0)} points on Hacker News"))
        if len(items) >= n:
            break
    return items


async def _arxiv(category: str, n: int) -> list[dict]:
    """Newest arXiv papers in a category such as cs.AI."""
    if not re.fullmatch(r"[A-Za-z-]+(\.[A-Za-z-]+)?", category):  # only letters, dots and dashes allowed
        return []
    url = f"https://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={n}"
    try:
        async with _http() as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        log.warning("arxiv failed: %s", error)
        return []
    feed = await asyncio.to_thread(feedparser.parse, response.content)
    return [
        _news_item(_plain(e.get("title", ""), 200), e.link, "arXiv", "ai",
                   published=_iso(e.get("published_parsed")), snippet=_plain(e.get("summary", "")))
        for e in feed.entries[:n] if e.get("link")
    ]


async def _hf_models(n: int) -> list[dict]:
    """Models that are trending on the Hugging Face hub."""
    try:
        async with _http() as client:
            response = await client.get(f"https://huggingface.co/api/models?sort=trendingScore&limit={n}")
            response.raise_for_status()
            models = response.json()
    except (httpx.HTTPError, ValueError) as error:
        log.warning("hugging face failed: %s", error)
        return []
    now = datetime.now(UTC).isoformat()
    return [
        _news_item(f"Trending on Hugging Face: {m['id']}", f"https://huggingface.co/{m['id']}", "Hugging Face", "ai",
                   published=now, snippet=f"{m.get('pipeline_tag') or 'model'}, {m.get('likes', 0)} likes, {m.get('downloads', 0)} downloads")
        for m in models if m.get("id")
    ]


async def _extract(url: str) -> dict:
    """Download a web page; return its clean article text and main picture."""
    if not is_safe_url(url):
        return {"text": "", "image_url": None, "error": "blocked url"}
    try:
        async with _http() as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError as error:
        return {"text": "", "image_url": None, "error": str(error)}
    if len(response.content) > 3_000_000:
        return {"text": "", "image_url": None, "error": "page too large"}

    def work() -> dict:
        page = response.text
        text = trafilatura.extract(page, include_comments=False, include_tables=False) or ""
        meta = trafilatura.extract_metadata(page)
        return {"text": text[:20000], "image_url": getattr(meta, "image", None) if meta else None}

    return await asyncio.to_thread(work)


# ---------------------------------------------------------------- the MCP server: 8 tools on the shelf
# Each tool returns its result as one JSON string, which keeps the client side simple.
mcp = FastMCP("news-tools")


@mcp.tool()
async def get_world_news(limit_per_feed: int = 6) -> str:
    """Top world headlines from BBC, Al Jazeera, NPR and The Hindu."""
    return json.dumps(await _read_group("world", "world", limit_per_feed))


@mcp.tool()
async def get_science_news(limit_per_feed: int = 5) -> str:
    """Recent scientific discoveries (ScienceDaily, Phys.org, NASA, Science News)."""
    return json.dumps(await _read_group("science", "science", limit_per_feed))


@mcp.tool()
async def get_wildlife_news(limit_per_feed: int = 5) -> str:
    """Wildlife and conservation news (Mongabay, ScienceDaily plants and animals)."""
    return json.dumps(await _read_group("wildlife", "science", limit_per_feed))


@mcp.tool()
async def get_ai_news(limit_per_feed: int = 5) -> str:
    """Latest posts from AI labs and AI news sites (OpenAI, DeepMind, Hugging Face, ...)."""
    return json.dumps(await _read_group("ai", "ai", limit_per_feed))


@mcp.tool()
async def get_hn_top(n: int = 8) -> str:
    """Top Hacker News stories about AI."""
    return json.dumps(await _hacker_news(n))


@mcp.tool()
async def get_arxiv_latest(category: str = "cs.AI", n: int = 5) -> str:
    """Newest arXiv papers in a category such as cs.AI, cs.LG or cs.CL."""
    return json.dumps(await _arxiv(category, n))


@mcp.tool()
async def get_hf_trending_models(n: int = 5) -> str:
    """Models trending on the Hugging Face hub."""
    return json.dumps(await _hf_models(n))


@mcp.tool()
async def extract_article(url: str) -> str:
    """Download a web page and return {text, image_url} with the clean article text."""
    return json.dumps(await _extract(url))


if __name__ == "__main__":
    mcp.run(transport="stdio")
