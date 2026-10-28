"""pipeline.py - the conductor. It runs the workers from agents.py one after another.

    fetch -> guard -> rank -> enrich -> summarize -> validate -> layout -> publish

The "conductor" is plain code (a LangGraph graph), not an LLM. Each step is a function that
receives the shared `state` (a dict) and returns only the new things it adds to it.
Workers never talk to each other: they only read from and write to the state.

Also in this file: the HTML renderer and the saving/loading of editions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import TypedDict

from jinja2 import Environment, FileSystemLoader, select_autoescape
from langgraph.graph import END, START, StateGraph
from langsmith import traceable
from pydantic import ValidationError

from agents import (
    SummaryError,
    build_layout,
    extract_texts,
    fetch_candidates,
    find_earlier,
    load_seen,
    parse_dt,
    rank,
    save_seen,
    summarize_one,
    validate_and_repair,
)
from core import ROOT, Article, Edition, LLMError, Story, clean_text, is_safe_url, metrics, settings

log = logging.getLogger("daily_agent")
MAX_AGE_DAYS = 5  # ignore news older than this
LLM_CONCURRENCY = 3  # free LLM tiers have rate limits, so summarize only a few stories at a time


class State(TypedDict, total=False):
    """The shared notebook that every step reads from and writes to."""

    demo: bool
    use_llm: bool
    candidates: list[dict]
    articles: list[Article]
    selected: dict[str, list[Article]]
    texts: dict[str, str]
    drafts: list[dict]
    stories: list[Story]
    edition: Edition


# ---------------------------------------------------------------- the steps
def load_sample() -> list[dict]:
    """Demo mode: fictional sample articles from sample_data.json (works offline)."""
    data = json.loads((ROOT / "sample_data.json").read_text(encoding="utf-8"))
    now = datetime.now(UTC)
    for item in data["candidates"]:
        item["published"] = (now - timedelta(hours=item.pop("hours_ago", 3))).isoformat()
    return data["candidates"]


async def fetch_step(state: State) -> dict:
    """Get raw news. Live mode asks the MCP server; demo mode reads the sample file."""
    return {"candidates": load_sample() if state["demo"] else await fetch_candidates()}


async def guard_step(state: State) -> dict:
    """INPUT guardrail: check the shape, block unsafe links, drop duplicates and old news, clean the text."""
    cutoff = datetime.now(UTC) - timedelta(days=MAX_AGE_DAYS)
    seen_urls = set()
    articles = []
    for raw in state["candidates"]:
        try:
            article = Article.model_validate(raw)  # SCHEMA guardrail: wrong shape -> skipped
        except ValidationError:
            metrics.schema_failures += 1
            continue
        published = parse_dt(article.published)
        if article.url in seen_urls or not is_safe_url(article.url) or (published and published < cutoff):
            continue
        seen_urls.add(article.url)
        article.title = clean_text(article.title, 200)
        article.snippet = clean_text(article.snippet, 500)
        article.text = clean_text(article.text)
        if article.image_url and not is_safe_url(article.image_url):
            article.image_url = None
        if len(article.title) >= 5:
            articles.append(article)
    if not articles:
        raise RuntimeError("No usable news was fetched (check your internet connection and the feed URLs in mcp_server.py).")
    return {"articles": articles}


async def rank_step(state: State) -> dict:
    return {"selected": await asyncio.to_thread(rank, state["articles"], state["use_llm"])}


async def enrich_step(state: State) -> dict:
    """Download the full text of the chosen stories (the MCP tool extract_article) and grab a picture."""
    chosen = [a for group in state["selected"].values() for a in group]
    extracted = await extract_texts(chosen, demo=state["demo"])
    texts, selected = {}, {}
    for category, group in state["selected"].items():
        selected[category] = []
        for article in group:
            text, image = extracted.get(article.id, (article.snippet, None))
            texts[article.id] = text
            image = article.image_url or (image if image and is_safe_url(image) else None)
            selected[category].append(article.model_copy(update={"image_url": image}))
    return {"texts": texts, "selected": selected}


async def summarize_step(state: State) -> dict:
    """The Summarizer writes a first draft for each chosen story (a few at a time)."""
    seen = load_seen()
    gate = asyncio.Semaphore(LLM_CONCURRENCY)

    async def draft(article: Article) -> dict:
        text = state["texts"][article.id]
        earlier = find_earlier(article.title, seen)
        story = None
        if state["use_llm"]:
            async with gate:
                try:
                    story = await asyncio.to_thread(summarize_one, article, text, None, earlier)
                except (SummaryError, LLMError):
                    story = None  # the validator will deal with it
        return {"article": article, "text": text, "story": story, "earlier": earlier}

    chosen = [a for group in state["selected"].values() for a in group]
    return {"drafts": list(await asyncio.gather(*(draft(a) for a in chosen)))}


async def validate_step(state: State) -> dict:
    return {"stories": await asyncio.to_thread(validate_and_repair, state["drafts"], state["use_llm"])}


async def layout_step(state: State) -> dict:
    """Pick the lead story and tagline, and pack everything into one Edition."""
    stories = state["stories"]
    big = [s for s in stories if s.category in ("world", "science")]
    layout = await asyncio.to_thread(build_layout, big, state["use_llm"])
    edition = Edition(
        date=settings.today() + ("-demo" if state["demo"] else ""),
        demo=state["demo"],
        tagline=layout.tagline,
        lead_id=layout.lead_id,
        world=[s for s in stories if s.category == "world"],
        science=[s for s in stories if s.category == "science"],
        ai=[s for s in stories if s.category == "ai"],
    )
    return {"edition": edition}


async def publish_step(state: State) -> dict:
    """Draw the HTML page, save it, and remember today's stories for tomorrow."""
    edition = state["edition"]
    edition.metrics = metrics.to_dict()
    save_edition(edition, render_edition(edition))
    if not state["demo"]:
        save_seen(edition)
    return {"edition": edition}


STEPS = [
    ("fetch", fetch_step), ("guard", guard_step), ("rank", rank_step), ("enrich", enrich_step),
    ("summarize", summarize_step), ("validate", validate_step), ("layout", layout_step), ("publish", publish_step),
]


def timed(name: str, step):
    """Wrap a step so its running time is recorded in the metrics."""

    async def wrapper(state: State) -> dict:
        start = time.perf_counter()
        try:
            return await step(state)
        finally:
            metrics.step_seconds[name] = round(time.perf_counter() - start, 3)

    return wrapper


@cache
def build_graph():
    """Connect the steps in a straight line: START -> fetch -> guard -> ... -> publish -> END."""
    graph = StateGraph(State)
    for name, step in STEPS:
        graph.add_node(name, timed(name, step))
    names = [name for name, _ in STEPS]
    graph.add_edge(START, names[0])
    for current, following in zip(names, names[1:], strict=False):
        graph.add_edge(current, following)
    graph.add_edge(names[-1], END)
    return graph.compile()


@traceable(name="daily_agent_run")
async def run_pipeline(demo: bool = False, use_llm: bool | None = None) -> Edition:
    """Make one edition. demo=True uses the sample data and, by default, no LLM (so no keys needed)."""
    metrics.reset()
    start = time.perf_counter()
    use_llm = (not demo) if use_llm is None else use_llm
    result = await build_graph().ainvoke({"demo": demo, "use_llm": use_llm})
    edition = result["edition"]
    # The total time is only known now, so update the saved files with the final numbers.
    metrics.total_seconds = round(time.perf_counter() - start, 3)
    edition.metrics = metrics.to_dict()
    save_edition(edition, render_edition(edition))
    return edition


# ---------------------------------------------------------------- the page (Jinja2)
# Jinja2 fills templates/newspaper.html with the edition. No LLM is involved, so the page cannot break,
# and autoescape means odd characters in a headline cannot inject HTML.
_env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["html"]))


def render_edition(edition: Edition) -> str:
    big = edition.world + edition.science  # the three big stories
    lead = next((s for s in big if s.id == edition.lead_id), big[0] if big else None)
    others = [s for s in big if s is not lead]
    ai_hero = next((s for s in edition.ai if s.image_url), edition.ai[0] if edition.ai else None)  # biggest AI picture
    ai_rest = [s for s in edition.ai if s is not ai_hero]
    try:
        pretty_date = datetime.strptime(edition.date[:10], "%Y-%m-%d").strftime("%A, %d %B %Y")
    except ValueError:
        pretty_date = edition.date
    return _env.get_template("newspaper.html").render(
        edition=edition, pretty_date=pretty_date, lead=lead, others=others, ai_hero=ai_hero, ai_rest=ai_rest,
        labels={"world": "World", "science": "Science & Nature", "ai": "AI & Tech"},
    )


# ---------------------------------------------------------------- saving and loading (files; S3 optional)
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(-demo)?$")  # only accept dates, so nobody can ask for ../../secret


def save_edition(edition: Edition, html: str) -> None:
    settings.editions_dir.mkdir(parents=True, exist_ok=True)
    (settings.editions_dir / f"{edition.date}.html").write_text(html, encoding="utf-8")
    (settings.editions_dir / f"{edition.date}.json").write_text(edition.model_dump_json(indent=2), encoding="utf-8")
    if settings.s3_bucket and not edition.demo:
        try:
            import boto3  # only needed on AWS

            s3 = boto3.client("s3", region_name=settings.aws_region)
            s3.put_object(Bucket=settings.s3_bucket, Key=f"editions/{edition.date}.html", Body=html.encode("utf-8"), ContentType="text/html; charset=utf-8")
            s3.put_object(Bucket=settings.s3_bucket, Key=f"editions/{edition.date}.json", Body=edition.model_dump_json().encode("utf-8"), ContentType="application/json")
        except Exception as error:  # a failed upload must not stop the newspaper
            log.warning("S3 upload failed: %s", error)


def load_html(date: str) -> str | None:
    path = settings.editions_dir / f"{date}.html"
    return path.read_text(encoding="utf-8") if DATE_RE.match(date) and path.exists() else None


def load_edition(date: str) -> Edition | None:
    path = settings.editions_dir / f"{date}.json"
    if not DATE_RE.match(date) or not path.exists():
        return None
    return Edition.model_validate_json(path.read_text(encoding="utf-8"))


def list_dates() -> list[str]:
    if not settings.editions_dir.exists():
        return []
    return sorted((p.stem for p in settings.editions_dir.glob("*.html")), reverse=True)
