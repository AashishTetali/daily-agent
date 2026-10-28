"""core.py - the foundation. Every other file imports from here.

  1. Settings      - every knob, read from the .env file
  2. Data shapes   - what an Article / Story / Edition looks like (Pydantic)
  3. Metrics       - counters and timers for the report card
  4. Guardrails    - safety checks for input (web text) and output (LLM summaries)
  5. LLM gateway   - one chat() function: try the main model, then the backup model
  6. Skills        - writing rules loaded from skills.md and added to prompts
"""

from __future__ import annotations

import html
import ipaddress
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")  # copies the values in .env into the environment
log = logging.getLogger("daily_agent")


# =====================================================================
# 1. SETTINGS
# =====================================================================
@dataclass
class Settings:
    # LLM names in LiteLLM format "provider/model". The backup is used if the main one fails.
    primary_model: str = os.getenv("PRIMARY_MODEL", "groq/openai/gpt-oss-20b")
    fallback_model: str = os.getenv("FALLBACK_MODEL", "gemini/gemini-2.5-flash")

    # How many stories go into each part of the newspaper.
    n_world: int = int(os.getenv("N_WORLD", "2"))
    n_science: int = int(os.getenv("N_SCIENCE", "1"))
    n_ai: int = int(os.getenv("N_AI", "5"))

    # Limits used by the guardrails.
    max_article_chars: int = 6000
    summary_min_chars: int = 40
    summary_max_chars: int = 600

    timezone: str = os.getenv("TIMEZONE", "Asia/Kolkata")
    generate_token: str = os.getenv("GENERATE_TOKEN", "")  # if set, POST /generate needs header X-Token
    s3_bucket: str = os.getenv("S3_BUCKET", "")  # if set, editions are also uploaded to S3
    aws_region: str = os.getenv("AWS_REGION", "ap-south-1")
    data_dir: Path = ROOT / "data"

    @property
    def editions_dir(self) -> Path:
        return self.data_dir / "editions"

    @property
    def seen_path(self) -> Path:
        return self.data_dir / "seen_stories.json"

    def today(self) -> str:
        return datetime.now(ZoneInfo(self.timezone)).strftime("%Y-%m-%d")


settings = Settings()

# How much we trust each news source (0 to 1). Used when ranking. Unknown sources get 0.5.
SOURCE_TRUST = {
    "BBC News": 1.0, "NASA": 1.0, "Al Jazeera": 0.5, "NPR": 0.9, "The Hindu": 0.9,
    "Science News": 0.9, "MIT Technology Review": 0.9, "Google DeepMind": 0.9, "OpenAI": 0.9,
    "ScienceDaily": 0.8, "Phys.org": 0.8, "Mongabay": 0.8, "Hugging Face": 0.8,
    "The Verge": 0.7, "Hacker News": 0.6, "arXiv": 0.6,
}


# =====================================================================
# 2. DATA SHAPES (Pydantic = a form with rules; wrong data is rejected)
# =====================================================================
Category = Literal["world", "science", "ai"]


class Article(BaseModel):
    """A raw news item from the MCP tools, before summarizing."""

    id: str
    title: str
    url: str
    source: str
    category: Category
    published: str | None = None  # ISO timestamp
    snippet: str = ""
    image_url: str | None = None
    text: str = ""  # full article text, filled in later
    cluster_size: int = 1  # how many different sources covered this story
    score: float = 0.0  # ranking score


class Story(BaseModel):
    """A finished, printable story. The LLM's answer must fit this shape."""

    id: str
    category: Category
    headline: str = Field(min_length=5, max_length=140)
    summary: str = Field(min_length=30, max_length=700)
    why_it_matters: str = Field(default="", max_length=200)
    source_name: str
    source_url: str = Field(pattern=r"^https?://")
    image_url: str | None = None
    is_update: bool = False  # True when it follows an earlier story we printed
    update_of: str | None = None


class Layout(BaseModel):
    """The layout agent's decisions."""

    tagline: str = Field(min_length=3, max_length=100)
    lead_id: str


class Edition(BaseModel):
    """One full newspaper."""

    date: str
    demo: bool = False  # True when built from the sample data (not real news)
    tagline: str
    lead_id: str | None = None
    world: list[Story] = []
    science: list[Story] = []
    ai: list[Story] = []
    metrics: dict = {}


# =====================================================================
# 3. METRICS (the numbers for your report card)
# =====================================================================
@dataclass
class RunMetrics:
    # LLM gateway
    llm_calls: int = 0
    llm_failures: int = 0
    fallback_count: int = 0  # times the backup model had to answer
    models_used: dict = field(default_factory=dict)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    # MCP tools
    tool_calls: int = 0
    tool_failures: int = 0
    # Guardrails
    inputs_cleaned: int = 0
    injections_removed: int = 0
    unsafe_urls_blocked: int = 0
    schema_failures: int = 0
    output_rejections: int = 0
    retries: int = 0
    extractive_fallbacks: int = 0
    # Performance
    step_seconds: dict = field(default_factory=dict)
    total_seconds: float = 0.0

    def reset(self) -> None:
        """Set every counter back to zero (other files keep using the same object)."""
        self.__dict__.update(RunMetrics().__dict__)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["cost_usd"] = round(data["cost_usd"], 6)
        return data


metrics = RunMetrics()  # the one shared copy; other files do e.g.  metrics.retries += 1


# =====================================================================
# 4. GUARDRAILS
#    Input : clean_text(), is_safe_url()  -> make web text safe to give to an LLM
#    Schema: the Pydantic classes above   -> reject wrong shapes
#    Output: check_summary()              -> check what the LLM wrote before we print it
# =====================================================================
def is_safe_url(url: str) -> bool:
    """Allow only public http(s) links. Blocks localhost and private network addresses."""
    ok = _is_public_url(url)
    if not ok:
        metrics.unsafe_urls_blocked += 1
    return ok


def _is_public_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".internal", ".localhost")):
        return False
    try:
        ip = ipaddress.ip_address(host)  # works only if the host is a raw IP address
    except ValueError:
        return True  # a normal domain name
    return not (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast)


# Sentences that try to give orders to an AI. A web page can contain these on purpose.
_INJECTION = re.compile(
    r"ignore (all |any |the )?(previous|above|prior|earlier) (instructions|prompts?|rules)"
    r"|disregard .{0,40}(instructions|rules)"
    r"|\bsystem prompt\b"
    r"|\byou are now\b",
    re.IGNORECASE,
)
_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.IGNORECASE | re.DOTALL)
_HTML_TAG = re.compile(r"<[^>]+>")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
# Invisible characters that can hide text from a human reader (zero-width space, joiners, BOM, controls).
_INVISIBLE = dict.fromkeys([*range(0, 9), 11, 12, *range(14, 32), *range(0x200B, 0x2010), *range(0x202A, 0x202F), 0x2060, 0xFEFF])


def clean_text(text: str, max_chars: int | None = None) -> str:
    """Make fetched web text safe: strip HTML and hidden characters, drop 'ignore your instructions' sentences."""
    max_chars = max_chars or settings.max_article_chars
    original = text or ""
    cleaned = _SCRIPT_OR_STYLE.sub(" ", original)
    cleaned = _HTML_TAG.sub(" ", cleaned)
    cleaned = html.unescape(cleaned).translate(_INVISIBLE)

    kept = []
    for sentence in _SENTENCE_SPLIT.split(cleaned):
        if _INJECTION.search(sentence):
            metrics.injections_removed += 1
        else:
            kept.append(sentence)
    cleaned = re.sub(r"\s+", " ", " ".join(kept)).strip()[:max_chars]

    if cleaned != original.strip():
        metrics.inputs_cleaned += 1
    return cleaned


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_OVERCLAIM = re.compile(r"\b(proves?|proven|cures?|miracle|guaranteed)\b", re.IGNORECASE)
_SPAM_PHRASES = ("click here", "subscribe now", "sponsored content", "buy now")


def numbers_missing(summary: str, source_text: str) -> list[str]:
    """Numbers that are in the summary but NOT in the article. A cheap hallucination detector."""
    source = source_text.replace(",", "")
    missing = []
    for raw in _NUMBER.findall(summary):
        number = raw.replace(",", "").rstrip(".")
        if number and number not in source:
            missing.append(number)
    return missing


def check_summary(summary: str, source_text: str) -> list[str]:
    """Output guardrail. Returns a list of problems; an empty list means the summary passed."""
    problems = []
    if len(summary) < settings.summary_min_chars:
        problems.append(f"summary too short (minimum {settings.summary_min_chars} characters)")
    if len(summary) > settings.summary_max_chars:
        problems.append(f"summary too long (maximum {settings.summary_max_chars} characters)")

    missing = numbers_missing(summary, source_text)
    if missing:
        problems.append(f"numbers not found in the source: {', '.join(missing)}")

    for word in _OVERCLAIM.findall(summary):  # e.g. "proves" when the article only says "suggests"
        if word.lower() not in source_text.lower():
            problems.append(f"overclaiming word '{word}' is not in the source")

    for phrase in _SPAM_PHRASES:
        if phrase in summary.lower():
            problems.append(f"spam phrase '{phrase}'")
    return problems


# =====================================================================
# 5. LLM GATEWAY
# =====================================================================
class LLMError(RuntimeError):
    """Raised when every model failed."""


@traceable(name="llm_chat", run_type="llm")  # shows this call in LangSmith when tracing is on
def chat(messages: list[dict], max_tokens: int = 900, temperature: float = 0.2) -> str:
    """Send messages to the main model. If it fails, try the backup model. Returns the reply text."""
    import litellm  # imported here so that importing core.py stays fast

    litellm.suppress_debug_info = True
    models = [settings.primary_model, settings.fallback_model]
    last_error = None

    for position, model in enumerate(models):
        metrics.llm_calls += 1
        try:
            response = litellm.completion(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, timeout=60
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                raise LLMError(f"{model} returned an empty reply")

            # Record tokens, cost and which model answered (for the metrics report).
            if response.usage:
                metrics.prompt_tokens += response.usage.prompt_tokens or 0
                metrics.completion_tokens += response.usage.completion_tokens or 0
            try:
                metrics.cost_usd += litellm.completion_cost(completion_response=response) or 0.0
            except Exception:  # free models often have no price listed
                pass
            metrics.models_used[model] = metrics.models_used.get(model, 0) + 1
            if position > 0:
                metrics.fallback_count += 1  # the backup model answered
            return text
        except Exception as error:  # any provider problem: remember it and try the next model
            metrics.llm_failures += 1
            last_error = error
            log.warning("model %s failed: %s", model, error)

    raise LLMError(f"all models failed: {last_error}")


def parse_json(reply: str) -> dict:
    """Find the {...} JSON object inside an LLM reply (models sometimes add ``` fences or extra words)."""
    reply = reply.replace("```json", "").replace("```", "")
    start, end = reply.find("{"), reply.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the reply")
    return json.loads(reply[start : end + 1])


# =====================================================================
# 6. SKILLS (writing rules kept in skills.md, added to the LLM prompts)
# =====================================================================
# Which skills each kind of agent gets.
SKILLS_FOR = {
    "world": ["summarize_news", "headline_style"],
    "science": ["summarize_news", "headline_style", "science_digest"],
    "ai": ["summarize_news", "headline_style", "ai_digest"],
    "layout": ["layout_rules"],
}


@cache
def _all_skills() -> dict[str, str]:
    """Read skills.md and split it into {skill name: text}. Each skill starts with a line '# Skill: name'."""
    text = (ROOT / "skills.md").read_text(encoding="utf-8")
    parts = re.split(r"^# Skill: (\w+)\s*$", text, flags=re.MULTILINE)
    # parts = [text before the first skill, name1, body1, name2, body2, ...]
    return {name: f"# Skill: {name}\n{body.strip()}" for name, body in zip(parts[1::2], parts[2::2], strict=True)}


def skills_for(kind: str) -> str:
    """The skill text for a story category ('world', 'science', 'ai') or for 'layout'."""
    return "\n\n".join(_all_skills()[name] for name in SKILLS_FOR[kind])
