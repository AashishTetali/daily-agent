"""main.py - the front door. Two ways in:

  Web  : uv run uvicorn main:app            -> http://127.0.0.1:8000
  Shell: uv run python main.py run --demo   -> make one edition (cron uses this too)
         uv run python main.py check-tools  -> test the MCP news tools, no LLM needed

Web pages:   GET /  (today's paper)   GET /edition/{date}   GET /edition/{date}/metrics
             GET /editions            GET /health           POST /generate?demo=false
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import secrets

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from core import settings
from pipeline import list_dates, load_edition, load_html, run_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="The Daily Agent")
_busy = asyncio.Lock()  # only one edition at a time (the metrics counters are shared)

START_PAGE = """<!doctype html><meta charset="utf-8"><title>The Daily Agent</title>
<body style="font-family:Georgia,serif;max-width:560px;margin:60px auto;padding:0 16px">
<h1>The Daily Agent</h1><p>No edition has been generated yet.</p>
<button onclick="go(false)">Generate today's edition</button>
<button onclick="go(true)">Generate demo edition</button>
<p id="msg" style="color:#555"></p>
<script>
async function go(demo){
  document.getElementById('msg').textContent='Working... this can take a minute.';
  const r = await fetch('/generate?demo='+demo,{method:'POST'});
  if(r.ok){const d=await r.json(); location.href='/edition/'+d.date;}
  else{document.getElementById('msg').textContent='Failed: '+await r.text();}
}
</script></body>"""


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    """Today's paper; if there is none yet, the newest saved one; if there is none at all, the start page."""
    return load_html(settings.today()) or next((load_html(d) for d in list_dates()), None) or START_PAGE


@app.get("/editions")
def editions() -> dict:
    return {"editions": list_dates()}


@app.get("/edition/{date}", response_class=HTMLResponse)
def edition_page(date: str) -> str:
    html = load_html(date)
    if html is None:
        raise HTTPException(404, "edition not found")
    return html


@app.get("/edition/{date}/metrics")
def edition_metrics(date: str) -> dict:
    edition = load_edition(date)
    if edition is None:
        raise HTTPException(404, "edition not found")
    return edition.metrics


@app.post("/generate")
async def generate(demo: bool = False, llm: bool | None = None, x_token: str | None = Header(default=None)) -> JSONResponse:
    """Make a new edition now (30-90 seconds). If GENERATE_TOKEN is set, the X-Token header must match."""
    if settings.generate_token and not secrets.compare_digest(x_token or "", settings.generate_token):
        raise HTTPException(401, "missing or wrong X-Token header")
    if _busy.locked():
        raise HTTPException(409, "an edition is already being generated")
    async with _busy:
        try:
            edition = await run_pipeline(demo=demo, use_llm=llm)
        except Exception as error:
            logging.getLogger("daily_agent").exception("generation failed")
            raise HTTPException(500, f"generation failed: {error}") from error
    return JSONResponse({"date": edition.date, "url": f"/edition/{edition.date}", "metrics": edition.metrics})


# ---------------------------------------------------------------- command line
async def check_tools(tool: str) -> None:
    from agents import call_tool, mcp_session

    async with mcp_session() as session:
        print("Tools offered by the MCP server:")
        for item in (await session.list_tools()).tools:
            print(f"  - {item.name}: {(item.description or '').strip().splitlines()[0]}")
        print(f"\nCalling {tool} ...")
        items = await call_tool(session, tool, {}) or []
        print(f"Got {len(items)} items. First 3:\n{json.dumps(items[:3], indent=2)}")


def cli() -> None:
    parser = argparse.ArgumentParser(description="The Daily Agent")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="make one edition")
    run.add_argument("--demo", action="store_true", help="use the sample data (offline, no keys needed)")
    run.add_argument("--llm", action="store_true", help="use the LLMs even in demo mode (tests your keys)")
    check = commands.add_parser("check-tools", help="list and try the MCP news tools")
    check.add_argument("tool", nargs="?", default="get_science_news")
    args = parser.parse_args()

    if args.command == "run":
        edition = asyncio.run(run_pipeline(demo=args.demo, use_llm=True if args.llm else None))
        print(f"Saved: {settings.editions_dir / (edition.date + '.html')}")
        print(json.dumps(edition.metrics, indent=2))
    else:
        asyncio.run(check_tools(args.tool))


if __name__ == "__main__":
    cli()
