# The Daily Agent

A multi-agent personalized newspaper: 2 world stories, 1 science or wildlife story and the 5 latest AI items, printed as a 2-page paper and served by FastAPI.

**Stack:** Python + uv, LangGraph, MCP (news tools server), LiteLLM (Groq with Gemini as backup), Pydantic, LangSmith, FastAPI, Jinja2, Docker, AWS (S3 / ECR / EC2).

Start with **HOW_IT_WORKS.md** (the plain explanation and an example). 

## Run it (Windows, PowerShell, VS Code)

```powershell
uv sync
copy .env.example .env                    # then add GROQ_API_KEY and GEMINI_API_KEY
uv run python main.py run --demo          # sample edition, no keys, no internet
uv run uvicorn main:app                   # open http://127.0.0.1:8000  (press "Generate")
uv run pytest -q                          # the 10 main tests
uv run python evals.py                    # metrics report (add --llm or --live for real runs)
uv run python main.py check-tools         # try the MCP news tools without any LLM
```

The first live run: `uv run python main.py run`. Do not use `uvicorn --reload` on Windows (it can break the MCP subprocess).

## Files

`main.py` web + CLI · `pipeline.py` the manager, page drawing, saving · `agents.py` the workers · `mcp_server.py` the 8 news tools · `core.py` settings, data shapes, metrics, guardrails, LLM gateway, skills loader · `skills.md` writing rules · `templates/newspaper.html` · `sample_data.json` · `evals.py` · `test_app.py`

## Docker

```powershell
docker build -t daily-agent .
docker run --env-file .env -p 8000:8000 daily-agent
```

## AWS (simple path)

1. S3: create a bucket, set `S3_BUCKET` in `.env` (editions get uploaded there).
2. IAM: give the EC2 instance a role that can write to that bucket.
3. ECR: create a repo, `docker build`, `docker tag`, `docker push`.
4. EC2: install Docker, pull the image, `docker run -d --restart unless-stopped --env-file .env -p 80:8000 <image>`. Set `GENERATE_TOKEN` first.
5. Daily run at 06:00 IST: cron entry `30 0 * * * curl -s -X POST -H "X-Token: <token>" "http://localhost:80/generate"`.

## Troubleshooting

- A source says `403` or `404`: fix its address in `FEEDS` in `mcp_server.py`; a dead feed is skipped.
- "All models failed": check the keys in `.env` and the model names (free-tier model names change).
- "No usable news was fetched": check your internet, then run `check-tools`.

## Honest limits

Feed URLs, live Groq/Gemini model names, LangSmith, Docker build and the AWS steps were not run end-to-end in the build environment (no internet to news sites, no Docker). Everything offline (tests, demo, metrics, MCP tool listing, page layout) was run. The demo news is fictional and marked with a banner. Photos depend on what each source provides; coloured panels fill in when there is none.
