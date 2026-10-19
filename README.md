# Metro Flow Forecast Agent

An AI agent that operates an econometric forecasting pipeline. The agent runs the models, evaluates the output, and writes up what the forecast says — in plain English.

I'm a business economist, not an AI engineer. I built this to automate the tedious parts of my own forecasting workflow: running the pipeline, checking the outputs, and drafting the assessment. The design principle is simple: **tools do the numbers, the agent does the words.** No language is generated inside the tools; the LLM only ever narrates what the code computed.

## How it works

```
you  →  LLM (with tools)  →  your Python function actually does the thing
                               →  result goes back to the LLM
                                   →  LLM answers you
```

The agent (`agent.py`) is a tool-calling loop over any OpenAI-compatible API. It reads its instructions from `system_prompt.txt` and exposes three tools:

| Tool | What it does |
|---|---|
| `run_forecast(vintage)` | Syncs the forecast repo first (`sync_repo`), then executes the R forecasting pipeline via `Rscript` and verifies the vintage output CSVs were actually written. `vintage` optionally overrides the config (`'retro'` / `'latest'`), restored byte-for-byte after the run. Reports the repo commit hash with the result. |
| `sync_repo()` | `git pull --ff-only` on the forecast checkout; returns the commit hash as provenance. Runs automatically at the start of every `run_forecast`. |
| `assess_forecast(vintage, msa, fundamental)` | Reads the forecast CSVs and reports forecast-over-forecast change, jump-off vs. actuals, driver moves, uncertainty, and segment accuracy. Analysis logic lives in `tools/assess_forecast.py`. |
| `tune_lags(fundamental)` | Coordinate-descent search over lag structure in `config.yml`, scored on retro-vintage backtests. Backs up the config before touching it. |

The R pipeline it drives lives in a separate repo: [Metro_Office_Flow_Forecast](https://github.com/s-g1t/Metro_Office_Flow_Forecast) — ARIMA models (fable) over office-market fundamentals, all parameters in `config.yml`. `run_forecast` syncs that repo (`git pull`) before every run and reports the commit hash, so every forecast is traceable to an exact code version.

## Setup

1. Clone the forecast repo somewhere on your machine, then clone this repo next to it (or set `PROJECT_DIR` in `run_forecast.py` to the forecast checkout — it expects the `R/` + `config.yml`-at-root layout).
2. `pip install openai python-dotenv pandas pyyaml`
3. Copy `.env.example` to `.env` and set your backend:

```
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/   # Gemini, free tier
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-2.0-flash
```

Any OpenAI-compatible endpoint works — local Ollama (`http://localhost:11434/v1`) included. No code changes needed to swap backends.

4. `python3 agent.py`

## Ground rules (from the system prompt)

1. Never edit the R pipeline directly — tunable things live in `config.yml` only.
2. Back up `config.yml` (timestamped) before any rewrite.
3. After every pipeline run, verify the output CSVs exist before claiming success.
4. Percentages over 100% are reported in volume terms, never quoted as percentages.

## Where this is going

This repo starts with the original v1 files. Follow the commit history to watch it grow: the assessment loop, the lag tuner, multi-vintage backtesting, and (eventually) a generalized narrative engine that works on any forecast pipeline, not just this one. One codebase, multiple configured instances.
