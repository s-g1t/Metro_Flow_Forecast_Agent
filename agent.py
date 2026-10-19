#!/usr/bin/env python3
"""
AI agents follow this loop:
    you send a text prompt ("run the latest forecast")
        ->  the LLM determines what function inputs are required and sends them back in JSON ("call run_forecast with vintage='latest'")
            ->  the agent parses the JSON and calls the Python function with those arguments
                ->  the result goes back to the LLM as text, which interprets the results to reply in plain English.
The model never runs code or touches your files — it only speaks in JSON tool requests and text answers. Python does all the real work.

Use the below .env configuration, so this same file works across any OpenAI-compatible API (OpenAI, Gemini, Grok, etc.) without code changes:
    LLM_BASE_URL  - Gemini's endpoint (default, free tier) or http://localhost:11434/v1 for Ollama (local, private)
    LLM_API_KEY   - your key (Gemini); nothing required for local models
    LLM_MODEL     - e.g. gemini-2.0-flash, or llama3.1 for Ollama

Run it from your laptop:  python agent.py
"""

###############################################################################################################
###### IMPORTS
###############################################################################################################
import json # parses the tool-call arguments the LLM sends back
import os # reads the .env settings (model name, api key, base url)
import time # sleeps between retries when the api rate-limits us
from pathlib import Path # builds file paths that work on any machine

from dotenv import load_dotenv # loads the .env file into environment variables
from openai import OpenAI, RateLimitError # OpenAI client (works with Gemini too); the rate-limit error to catch

load_dotenv() # reads LLM_BASE_URL / LLM_API_KEY / LLM_MODEL from .env if present

# the actual tool implementations live in the tools/ folder; this just imports them
from tools import run_forecast, assess_forecast, tune_lags

###############################################################################################################
###### CONFIG
###############################################################################################################
# the folder this file lives in, so system_prompt.txt is found no matter where you run from
BASE = Path(__file__).parent
# the agent's instructions, loaded once at startup and sent with every LLM call
SYSTEM_PROMPT = (BASE / "system_prompt.txt").read_text()

###############################################################################################################
###### TOOL DEFINITIONS
###############################################################################################################
# TOOLS holds the function schemas the LLM can call: each dict gives the
# function name, a description, and its arguments in OpenAI's tool format
TOOLS = [
    # run_forecast: run the R pipeline for a vintage, then verify the output CSVs
    {
        "type": "function",
        "function": {
            "name": "run_forecast",
            "description": "Run the forecast pipeline for a specified vintage (latest or retro), then verify the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    # vintage: which forecast run to produce
                    "vintage": {
                        "type": "string",
                        "enum": ["retro", "latest"],
                        "description": "'retro' runs a prior vintage forecast, 'latest' finds the max vintage in the data."
                                       "Omitting a choice runs defaults to whatever is in the config.yml.",
                    },
                    # vintage_date: an explicit cutoff date, forces a retro run at that date
                    "vintage_date": {
                        "type": "string",
                        "description": "The vintage (i.e. forecast run period) in ISO YYYY-MM-DD format (e.g. '2017-06-01' for 2017 Q2)"
                                       "Must fall within the dataset's date range, or the agent will not run the forecast.",
                    },
                },
            },
        },
    },
    # assess_forecast: break down the latest forecast — FoF change, vs actuals, driver contributions
    {
        "type": "function",
        "function": {
            "name": "assess_forecast",
            "description": "Analyze the latest forecast: FoF variance, forecast vs. actual comparison, and driver contributions.",
            "parameters": {
                "type": "object",
                "properties": {
                    # msa: narrow the assessment to one metro, or leave blank for all
                    "msa": {
                        "type": "string",
                        "description": "Select an MSA (e.g. 'Chicago') to assess. Leaving blank will default to all MSAs in the dataset.",
                    },
                    # fundamental: narrow to one market fundamental, or leave blank for all
                    "fundamental": {
                        "type": "string",
                        "enum": ["total_inventory", "total_delivery", "total_absorption", "total_vacancy", "wgtavg_rent"],
                        "description": "Select a fundamental to assess. Leaving blank will assess all the market fundamentals.",
                    },
                    # vintage: which forecast vintage to assess, or leave blank for the latest
                    "vintage": {
                        "type": "string",
                        "description": "Select a vintage to assess. Leaving blank will default to the highest quarter label in the directory."
                                       "If the agent already ran a forecast in the session, if left blank, default to that one instead.",
                    },
                },
            },
        },
    },
    # tune_lags: grid-search the lag structure for the best out-of-sample model, per fundamental
    {
        "type": "function",
        "function": {
            "name": "tune_lags",
            "description": "Hyperparameter tuner: Define a pre-whitened grid space and searches through it to select optimal OOS model structure.",
            "parameters": {
                "type": "object",
                "properties": {
                    # fundamental: which market fundamental to tune (required)
                    "fundamental": {
                        "type": "string",
                        "enum": ["total_inventory", "total_vacancy", "total_absorption", "wgtavg_rent"],
                        "description": "Which fundamentals needs tuning.",
                    },
                },
                "required": ["fundamental"],
            },
        },
    },
]

###############################################################################################################
###### FUNCTION DISPATCH
###############################################################################################################
# maps the tool names the LLM asks for to the real Python functions that run them
FUNCTIONS = {
    "run_forecast": run_forecast.run_forecast,
    "assess_forecast": assess_forecast.assess_forecast,
    "tune_lags": tune_lags.tune_lags,
}

# which model to call; override with LLM_MODEL in .env
MODEL = os.environ.get("LLM_MODEL", "gemini-3.8-flash")

###############################################################################################################
###### _RETRY_DELAY_SECONDS
###############################################################################################################
def _retry_delay_seconds(exc: RateLimitError) -> int:
    """Pull Gemini's suggested retry delay out of the 429 error; else 30s."""
    try:
        body = exc.body
        # the error body is sometimes a raw string, so parse it first
        if isinstance(body, str):
            body = json.loads(body)
        # dig out the details list where Gemini puts the retryDelay hint
        details = (body.get("error", {}).get("details", [])
                   if isinstance(body, dict) else [])
        for item in details:
            delay = item.get("retryDelay")
            if delay:
                # strip the trailing "s", convert to int, never wait less than 30s
                return max(30, int(str(delay).rstrip("s")))
    except Exception:  # noqa: BLE001 — fall back to the fixed wait
        pass
    # no usable hint from the server: default to 30 seconds
    return 30

###############################################################################################################
###### CHAT_WITH_RETRY
###############################################################################################################
def chat_with_retry(client, **kwargs):
    """
    Throws a time-out in the loop to avoid limit kicks.
    Gemini's free tier only allows 5 requests/minute per model and the tool loop can burn through that fast.
    On a 429, wait for the server's suggested retry delay (or 30s) and try again, up to 6 attempts.
    """
    # try the call up to 6 times
    for attempt in range(1, 7):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:  # noqa: BLE001
            # rate-limited: figure out how long to wait, print it, and sleep
            wait = _retry_delay_seconds(e)
            print(f"  [rate limit] free-tier quota hit — waiting {wait}s "
                  f"(attempt {attempt}/6)…")
            time.sleep(wait)
    # final attempt with no catch: let the real error raise if still throttled
    return client.chat.completions.create(**kwargs)

###############################################################################################################
###### MAIN
###############################################################################################################
def main() -> None:
    # build the OpenAI client pointed at the configured backend (Gemini by default, Ollama if set in .env)
    client = OpenAI(
        base_url=os.environ.get(
            "LLM_BASE_URL",
            "https://generativelanguage.googleapis.com/v1beta/openai/",
        ),
        api_key=os.environ.get("LLM_API_KEY", "not-needed"),
    )
    # the conversation history: starts with the system prompt, then alternates you / agent / tool results
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Forecast agent ready. Type 'quit' to exit.\n")

    # outer loop: keep chatting until you quit
    while True:
        user = input("you: ").strip()
        if user.lower() in {"quit", "exit"}:
            break
        if not user:
            continue
        # add your message to the history
        messages.append({"role": "user", "content": user})

        # inner loop: keep calling tools until the model just answers
        while True:
            resp = chat_with_retry(
                client, model=MODEL, messages=messages, tools=TOOLS,
                #max_tokens=8192,
                max_tokens=32768,
            )
            msg = resp.choices[0].message
            # debug line: shows why the model stopped and how many tokens the call used
            print(f"  [debug] finish_reason={resp.choices[0].finish_reason} "
                  f"prompt_tokens={resp.usage.prompt_tokens} "
                  f"completion_tokens={resp.usage.completion_tokens}")
            messages.append(msg)

            # if the model wrote text, show it
            if msg.content:
                print("\n" + msg.content + "\n")
            # no tool calls means the model is done; back to waiting for your next message
            if not msg.tool_calls:
                break

            # run each tool the model asked for and feed the results back
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                print(f"  [tool] {tc.function.name}({json.dumps(args)})")
                try:
                    result = FUNCTIONS[tc.function.name](**args)
                except Exception as e:  # noqa: BLE001 — feed errors back to the model
                    result = f"TOOL ERROR: {e}"
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": str(result)})


###############################################################################################################
###### ENTRY POINT
###############################################################################################################
# only run main() when you execute this file directly (not when it's imported)
if __name__ == "__main__":
    main()
