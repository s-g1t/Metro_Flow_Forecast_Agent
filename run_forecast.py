"""run_forecast — shell out to Rscript, then verify the vintage CSVs exist."""
import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import yaml

# set the project directory: set FORECAST_REPO_DIR in .env to override,
# otherwise uses the local forecast checkout below
PROJECT_DIR = Path(os.environ.get("FORECAST_REPO_DIR", "~/RStudio/Projects/Market_Office_Flow_Forecast")).expanduser()

###############################################################################################################
###### SYNC_REPO
###############################################################################################################
def sync_repo() -> str:
    """git pull the forecast repo; return the commit hash as provenance."""
    # pull latest, fast-forward only so a dirty checkout fails loudly instead of merging
    pull = subprocess.run(["git", "pull", "--ff-only"], cwd=PROJECT_DIR,
                          capture_output=True, text=True, timeout=120)
    if pull.returncode != 0:
        return f"SYNC FAILED: {pull.stderr[-500:] or pull.stdout[-500:]}"
    # grab the commit hash so the run reports exactly which code version it used
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                         capture_output=True, text=True, timeout=30)
    commit = rev.stdout.strip()
    return f"repo synced to {commit}" if commit else "SYNC OK (hash unreadable)"

###############################################################################################################
###### _OVERRIDE_VINTAGE_VALUE
###############################################################################################################
def _override_vintage_value(text: str, key: str, value: str) -> str:
    """
    First, we need to parse the config.yml apart line by line to find and overwrite the vintage.
    Allows for surgical edits to the config.yml file, replacing specific lines and returning the entire "edited" config.
    The input gets returned unchanged if the vintage block or the key line isn't found.
    """
    # split the config text into a list of lines
    lines = text.split("\n")
    out, in_vintage = [], False
    # loop through each config line
    for line in lines:
        # strip leading whitespace so the vintage-header match works even if indented
        stripped = line.lstrip()
        # if the line is the vintage block header, then in_vintage is TRUE and the vintage block has been identified
        if re.match(r"^vintage:\s*(#.*)?$", stripped):
            in_vintage = True
        # else if the line isn't indented the next top-level key has been reach (i.e. now past the vintage block)
        elif in_vintage and line and not line[0].isspace():
            in_vintage = False
        # if inside the vintage block, check whether this line is the right key
        if in_vintage:
            # match the key's line, capturing the key prefix and any trailing
            # comment separately, so only the value gets swapped
            m = re.match(rf"^(\s*{re.escape(key)}:\s*[\"']?)([^\"'#\s]+)([\"']?\s*(#.*)?)$", line)
            # if the line matched, swap in the new value
            if m:
                line = f"{m.group(1)}{value}{m.group(3)}"
                # only change the first matching line, then stop looking
                in_vintage = False
        out.append(line)
    return "\n".join(out)

###############################################################################################################
###### RUN_FORECAST
###############################################################################################################
def run_forecast(vintage: str | None = None, vintage_date: str | None = None) -> str:
    """
    Run MOF_Forecast_Fundamentals.R and validate output.
    Vintage configurations:
        'retro' temporarily overwrites vintage in the config.yml,
        'latest' defaults to max vintage in the data,
        'none' defaults to the vintage date in the config.yml.
    Inputting a vintage date will force the vintage.desc to 'retro' for the current run, overriding the config.yml
    If the vintage date doesn't fall within the range of the input data, the agent will not run the forecast.
    Overrides are surgical one-line edits and the file is restored byte-for-byte afterwards.
    """

    # NOTE: abort lets the LLM know to kill the query, returned error messages will keep the session going
    # sync the forecast repo first, so the run always uses the latest pushed code/data
    sync_msg = sync_repo()
    if sync_msg.startswith("SYNC FAILED"):
        return f"ABORT: {sync_msg} — check the repo remote."
    # check to see if the forecast R script is in the project directory (R/ subdir in the repo layout)
    script = PROJECT_DIR/"R"/"MOF_Forecast_Fundamentals.R"
    if not script.exists():
        return f"ABORT: R script not found at {script} — check PROJECT_DIR."
    # load the yaml config file
    cfg_path = PROJECT_DIR/"config.yml"
    config = yaml.safe_load(cfg_path.read_text())
    forecast_dir = PROJECT_DIR/config["paths"]["forecast_dir"].lstrip("./")

###############################################################################################################
###### FUNCTION INPUT CHECK
###############################################################################################################
    # check to see if vintage_date is a valid yyyy-mm-dd format
    if vintage_date:
        try:
            req = pd.to_datetime(vintage_date).date() # lubridate's ymd(vintage_date)
        except (ValueError, TypeError):
            return f"ABORT: vintage_date '{vintage_date}' is not a parseable date (use YYYY-MM-DD)."
        date_desc = config["data"]["date_desc"]
        fund = pd.read_csv(PROJECT_DIR/config["data"]["offc.fundamentals"],
                           usecols=[date_desc], parse_dates=[date_desc])
        # find the min and max dates
        lo, hi = fund[date_desc].min().date(), fund[date_desc].max().date()
        # check if vintage_date is within the min and max
        if not lo <= req <= hi:
            return (f"ABORT: vintage_date {req} is outside the dataset range ({lo} to {hi}).")
        # convert the parsed vintage_date back into a string
        vintage_date = req.isoformat()

###############################################################################################################
###### CONFIG FILE OVERRIDES
###############################################################################################################
    # load the config file in as one long text string, and alias it
    original = cfg_path.read_text()
    patched = original
    # check for a vintage_date function input and default to retro and pass the date string input to retro_date
    if vintage_date:
        patched = _override_vintage_value(patched, "desc", "retro")
        patched = _override_vintage_value(patched, "retro_date", vintage_date)
    # if the vintage function input is retro or latest then pass it, otherwise keep the original config string
    elif vintage in ("retro", "latest"):
        patched = _override_vintage_value(patched, "desc", vintage)
    # if the aliased and original config strings don't match then write the aliased back to the config file
    if patched != original:
        cfg_path.write_text(patched)

###############################################################################################################
###### RUN THE R SCRIPT
###############################################################################################################
    # snapshot the forecast dir before the run, so validation checks this run's
    # outputs instead of stale files from earlier vintages
    before = {p.name: p.stat().st_mtime for p in forecast_dir.glob("*.csv")}
    # run the forecast script: subprocess.run uses Rscript to run code without opening RStudio
    try:
        proc = subprocess.run(["Rscript", str(script)], cwd=PROJECT_DIR,
                              capture_output=True, text=True, timeout=3600)
    finally:
        # if any changes were made to the config text string, write the original back to the config file
        if patched != original:
            cfg_path.write_text(original)
    # if Rscript failed (0 is a successful run), grab the end of the error log (last 2k chars)
    if proc.returncode != 0:
        tail = proc.stderr[-2000:] or proc.stdout[-2000:]
        return f"R PIPELINE FAILED (exit {proc.returncode}). Tail of log:\n{tail}"

###############################################################################################################
###### FILE OUTPUT CHECK
###############################################################################################################
    # check this run's outputs: only files created or modified by the run count
    after = {p.name: p.stat().st_mtime for p in forecast_dir.glob("*.csv")}
    fresh = {n for n, m in after.items() if n not in before or m != before[n]}
    # create missing and found lists
    missing, found = [], []
    # loop through the expected output list
    for name in config["output"]:
        # newest file this run wrote for this dataset, if any
        matches = sorted(n for n in fresh if n.startswith(f"{name}_") and n.endswith(".csv"))
        # if there's a match name goes in the found list, else name goes into the missing list
        (found if matches else missing).append(f"{name}: {matches[-1]}" if matches else name)
    if missing:
        return ("PIPELINE RAN but expected outputs are missing: "
                f"{missing}. Found: {found}. Check the R log.")
    mode = f"retro @ {vintage_date}" if vintage_date else vintage or config["vintage"]["desc"]
    return (f"PIPELINE OK (vintage mode: {mode}; {sync_msg}). CSVs verified "
            f"in {forecast_dir}:\n" + "\n".join(f"  - {f}" for f in found))
