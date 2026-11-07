"""tune_lags — agent 2. CCF-guided coordinate-descent search over one fundamental's lags.

The loop: propose lag config -> write to config.yml -> run retro vintage ->
score forecasts against actuals that arrived after the retro date -> keep the winner.

Candidates come from the CCF lag screener (MOF_CCF_Lag_Screener.R), which writes
corr_offc_fcast_vars_peaks.csv to the config's output dir: the top
cross-correlation lags per regressor on a pre-defined lag grid, computed on
first differences — the same table a person would use to hand-pick lags, so
the agent works from the identical artifact. For each
regressor the candidate set is {peak-1, peak, peak+1} around every peak, PLUS
the lag currently in config.yml — the incumbent always defends itself, so a
hand-edited config value acts as a manual override candidate and is scored head
to head against the screener's challengers instead of being silently replaced.
If the spec file is missing (screener not run yet), candidates fall back to ±2
around the current value.

Scoring: retro vintage forecasts vs. the actual rows in the NEWEST finalcast CSV
on disk (so run `run_forecast("latest")` at least once first — that provides the
actuals to score against). Metric: mean RMSE across MSAs, 1–4 steps ahead.

2021 validation gate: the search winner is NOT adopted on the search score
alone. It must also beat the initial (pre-tune) lags head to head on a long
out-of-sample window — both lag sets are re-run as a retro vintage at the
2021 Q1 origin (config vintage retro_date is temporarily set to
GATE_ORIGIN_DATE, then restored) and scored over every step ahead through the
newest actuals. That origin is honestly out-of-sample (the fit never sees
post-2021 data), so holding up across it is evidence of low model drift; it is
used as a judge for this one comparison only, never as the search objective,
so selection pressure on a single window stays minimal. If the winner fails
the gate, the initial lags are kept and the rejection is logged. If the gate
itself cannot be scored, the winner is adopted on the search score alone and
the log says so.

History: every evaluation (timestamp, fundamental, lag set, RMSE, phase —
"search" or "gate_2021" — and which lag set was finally adopted) is appended
to tune_history.csv in the config's output dir, so tuning sessions can be
compared against each other instead of living only in the returned log.
Each evaluation also records WAPE (100 * sum|error| / sum|actual|, in
percent) alongside the RMSE — display only, for human readability;
selection and the gate stay on RMSE. If the screener spec cannot be
loaded, the tune says so loudly (a warning in the log and the prints)
instead of silently falling back to ±2 candidates.

Safety: backs up config.yml before the first change; the adopted lags are left
in config.yml when the search ends; every candidate run is a full pipeline
execution, so a cap on evaluations is enforced. The whole tune is wrapped so
an interrupt or crash mid-search (Ctrl+C, SIGTERM, an exception) restores
config.yml to its exact pre-tune contents — lags and vintage retro_date
alike — instead of leaving a half-tuned file behind; the timestamped backup
remains the backstop for a hard kill (SIGKILL, power loss) that no code can
catch. Progress is printed per evaluation so a long tune is visibly alive.

Scope: lags are tuned once per fundamental and applied to all segments — the
forecast's lag construction is per fundamental, so segment-level lags are a
separate, gated pipeline enhancement, not part of this tool.
"""
from __future__ import annotations

import json
import re
import shutil
import signal
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .run_forecast import run_forecast

# Already set to your R project root (where config.yml and the .Rproj sit) — no editing needed.
PROJECT_DIR = Path("/Users/stevebook/RStudio/Projects/Market_Office_Flow_Forecast")
CONFIG_PATH = PROJECT_DIR / "config.yml"
# The tune history is an agent-side artifact, so it lives under the agent
# project root (Stephen's call, Oct 24) — not with the R pipeline's outputs.
AGENT_DIR = Path("/Users/stevebook/RStudio/Projects/Market_Office_Flow_Forecast_Agent")
MAX_EVALS = 40          # full pipeline runs are expensive; coordinate descent, one pass
LAG_MIN, LAG_MAX = 0, 12
STEPS_AHEAD = 4
GATE_ORIGIN_DATE = "2021-01-01"   # 2021 Q1 origin for the validation gate
GATE_STEPS_AHEAD = 20             # long window: every step through the newest actuals


def _load_spec(cfg: dict) -> dict:
    """Read the lag spec (top correlation peaks per regressor) from the
    CCF screener's peaks CSV.

    MOF_CCF_Lag_Screener.R writes corr_offc_fcast_vars_peaks.csv to the
    config's output dir: one row per fundamental x regressor x peak (lag
    and correlation) — the same top-lags table a person would hand-pick
    from. Returns {} when the screener hasn't been run — candidates then
    fall back to ±2 around the current config values.

    The path is based on AGENT_DIR, not PROJECT_DIR: on Stephen's laptop
    the screener's outputs land under the agent project root (the same
    layout as tune_history.csv). Basing this on PROJECT_DIR pointed at
    the non-Agent root and silently loaded nothing — the first real
    tune (Oct 24, 2026) ran entirely on fallback candidates before the
    mismatch was caught.
    """
    csv_path = AGENT_DIR / cfg["paths"]["output_dir"] / \
        "corr_offc_fcast_vars_peaks.csv"
    try:
        table = pd.read_csv(csv_path)
    except (OSError, ValueError):
        return {}
    spec: dict = {}
    for _, row in table.iterrows():
        spec.setdefault(row["offc_fundamental"], {}) \
            .setdefault(row["variable"], []).append(int(row["lag"]))
    return {f: {v: sorted(ls) for v, ls in regs.items()}
            for f, regs in spec.items()}


def _ccf_candidates(spec: dict, fundamental: str, regressor: str,
                    current: int) -> list[int]:
    """Candidate lags for one regressor.

    {p-1, p, p+1} around each screener peak, clipped to [LAG_MIN, LAG_MAX],
    plus `current` — the value presently in config.yml. The incumbent is
    always a candidate: a hand-edited config lag is a manual override that
    gets scored against the screener's challengers, never silently dropped.
    """
    peaks = spec.get(fundamental, {}).get(regressor, [])
    cands = {current}
    if not peaks:
        for d in (-2, -1, 1, 2):
            if LAG_MIN <= current + d <= LAG_MAX:
                cands.add(current + d)
        return sorted(cands)
    for p in peaks:
        for d in (-1, 0, 1):
            if LAG_MIN <= int(p) + d <= LAG_MAX:
                cands.add(int(p) + d)
    return sorted(cands)


def _set_lags(text: str, fundamental: str, lags: dict[str, int]) -> str:
    """Surgically replace lag values for one fundamental in config.yml text.

    Only the integer on each matching regressor line is touched — comments,
    spacing, quoting and key order are preserved byte-for-byte everywhere else.
    """
    lines = text.split("\n")
    out, in_lags, fund_indent, in_fund = [], False, None, False
    for line in lines:
        stripped = line.lstrip()
        if re.match(r"^lags:\s*(#.*)?$", stripped):
            in_lags, fund_indent, in_fund = True, None, False
        elif in_lags and line and not line[0].isspace():
            in_lags, in_fund = False, False  # next top-level key: left lags
        elif in_lags and stripped:
            indent = len(line) - len(stripped)
            key = re.match(r"^([\w.]+):", stripped)
            if key and fund_indent is None:
                fund_indent = indent  # first key under lags: is a fundamental
            if key and fund_indent is not None and indent == fund_indent:
                in_fund = key.group(1) == fundamental
            elif in_fund and fund_indent is not None and indent > fund_indent:
                m = re.match(r"^([\w.]+)(\s*:\s*)(\d+)(\s*(#.*)?)$", stripped)
                if m and m.group(1) in lags:
                    line = (" " * indent + m.group(1) + m.group(2)
                            + str(lags[m.group(1)]) + m.group(4))
        out.append(line)
    return "\n".join(out)


def _set_retro_date(text: str, date_str: str) -> str:
    """Surgically replace the vintage retro_date value in config.yml text.

    retro_date is a unique key in the file (vintage section); only its value
    is touched, quoting normalized to double quotes.
    """
    lines = text.split("\n")
    out = []
    for line in lines:
        m = re.match(r"^(\s*retro_date\s*:\s*)(\"[^\"]*\"|'[^']*'|[^#\s]+)"
                     r"(\s*(#.*)?)$", line)
        if m:
            line = m.group(1) + f'"{date_str}"' + (m.group(3) or "")
        out.append(line)
    return "\n".join(out)


def _wape_str(wape: float | None) -> str:
    """Format a WAPE percentage for logs and prints ('n/a' when the
    merged actuals sum to zero and the percentage is undefined)."""
    return f"{wape:.2f}%" if wape is not None else "n/a"


def _retro_label(date_str: str) -> str:
    """Vintage label for a retro_date, in the R pipeline's format.

    MOF_Forecast_Fundamentals.R names each vintage
    format(yearquarter(vintage_date), "%YQ%q") — e.g. retro_date
    2025-10-01 -> "2025Q4". The scorer must be handed the label of the
    retro run that just executed; the newest vintage label is the
    actuals source, never the forecast source.
    """
    d = pd.to_datetime(date_str)
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def _score_against_actuals(retro_vintage: str, fundamental: str,
                           steps_ahead: int = STEPS_AHEAD
                           ) -> tuple[float | None, float | None]:
    """Score the retro vintage's forecasts of ONE fundamental vs
    newest-vintage actuals, up to `steps_ahead` quarters ahead.
    Returns (rmse, wape_pct): the RMSE the search and gate select on,
    plus WAPE — the volume-weighted absolute percentage error,
    100 * sum|error| / sum|actual| over the same merged pairs —
    reported alongside for human readability. (None, None) when there
    is nothing to score.

    The frames are filtered to the fundamental being tuned before
    merging. Scoring pooled across all five fundamentals (the pre-v10
    behavior) ranks identically — the other fundamentals' errors are
    a constant background within a tune — but the reported numbers
    are the whole system's, inflated and unit-mixed, and badly
    understate the tuned fundamental's own improvement. Per-fundamental
    scores are the honest ones, the kind you can say out loud in a
    demo. (Scores from v10 on are therefore not comparable to the
    pooled scores logged by earlier versions.)

    The finalcast CSVs carry no steps_ahead column — the R pipeline
    computes one, then drops it in the final select() that shapes the
    written files — so the horizon is derived here instead: quarters
    between the vintage (production) quarter and each forecast quarter,
    from the Period ordinals assess_forecast's loader puts in `q`.

    Actuals come from the newest vintage's REPORT file, via
    assess_forecast's _actuals: finalcast files are forecast-only by
    construction (the R export binds the actual rows into the report
    file, offc_fund_forecast_{vintage}.csv, and nowhere else), so
    filtering a finalcast frame for cast_desc == "actual" is always
    empty — the merge then has nothing to join and the score is None.
    """
    from .assess_forecast import _vintages, _load, _q, _actuals
    vintages = _vintages("finalcast.offc_fund")
    if len(vintages) < 2:
        return None, None
    actuals = _actuals(vintages[0])
    if actuals is None:
        return None, None
    retro = _load("finalcast.offc_fund", retro_vintage)
    key = ["msa_desc", "class_desc", "submkt_desc", "offc_fundamental_desc", "q"]
    actual = actuals[actuals["offc_fundamental_desc"] == fundamental][key + ["offc_fundamental_val"]]
    retro_f = retro[retro["offc_fundamental_desc"] == fundamental]
    fc = retro_f[retro_f["cast_desc"] == "forecast"][key + ["offc_fundamental_val"]].copy()
    origin = _q(retro_vintage).ordinal
    fc = fc[(fc["q"].map(lambda p: p.ordinal) - origin) <= steps_ahead]
    m = actual.merge(fc, on=key, suffixes=("_act", "_fc"))
    if m.empty:
        return None, None
    err = m["offc_fundamental_val_act"] - m["offc_fundamental_val_fc"]
    rmse = float(np.sqrt((err ** 2).mean()))
    denom = float(m["offc_fundamental_val_act"].abs().sum())
    wape = float(100.0 * err.abs().sum() / denom) if denom > 0 else None
    return rmse, wape


def _append_history(cfg: dict, rows: list[dict]) -> Path:
    """Append one tuning session's evaluations to the tune history CSV."""
    hist_path = AGENT_DIR / cfg["paths"]["output_dir"] / "tune_history.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["timestamp", "fundamental", "eval",
                                     "phase", "lags", "rmse", "wape_pct",
                                     "winner"])
    df.to_csv(hist_path, mode="a", header=not hist_path.exists(), index=False)
    return hist_path


def _tune_lags_impl(fundamental: str) -> str:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    if fundamental not in cfg["lags"]:
        return f"Unknown fundamental {fundamental!r}. Choices: {list(cfg['lags'])}"
    regressors = list(cfg["lags"][fundamental].keys())
    if not regressors:
        return f"No lag keys configured for {fundamental}."

    spec = _load_spec(cfg)
    if spec:
        spec_note = "candidates: CCF-guided (screener peaks + incumbent)"
    else:
        # Loud, not silent: a fallback search is a local search, and the
        # user should know the screener's guidance never engaged.
        spec_note = ("WARNING: no CCF screener spec loaded — candidates "
                     "fall back to ±2 around the current lags")
        print(f"  [tune] {spec_note}", flush=True)
    orig_retro_date = str(cfg["vintage"]["retro_date"])

    # Back up before touching anything.
    backup = CONFIG_PATH.with_name(
        f"config.yml.bak-{datetime.now():%Y%m%d-%H%M%S}")
    shutil.copy(CONFIG_PATH, backup)

    history: list[dict] = []

    def _record(lags: dict[str, int], score: float | None,
                wape: float | None, phase: str) -> None:
        history.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "fundamental": fundamental,
            "eval": len(history) + 1,
            "phase": phase,
            "lags": json.dumps({k: int(v) for k, v in lags.items()},
                               sort_keys=True),
            "rmse": score,
            "wape_pct": round(wape, 4) if wape is not None else None,
            "winner": False,
        })

    def run_with(lags: dict[str, int]) -> tuple[float | None, float | None]:
        # Surgical lag edit — preserves config.yml comments/formatting
        # (no yaml.dump rewrite), then a genuine retro run for scoring.
        CONFIG_PATH.write_text(
            _set_lags(CONFIG_PATH.read_text(), fundamental,
                      {k: int(v) for k, v in lags.items()}))
        res = run_forecast("retro")
        if res.startswith(("ABORT", "R PIPELINE FAILED", "PIPELINE RAN but")):
            score, wape = None, None
        else:
            score, wape = _score_against_actuals(_retro_label(orig_retro_date),
                                                 fundamental)
        _record(lags, score, wape, "search")
        print(f"  [tune] search eval {len(history)}: {lags} -> "
              f"RMSE {score if score is not None else 'FAIL'} "
              f"(WAPE {_wape_str(wape)})", flush=True)
        return score, wape

    def run_gate(lags: dict[str, int]) -> tuple[float | None, float | None]:
        # Same write -> retro run -> score cycle as run_with, but at the
        # gate origin: vintage retro_date is temporarily moved to
        # GATE_ORIGIN_DATE and the run is scored over the full long window.
        # retro_date is restored by the caller once both gate runs finish.
        CONFIG_PATH.write_text(
            _set_retro_date(
                _set_lags(CONFIG_PATH.read_text(), fundamental,
                          {k: int(v) for k, v in lags.items()}),
                GATE_ORIGIN_DATE))
        res = run_forecast("retro")
        if res.startswith(("ABORT", "R PIPELINE FAILED", "PIPELINE RAN but")):
            score, wape = None, None
        else:
            score, wape = _score_against_actuals(
                _retro_label(GATE_ORIGIN_DATE), fundamental,
                steps_ahead=GATE_STEPS_AHEAD)
        _record(lags, score, wape, "gate_2021")
        print(f"  [tune] gate eval {len(history)}: {lags} -> "
              f"RMSE {score if score is not None else 'FAIL'} "
              f"(WAPE {_wape_str(wape)})", flush=True)
        return score, wape

    # Baseline: need a newer vintage's actuals to score against.
    from .assess_forecast import _vintages
    if len(_vintages("finalcast.offc_fund")) < 2:
        return ("Need two vintages to score against — run `run_forecast('latest')` first "
                "so there are actuals to compare the retro forecasts to.")

    current = {k: int(v) for k, v in cfg["lags"][fundamental].items()}
    best = dict(current)
    best_score, best_wape = run_with(current)
    if best_score is None:
        return "Baseline run failed or had nothing to score — check the R pipeline."
    baseline_score, baseline_wape = best_score, best_wape
    evals, log = 1, [spec_note,
                     f"baseline RMSE={best_score:.4g} "
                     f"(WAPE {_wape_str(best_wape)}) @ {current}"]

    # Coordinate descent: one pass over regressors, candidates guided by the
    # CCF screener's peaks with the incumbent always in the set.
    for reg in regressors:
        if evals >= MAX_EVALS:
            break
        improved = False
        for cand in _ccf_candidates(spec, fundamental, reg, best[reg]):
            if cand == best[reg] or evals >= MAX_EVALS:
                continue
            trial = dict(best); trial[reg] = cand
            score, wape = run_with(trial); evals += 1
            log.append(f"  {reg}={cand}: "
                       f"RMSE={score if score is not None else 'FAIL'} "
                       f"(WAPE {_wape_str(wape)})")
            if score is not None and score < best_score:
                best, best_score, best_wape, improved = trial, score, wape, True
        log.append(f"  -> {reg}: kept {best[reg]} (RMSE {best_score:.4g})")

    # --- 2021 validation gate -------------------------------------------
    # The search winner must also beat the initial lags on the long
    # out-of-sample window from the 2021 Q1 origin. Skipped when the search
    # kept the initial lags (nothing to validate). If the gate cannot be
    # scored, the winner is adopted on the search score alone.
    adopted = dict(best)
    if best == current:
        gate_note = ("gate skipped — the search kept the initial lags, "
                     "nothing to validate")
        log.append(f"  GATE: {gate_note}")
    else:
        gate_winner, gate_winner_wape = run_gate(best)
        gate_initial, gate_initial_wape = run_gate(current)
        # Restore the production retro_date no matter how the gate went.
        CONFIG_PATH.write_text(
            _set_retro_date(CONFIG_PATH.read_text(), orig_retro_date))
        if gate_winner is None or gate_initial is None:
            gate_note = ("gate could not be scored — winner adopted on the "
                         "search score alone")
        elif gate_winner < gate_initial:
            gate_note = (f"gate PASSED — winner {gate_winner:.4g} "
                         f"(WAPE {_wape_str(gate_winner_wape)}) beats "
                         f"initial {gate_initial:.4g} "
                         f"(WAPE {_wape_str(gate_initial_wape)}) "
                         f"on the 2021 window")
        else:
            adopted = dict(current)
            gate_note = (f"gate REJECTED the search winner — winner "
                         f"{gate_winner:.4g} "
                         f"(WAPE {_wape_str(gate_winner_wape)}) vs initial "
                         f"{gate_initial:.4g} "
                         f"(WAPE {_wape_str(gate_initial_wape)}) "
                         f"on the 2021 window; initial lags kept")
        log.append(f"  GATE: {gate_note}")

    # Write the adopted lags (surgical — config.yml comments and formatting
    # untouched). On a gate rejection this restores the initial set.
    CONFIG_PATH.write_text(_set_lags(CONFIG_PATH.read_text(), fundamental,
                                     {k: int(v) for k, v in adopted.items()}))

    # Record the session: flag the search evaluation that produced the
    # adopted lag set, and append every evaluation to the tune history CSV.
    adopted_json = json.dumps({k: int(v) for k, v in adopted.items()},
                              sort_keys=True)
    for row in history:
        if row["phase"] == "search" and row["lags"] == adopted_json:
            row["winner"] = True
            break
    hist_path = _append_history(cfg, history)

    headline = (f"TUNE DONE for {fundamental} ({evals} search runs). "
                f"Baseline RMSE={baseline_score:.4g} "
                f"(WAPE {_wape_str(baseline_wape)}) -> search winner: {best} "
                f"with RMSE={best_score:.4g} (WAPE {_wape_str(best_wape)}).")
    if adopted != best:
        headline += f" Adopted: {adopted} (gate rejection — see log)."
    return (headline + "\n"
            f"Backup of the pre-tune config: {backup.name}\n"
            f"Tune history appended to {hist_path.name} ({len(history)} evaluations).\n"
            + "\n".join(log))


def tune_lags(fundamental: str) -> str:
    """Crash-safe wrapper around the tuning implementation.

    Captures config.yml's exact pre-tune contents up front. If the tune
    ends abnormally — Ctrl+C, SIGTERM, or an exception mid-search — the
    file is restored byte-for-byte (trial lags and the gate's temporary
    retro_date alike), so an interrupted tune never leaves a half-tuned
    config behind. A normal return, even an error message, counts as a
    controlled end and keeps whatever state the implementation left.
    A hard kill (SIGKILL, power loss) cannot be caught by any code — the
    timestamped backup written by the implementation is the recovery
    path for that case.
    """
    original_text = CONFIG_PATH.read_text()
    completed = False

    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt("terminated (SIGTERM)")

    old_handler = None
    try:
        old_handler = signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        old_handler = None  # not the main thread — Ctrl+C path still applies
    try:
        result = _tune_lags_impl(fundamental)
        completed = True
        return result
    finally:
        if old_handler is not None:
            signal.signal(signal.SIGTERM, old_handler)
        if not completed:
            CONFIG_PATH.write_text(original_text)
            print("  [tune] tune ended abnormally — config.yml restored "
                  "to its exact pre-tune state", flush=True)
