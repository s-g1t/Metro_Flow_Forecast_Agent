"""assess_forecast — forecast-over-forecast, jump-off attribution, driver moves, uncertainty, segment accuracy.

Works off the pipeline's own CSVs:
  {paths.forecast_dir}/finalcast.offc_fund_{vintage}.csv   (forecasts + intervals)
  {paths.forecast_dir}/coefs.offc_fund_{vintage}.csv
  {paths.output_dir}/offc_metro_forecast_vars_{vintage}.csv   (driver file)
  {paths.report_dir}/offc_fund_forecast_{vintage}.csv         (actuals + forecasts)

No R needed — pure pandas so the agent can run it anywhere.

What the report gives the model, per fundamental and segment:
  - FoF at the max overlapping horizon: new vs old level, net change, % change
    (vacancy also gets the vacancy-rate change in percentage points)
  - Jump-off: latest actual (from the report file) vs the prior vintage's
    forecast for that quarter
  - Driver moves: top 2 econometric regressors ranked behind the scenes by
    |coef x driver| level contribution; each reported with its own
    quarter-over-quarter impulse in native units at the lagged date the model
    consumed (e.g. "+2.1M over Q4 2022 → Q1 2023 (12-quarter lag)").
    no coefficients appear in the report.
  - Uncertainty: 95% interval half-width vs point forecast, ranked across fundamentals
  - Segment accuracy: which class/submarket beat/missed the prior forecast,
    plus consecutive-quarter streaks
"""
from __future__ import annotations

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
    # grab the commit hash so the report shows exactly which code version the CSVs came from
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR,
                         capture_output=True, text=True, timeout=30)
    commit = rev.stdout.strip()
    return f"repo synced to {commit}" if commit else "SYNC OK (hash unreadable)"

###############################################################################################################
###### DRIVER_NAMES
###############################################################################################################
# Plain-english display names for the econometric regressors.
# The agent never sees raw column names — it sees these.
#
# HOW TO CONFIGURE:
#   - left side: the driver column base name, exactly as it appears in the
#     driver file (offc_metro_forecast_vars_{vintage}.csv) and in config.yml
#     under lags: — WITHOUT the _lag{n} suffix the R script adds.
#     do not rename the left side unless the column itself is renamed.
#   - right side: whatever words you want the agent to say. free text.
#     this is the only place display names live, so edit here once and every
#     report picks it up (the file is read fresh each run, no restart needed).
#   - adding a driver to config.yml lags:? add one line here.
#   - anything not listed falls back to the column name with underscores
#     turned into spaces.
#   - shock interactions (columns ending in _shock), AR/MA terms, the intercept
#     and seasonal dummies are never ranked, so they need no entries.
DRIVER_NAMES = {
    "metro_gmp_smoothed": "annualized economic productivity (GMP)",
    "metro_residential_supply_smoothed": "residential supply",
    "metro_constr_employment_smoothed": "construction employment",
    "metro_office_employment_smoothed": "office employment",
    "metro_office_pop_age_25_54_smoothed": "prime-age population (25-54)",
    "metro_gmp": "economic productivity (GMP)",
    "metro_employment_total": "total employment",
    "metro_total_vacancy": "total vacancy",
    "metro_office_employment": "office employment",
    "metro_cpi": "consumer prices (CPI)",
    "inventory_delta": "inventory change",
    "vacancy_delta": "vacancy change",
}

###############################################################################################################
###### _UNITS
###############################################################################################################
# display units per fundamental: "$" prefixes, anything else suffixes
_UNITS = {
    "total_inventory": "sq ft",
    "total_absorption": "sq ft",
    "total_vacancy": "sq ft",
    "total_delivery": "sq ft",
    "wgtavg_rent": "$",
}

###############################################################################################################
###### FUNDAMENTALS
###############################################################################################################
# the series this assessment covers. the pipeline files carry ~11 series
# (rolling_*, total_construction, ...); the write-up covers these five.
# total_vacancy_rt is supporting only — it feeds the vacancy-rate pp lines.
FUNDAMENTALS = ["total_inventory", "total_delivery", "total_absorption",
                "total_vacancy", "wgtavg_rent"]

###############################################################################################################
###### _CONFIG
###############################################################################################################
def _config() -> dict:
    # read the config file fresh every time so yaml edits take effect without restarting
    return yaml.safe_load((PROJECT_DIR / "config.yml").read_text())

###############################################################################################################
###### _FORECAST_DIR
###############################################################################################################
def _forecast_dir() -> Path:
    # pull the forecast directory out of the config and turn it into a real path
    cfg = _config()
    return PROJECT_DIR / cfg["paths"]["forecast_dir"].lstrip("./")

###############################################################################################################
###### _OUTPUT_DIR
###############################################################################################################
def _output_dir() -> Path:
    # the R script writes the vintaged driver file here each run
    cfg = _config()
    return PROJECT_DIR / cfg["paths"]["output_dir"].lstrip("./")

###############################################################################################################
###### _REPORT_DIR
###############################################################################################################
def _report_dir() -> Path:
    # the R script writes offc_fund_forecast_{vintage}.csv here each run —
    # actuals + forecasts in one file (finalcast files are forecast-only)
    cfg = _config()
    return PROJECT_DIR / cfg["paths"]["report_dir"].lstrip("./")

###############################################################################################################
###### _ACTUALS
###############################################################################################################
def _actuals(vintage: str) -> pd.DataFrame | None:
    # actual rows for one vintage, from the report file. returns None when the
    # file is missing so the rest of the assessment still runs.
    p = _report_dir() / f"offc_fund_forecast_{vintage}.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    date_desc = _config()["data"]["date_desc"]
    if date_desc not in df.columns:
        return None
    # turn the date column into quarter labels the rest of the code can use
    df["q"] = df[date_desc].map(_q)
    # the report file holds actuals (vintage "Historical") and forecasts
    return df[df["cast_desc"] == "actual"].copy()

###############################################################################################################
###### _VINTAGES
###############################################################################################################
def _vintages(stem: str) -> list[str]:
    """All vintage labels present for a CSV stem, highest quarter label first."""
    # collect the vintage labels from the filenames
    labels = set()
    for p in _forecast_dir().glob(f"{stem}_*.csv"):
        m = re.match(rf"{re.escape(stem)}_(.+)\.csv$", p.name)
        if m:
            labels.add(m.group(1))
    # sort by quarter label, highest first, so the comparison is chronological
    # (not by write time — a retro run should not reorder the comparison)
    # _q is defined below but resolved when this runs
    return sorted(labels, key=_q, reverse=True)

###############################################################################################################
###### _Q
###############################################################################################################
def _q(date: str) -> pd.Period:
    """Parse '2025 Q3' / '2025Q3' / '2025-03' style quarter labels."""
    # clean up whatever came in, then try the quarter formats one at a time
    s = str(date).strip()
    # first try: a 4-digit year, optional space, Q, and a quarter digit
    m = re.search(r"(\d{4})\s*Q(\d)", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")
    # second try: yyyy-mm, convert the month into a quarter number
    m = re.search(r"(\d{4})-(\d{2})", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{(int(m.group(2)) - 1) // 3 + 1}", freq="Q")
    # nothing matched, so blow up with the bad value in the message
    raise ValueError(f"unparseable quarter: {date!r}")

###############################################################################################################
###### _QLABEL
###############################################################################################################
def _qlabel(p: pd.Period) -> str:
    # "2023Q1" -> "Q1 2023" for the report
    return f"Q{p.quarter} {p.year}"

###############################################################################################################
###### _FMT
###############################################################################################################
def _fmt(x) -> str:
    # compact human-readable numbers: 4.36M, 152,600, 42.50
    if x is None or pd.isna(x):
        return "n/a"
    ax = abs(x)
    if ax >= 1_000_000:
        return f"{x / 1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x:,.0f}"
    if ax >= 100:
        return f"{x:,.1f}"
    return f"{x:.2f}"

def _with_unit(x, fundamental: str) -> str:
    # attach the fundamental's unit: "$42.50", "4.36M sq ft"
    s = _fmt(x)
    if s == "n/a":
        return s
    u = _UNITS.get(fundamental, "")
    if u == "$":
        return f"${s}"
    return f"{s} {u}".strip()

def _vol(new, old, fundamental: str) -> str:
    # "4.36M sq ft vs 4.21M sq ft (+152,600 sq ft, +3.6%)"
    # thin base: when the % would exceed 100%, report volume terms only —
    # the agent must never see a quotable percentage above 100%
    if pd.isna(new) or pd.isna(old):
        return f"{_with_unit(new, fundamental)} vs {_with_unit(old, fundamental)} (n/a)"
    d = new - old
    ds = f"+{_with_unit(abs(d), fundamental)}" if d >= 0 else f"-{_with_unit(abs(d), fundamental)}"
    base = f"{_with_unit(new, fundamental)} vs {_with_unit(old, fundamental)} ({ds}"
    if old == 0:
        return base + ", % n/a — zero base)"
    if abs(d / old) > 1:
        return base + " — base near zero, % not meaningful)"
    return base + f", {d / old:+.1%})"

###############################################################################################################
###### _LOAD
###############################################################################################################
def _load(stem: str, vintage: str, parse_dates: bool = True) -> pd.DataFrame:
    # read the one csv for this dataset and vintage
    df = pd.read_csv(_forecast_dir() / f"{stem}_{vintage}.csv")
    # the coefs file has no date column (it carries a vintage column instead)
    # and nothing downstream uses q for it, so skip the date work when the
    # caller says it doesn't need it
    if not parse_dates:
        return df
    # the date column name lives in the config (same one run_forecast uses)
    date_desc = _config()["data"]["date_desc"]
    # fail fast with a clear message if the column is missing, instead of
    # crashing later on a mysterious missing "q" key in the merges
    if date_desc not in df.columns:
        raise ValueError(f"date column '{date_desc}' not found in {stem}_{vintage}.csv")
    # parse every date value into a quarterly period so later merges can join on it
    df["q"] = df[date_desc].map(_q)
    return df

###############################################################################################################
###### _DRIVER_VARS
###############################################################################################################
def _driver_vars(vintage: str) -> pd.DataFrame:
    # the vintaged driver file the R script writes to output_dir each run:
    # metro-aggregated fundamentals plus the raw driver columns, by date and msa
    p = _output_dir() / f"offc_metro_forecast_vars_{vintage}.csv"
    if not p.exists():
        raise ValueError(f"driver file {p.name} not found in output_dir — "
                         "the R script writes it each run, older vintages may predate it")
    df = pd.read_csv(p)
    date_desc = _config()["data"]["date_desc"]
    if date_desc not in df.columns:
        raise ValueError(f"date column '{date_desc}' not found in {p.name}")
    df["q"] = df[date_desc].map(_q)
    return df

###############################################################################################################
###### _REGRESSORS
###############################################################################################################
def _regressors(fundamental: str) -> dict:
    # the econometric regressors for this fundamental, from config lags
    # shock interactions are excluded: they are episodic covid dummies,
    # not narratable driver moves
    lags = _config().get("lags", {}).get(fundamental, {})
    return {k: v for k, v in lags.items() if not k.endswith("_shock")}

###############################################################################################################
###### _BASE_SERIES
###############################################################################################################
def _base_series(drv: pd.DataFrame, msa: str, base: str):
    # one quarterly series for an msa: a raw driver column, or a delta
    # derived the same way the R script builds it. returns None when the
    # column is not available. metro-level only, so this is approximate
    # for class/submarket groups — fine for ranking, not for the forecast.
    d = drv[drv["msa_desc"] == msa].sort_values("q")
    if d.empty:
        return None
    if base == "inventory_delta":
        if "metro_total_inventory" not in d.columns:
            return None
        return d.set_index("q")["metro_total_inventory"].diff()
    if base == "vacancy_delta":
        if "metro_total_vacancy" not in d.columns:
            return None
        return d.set_index("q")["metro_total_vacancy"].diff()
    if base not in d.columns:
        return None
    return d.set_index("q")[base]

###############################################################################################################
###### _DISPLAY_NAME
###############################################################################################################
def _display_name(base: str) -> str:
    # glossary lookup, falling back to the column name prettified
    if base in DRIVER_NAMES:
        return DRIVER_NAMES[base]
    return base.replace("_", " ")

###############################################################################################################
###### _FOF
###############################################################################################################
def _fof(new: pd.DataFrame, old: pd.DataFrame,
         msa: str | None, fundamental: str | None, actuals=None):
    """Per-group FoF at the max overlapping horizon, jump-off, uncertainty.

    Returns (seg_order, seg_blocks, unc_rows): seg_order maps fundamental ->
    [(class, submarket), ...] in display order; seg_blocks maps
    (fundamental, class, submarket) -> report lines for that segment;
    unc_rows is [(fundamental, unc_pct), ...] for the cross-fundamental
    uncertainty ranking.

    Segments are grouped by submarket within each fundamental's block —
    submarkets ordered by their max segment level at the horizon, segments
    highest-first within each submarket — so the write-up covers one
    submarket at a time.
    """
    # limit to the requested slice for group discovery
    f = new.copy()
    if msa:
        f = f[f["msa_desc"].str.contains(msa, case=False, na=False)]
    if fundamental:
        f = f[f["offc_fundamental_desc"] == fundamental]
    else:
        # the files carry extra series (rolling_*, construction, ...);
        # the assessment covers the FUNDAMENTALS list
        f = f[f["offc_fundamental_desc"].isin(FUNDAMENTALS)]
    groups = f[["msa_desc", "offc_fundamental_desc",
                "class_desc", "submkt_desc"]].drop_duplicates()
    # inventory lookup for vacancy rates: (source, msa, class, submkt, q, cast) -> value
    inv = {}
    for r in new[new["offc_fundamental_desc"] == "total_inventory"].itertuples():
        inv[("new", r.msa_desc, r.class_desc, r.submkt_desc, r.q, r.cast_desc)] = r.offc_fundamental_val
    for r in old[old["offc_fundamental_desc"] == "total_inventory"].itertuples():
        inv[("old", r.msa_desc, r.class_desc, r.submkt_desc, r.q, r.cast_desc)] = r.offc_fundamental_val
    if actuals is not None:
        for r in actuals[actuals["offc_fundamental_desc"] == "total_inventory"].itertuples():
            inv[("act", r.msa_desc, r.class_desc, r.submkt_desc, r.q, r.cast_desc)] = r.offc_fundamental_val

    # pass 1: per-group horizon level, so segments can be listed highest-first
    recs = []
    for g in groups.itertuples():
        gm, gf, gc, gs = g.msa_desc, g.offc_fundamental_desc, g.class_desc, g.submkt_desc
        # slice the two vintages down to this group
        ng = new[(new["msa_desc"] == gm) & (new["offc_fundamental_desc"] == gf) &
                 (new["class_desc"] == gc) & (new["submkt_desc"] == gs)]
        og = old[(old["msa_desc"] == gm) & (old["offc_fundamental_desc"] == gf) &
                 (old["class_desc"] == gc) & (old["submkt_desc"] == gs)]
        nf = ng[ng["cast_desc"] == "forecast"]
        of = og[og["cast_desc"] == "forecast"]
        # inner-join on quarter: only overlapping forecast rows compare
        m = nf[["q", "offc_fundamental_val", "lower", "upper"]].merge(
            of[["q", "offc_fundamental_val"]], on="q", suffixes=("_new", "_old"), how="inner")
        if m.empty:
            recs.append((float("-inf"), gm, gf, gc, gs, None, None, None, None))
            continue
        # headline at the max overlapping horizon
        H = m["q"].max()
        hrow = m[m["q"] == H].iloc[0]
        vn = hrow["offc_fundamental_val_new"]
        recs.append((vn if not pd.isna(vn) else float("-inf"), gm, gf, gc, gs, m, of, H, hrow))
    # pass 2: emit lines — group by submarket within each fundamental
    # (submarkets ordered by max segment level at the horizon), segments
    # highest-first within each submarket
    from collections import defaultdict
    by_gf = defaultdict(list)
    for r in recs:
        by_gf[r[2]].append(r)
    seg_order = {}
    seg_blocks = {}
    unc_rows = []
    for gf, gr in by_gf.items():
        submax = {}
        for r in gr:
            submax[r[4]] = max(submax.get(r[4], float("-inf")), r[0])
        subrank = {s: i for i, s in
                   enumerate(sorted(submax, key=lambda s: submax[s], reverse=True))}
        gr.sort(key=lambda r: (subrank[r[4]],
                               -r[0] if r[0] != float("-inf") else float("inf")))
        order = []
        for _, gm, _, gc, gs, m, of, H, hrow in gr:
            order.append((gc, gs))
            blines = seg_blocks.setdefault((gf, gc, gs), [])
            if m is None:
                blines.append(f"  - {gm} / {gf} / {gc} {gs}: no overlapping forecast rows.")
                continue
            blines.append(f"  - {gm} / {gf} / {gc} {gs} @ {_qlabel(H)}: "
                         f"{_vol(hrow['offc_fundamental_val_new'], hrow['offc_fundamental_val_old'], gf)}")
            # jump-off: latest actual (report file) vs the prior vintage's forecast
            # for that quarter — the finalcast files are forecast-only
            if actuals is not None:
                na = actuals[(actuals["msa_desc"] == gm) &
                             (actuals["offc_fundamental_desc"] == gf) &
                             (actuals["class_desc"] == gc) &
                             (actuals["submkt_desc"] == gs)]
                if not na.empty:
                    T = na["q"].max()
                    actual_T = na[na["q"] == T].iloc[0]["offc_fundamental_val"]
                    ofT = of[of["q"] == T]
                    if ofT.empty:
                        blines.append(f"      jump-off: actual {_qlabel(T)} {_with_unit(actual_T, gf)} "
                                     "(no prior forecast for that quarter)")
                    else:
                        fT = ofT.iloc[0]["offc_fundamental_val"]
                        blines.append(f"      jump-off: actual {_qlabel(T)} vs prior forecast: {_vol(actual_T, fT, gf)}")
                        # vacancy also shows the rate jump-off in percentage points
                        if gf == "total_vacancy":
                            r_act = inv.get(("act", gm, gc, gs, T, "actual"))
                            r_fc = inv.get(("old", gm, gc, gs, T, "forecast"))
                            if all(v is not None and not pd.isna(v) and v != 0
                                   for v in [r_act, r_fc, actual_T, fT]):
                                ra, rf = actual_T / r_act, fT / r_fc
                                blines.append(f"      actual rate {_qlabel(T)}: {ra:.1%} vs prior forecast "
                                             f"{rf:.1%} ({(ra - rf) * 100:+.1f}pp)")
            # uncertainty at H from the 95% interval half-width; when the base is
            # near zero the percent explodes, so report volume terms instead
            lo, hi, fc = hrow["lower"], hrow["upper"], hrow["offc_fundamental_val_new"]
            if not any(pd.isna([lo, hi, fc])) and fc != 0:
                half = (hi - lo) / 2
                pct = half / fc
                if abs(pct) > 1:
                    blines.append(f"      uncertainty @ {_qlabel(H)}: ±{_with_unit(half, gf)} "
                                 "(base near zero — % not meaningful)")
                else:
                    blines.append(f"      uncertainty @ {_qlabel(H)}: ±{pct:.1%}")
                unc_rows.append((gf, pct))
            # vacancy also reports the rate move in percentage points
            if gf == "total_vacancy":
                rn = inv.get(("new", gm, gc, gs, H, "forecast"))
                ro = inv.get(("old", gm, gc, gs, H, "forecast"))
                vn, vo = hrow["offc_fundamental_val_new"], hrow["offc_fundamental_val_old"]
                if all(v is not None and not pd.isna(v) and v != 0 for v in [rn, ro]):
                    r_new, r_old = vn / rn, vo / ro
                    blines.append(f"      vacancy rate @ {_qlabel(H)}: {r_new:.1%} vs {r_old:.1%} "
                                 f"({(r_new - r_old) * 100:+.1f}pp)")
        seg_order[gf] = order
    return seg_order, seg_blocks, unc_rows

###############################################################################################################
###### _UNC_RANKING
###############################################################################################################
def _unc_ranking(unc_rows) -> str:
    # rank fundamentals by worst interval width: "vacancy is most susceptible..."
    best = {}
    for gf, pct in unc_rows:
        if pct is not None and (gf not in best or pct > best[gf]):
            best[gf] = pct
    if not best:
        return "Forecast uncertainty: no interval bounds available."
    order = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    bits = []
    for gf, p in order:
        bits.append(f"{gf} ±>100% (thin base)" if abs(p) > 1 else f"{gf} ±{p:.1%}")
    return ("Most susceptible to over/under-prediction "
            "(95% interval half-width vs point forecast): " + ", ".join(bits))

###############################################################################################################
###### _ATTRIBUTION
###############################################################################################################
def _attribution(new: pd.DataFrame, coefs: pd.DataFrame, drv_new: pd.DataFrame,
                 msa: str | None, fundamental: str | None) -> str:
    """Top-2 drivers per fundamental, pooled across segments.

    Returns blocks mapping (fundamental, class, submarket) -> report lines
    for that segment.

    Ranking is per segment by |coef x driver| level contribution — drivers
    hit each subsector differently, so pooling across segments would mislead.
    Each segment's coefficients come from the model spec config final_models
    selected for it — the coefs file carries every candidate spec, so the
    lookup filters to the winner (pure AR specs have no regressor terms).

    Each driver is reported with its own quarter-over-quarter impulse in
    native units at the lagged date the model consumed — e.g. "+2.10M over
    Q4 2022 → Q1 2023 (12-quarter lag)". No coefficients appear in the report.
    Only the new vintage's driver file is needed: the R script writes the
    driver file from the current input CSV with no vintage truncation, so a
    vintage-vs-vintage driver comparison is structurally ~zero.
    """
    # same group grain and filters as _fof
    f = new.copy()
    if msa:
        f = f[f["msa_desc"].str.contains(msa, case=False, na=False)]
    if fundamental:
        f = f[f["offc_fundamental_desc"] == fundamental]
    else:
        f = f[f["offc_fundamental_desc"].isin(FUNDAMENTALS)]
    groups = f[["msa_desc", "offc_fundamental_desc",
                "class_desc", "submkt_desc"]].drop_duplicates()
    # model spec selected per segment in config final_models — the coefs file
    # carries every candidate spec, so filter to the winner before ranking.
    # a segment missing from final_models keeps its unfiltered coefficients.
    sel = {(r["offc_fundamental_desc"], r["class_desc"], r["submkt_desc"]): r["model"]
           for r in _config().get("final_models", [])}
    # pre-sort the driver frame per msa once so the per-term lookups stay cheap
    dn = {m: d.sort_values("q") for m, d in drv_new.groupby("msa_desc")}

    # per-segment |beta x| level contributions and beta*dq impulses —
    # no pooling across segments: drivers hit each subsector differently
    seg = {}
    for g in groups.itertuples():
        gm, gf, gc, gs = g.msa_desc, g.offc_fundamental_desc, g.class_desc, g.submkt_desc
        # first forecast quarter of the new vintage: the step the regressors feed
        nf = new[(new["msa_desc"] == gm) & (new["offc_fundamental_desc"] == gf) &
                 (new["class_desc"] == gc) & (new["submkt_desc"] == gs) &
                 (new["cast_desc"] == "forecast")]
        if nf.empty:
            continue
        F = nf["q"].min()
        # coefficients from the selected spec only
        cc = coefs[(coefs["msa_desc"] == gm) & (coefs["offc_fundamental_desc"] == gf) &
                   (coefs["class_desc"] == gc) & (coefs["submkt_desc"] == gs)]
        model = sel.get((gf, gc, gs))
        if model is not None:
            cc = cc[cc[".model"] == model]
        regs = _regressors(gf)
        if cc.empty or not regs:
            continue
        bmap = dict(zip(cc["term"].astype(str).str.strip(), cc["estimate"]))
        dm = dn.get(gm)
        if dm is None:
            continue
        # score each configured regressor by |coef x driver| at its regressor date
        for base, lag in regs.items():
            beta = bmap.get(f"{base}_lag{lag}")
            if beta is None or pd.isna(beta):
                continue
            # the regressor feeding F lives at t* = F - lag in the base series
            t_star = F - lag
            s = _base_series(dm, gm, base)
            if s is None or t_star not in s.index or (t_star - 1) not in s.index:
                continue
            x, x_prev = s.loc[t_star], s.loc[t_star - 1]
            if pd.isna(x) or pd.isna(x_prev):
                continue
            dq = x - x_prev
            # t* is metro-level so dating rarely differs across segments
            seg[(gm, gf, gc, gs, base)] = [abs(beta * x), beta * dq,
                                          (lag, dq, _qlabel(t_star - 1), _qlabel(t_star))]

    blocks = {}
    # emit the top 2 drivers per segment
    by_seg = {}
    for (gm, gf, gc, gs, base), (score, push, disp) in seg.items():
        by_seg.setdefault((gm, gf, gc, gs), []).append((score, base, push, disp))
    for (gm, gf, gc, gs), items in by_seg.items():
        items.sort(key=lambda t: t[0], reverse=True)
        blines = blocks.setdefault((gf, gc, gs), [])
        blines.append(f"  - {gm} / {gf} / {gc} {gs} drivers:")
        for _, base, push, (lag, dq, earlier, later) in items[:2]:
            move = f"+{_fmt(dq)}" if dq > 0 else _fmt(dq)
            # direction of this segment's impulse on the forecast: sign of
            # beta*dq. the coefficient values themselves never appear.
            direction = ("leaving the forecast little changed" if push == 0
                         else "pushing the forecast up" if push > 0
                         else "pushing the forecast down")
            blines.append(f"      {_display_name(base)}: {move} over {earlier} → {later} "
                         f"({lag}-quarter lag), {direction}")
    # segments with no drivers (pure specs, or no regressors configured)
    # get one explicit line so the write-up skips them
    for g in groups.itertuples():
        gm, gf, gc, gs = g.msa_desc, g.offc_fundamental_desc, g.class_desc, g.submkt_desc
        if (gf, gc, gs) not in blocks:
            blines = blocks.setdefault((gf, gc, gs), [])
            blines.append(f"  - {gm} / {gf} / {gc} {gs}: no drivers "
                         "(no regressors in the selected specification).")
    return blocks

###############################################################################################################
###### _VS_ACTUALS
###############################################################################################################
def _vs_actuals(old: pd.DataFrame, actuals,
                msa: str | None, fundamental: str | None) -> str:
    """Segment accuracy: latest actual vs prior vintage's forecast, plus streaks."""
    if actuals is None:
        return "Segment check: actuals unavailable (no report file in report_dir)."
    # join keys: match each actual to the prior vintage's forecast for it
    key = ["msa_desc", "class_desc", "submkt_desc", "offc_fundamental_desc", "q"]
    act = actuals[key + ["offc_fundamental_val"]]
    oldfc = old[old["cast_desc"] == "forecast"][key + ["offc_fundamental_val"]]
    m = act.merge(oldfc, on=key, suffixes=("_act", "_fc"), how="inner")
    # same filters as everywhere else
    if msa:
        m = m[m["msa_desc"].str.contains(msa, case=False, na=False)]
    if fundamental:
        m = m[m["offc_fundamental_desc"] == fundamental]
    else:
        m = m[m["offc_fundamental_desc"].isin(FUNDAMENTALS)]
    if m.empty:
        return "Segment check: no actuals yet overlap the previous vintage's forecasts."
    m["err"] = m["offc_fundamental_val_act"] - m["offc_fundamental_val_fc"]

    beats, misses, streaks = [], [], []
    # walk each segment newest-first
    for (gm, gf, gc, gs), g in m.groupby(
            ["msa_desc", "offc_fundamental_desc", "class_desc", "submkt_desc"]):
        g = g.sort_values("q", ascending=False).reset_index(drop=True)
        latest_q = g.loc[0, "q"]
        av, fv = g.loc[0, "offc_fundamental_val_act"], g.loc[0, "offc_fundamental_val_fc"]
        latest_err = av - fv
        # streak: consecutive quarters from the latest with the same miss/beat sign
        s = 1 if latest_err > 0 else (-1 if latest_err < 0 else 0)
        n = 0
        if s != 0:
            for e in g["err"]:
                if (e > 0 and s > 0) or (e < 0 and s < 0):
                    n += 1
                else:
                    break
        label = f"{gm} / {gf} / {gc} {gs}"
        if latest_err > 0:
            beats.append((latest_err, label, latest_q, av, fv, gf))
        elif latest_err < 0:
            misses.append((latest_err, label, latest_q, av, fv, gf))
        if n >= 2:
            streaks.append((n, s, label))
    lines = ["Segment check — latest actuals vs the prior vintage's forecasts:"]
    # biggest beats and misses at the latest quarter, with levels then deltas
    beats.sort(key=lambda b: b[0], reverse=True)
    misses.sort(key=lambda b: b[0])
    for err, label, q, av, fv, gf in beats[:3]:
        lines.append(f"  beat by the most: {label} {_vol(av, fv, gf)} ({_qlabel(q)})")
    for err, label, q, av, fv, gf in misses[:3]:
        lines.append(f"  missed by the most: {label} {_vol(av, fv, gf)} ({_qlabel(q)})")
    # longest streaks
    streaks.sort(key=lambda s: s[0], reverse=True)
    for n, s, label in streaks[:3]:
        direction = "above" if s > 0 else "below"
        lines.append(f"  streak: {label} {direction} forecast {n} straight quarters")
    return "\n".join(lines)

###############################################################################################################
###### ASSESS_FORECAST
###############################################################################################################
def assess_forecast(vintage: str | None = None,
                    msa: str | None = None,
                    fundamental: str | None = None) -> str:
    """Full assessment of one vintage against its predecessor by quarter label.

    vintage: which vintage to assess — "2022 Q1" finds "2022Q1" on disk.
    None = highest quarter label on disk. The comparison vintage is always
    the next-highest quarter label below it.

    The intended calling pattern is one call per fundamental
    (fundamental="total_inventory", then "total_delivery", ...): each call
    returns only that fundamental's numbers, and the agent writes one
    paragraph per call. An unfiltered call (no fundamental) returns every
    fundamental's block plus a CLOSING block with the cross-fundamental
    uncertainty ranking and segment accuracy, for ad-hoc questions.
    """
    # sync the forecast repo first, so the assessment reads the latest pushed CSVs
    sync_msg = sync_repo()
    if sync_msg.startswith("SYNC FAILED"):
        return f"ASSESSMENT ABORTED: {sync_msg} — check the repo remote."
    # bad values in the csvs (a missing date column, an unparseable quarter)
    # come back as a clean abort message instead of a raw traceback
    try:
        # list the vintages on disk, highest quarter label first
        vintages = _vintages("finalcast.offc_fund")
        # no vintages at all means the pipeline never ran, stop here
        if len(vintages) < 1:
            return "No finalcast CSVs found — run the pipeline first."
        # if the caller named a vintage, find its label on disk
        if vintage:
            # normalize the request to a quarter so "2022 Q1" matches "2022Q1"
            wanted = _q(vintage)
            # loop through the labels on disk to find the one with the same quarter
            new_v = None
            new_idx = None
            for i, label in enumerate(vintages):
                if _q(label) == wanted:
                    new_v = label
                    new_idx = i
                    break
            # no match, abort and list what's available
            if new_v is None:
                available = ", ".join(vintages)
                return (f"ASSESSMENT ABORTED: vintage {vintage!r} not found. "
                        f"Available vintages (highest quarter first): {available}")
        else:
            # no vintage named, use the highest quarter label on disk
            new_v = vintages[0]
            new_idx = 0
        # load the assessed vintage's forecasts and its coefficients
        # (coefs has no date column, so skip the date work for it)
        new = _load("finalcast.offc_fund", new_v)
        coefs = _load("coefs.offc_fund", new_v, parse_dates=False)
        # actuals come from the report file (finalcast files are forecast-only);
        # None when the file is missing — the rest of the assessment still runs
        actuals = _actuals(new_v)

        # start the report with the vintage header
        parts = [f"ASSESSMENT — vintage {new_v}"]
        # comparison sections need a predecessor by quarter label
        if new_idx + 1 < len(vintages):
            old_v = vintages[new_idx + 1]
            old = _load("finalcast.offc_fund", old_v)
            parts.append(f"Previous vintage for comparison: {old_v}")
            if actuals is None:
                parts.append("(actuals unavailable — no report file in report_dir; "
                             "jump-off and segment accuracy will be skipped)")
            # the report is pivoted by fundamental: each block holds that
            # fundamental's FoF lines then its driver lines, so the write-up
            # comes back one section per fundamental
            seg_order, seg_blocks, _unc_rows_filtered = _fof(new, old, msa, fundamental, actuals)
            # the closing (uncertainty ranking + segment check) is only built
            # for unfiltered calls — the per-fundamental loop ends after the
            # fifth paragraph, so per-fundamental responses skip it entirely
            unc_rows_all = None
            if fundamental is None:
                _, _, unc_rows_all = _fof(new, old, msa, None, actuals)
            drv_error = None
            attr_blocks = {}
            try:
                drv_new = _driver_vars(new_v)
                attr_blocks = _attribution(new, coefs, drv_new, msa, fundamental)
            except ValueError as e:
                drv_error = f"Drivers behind the change: unavailable — {e}"
            order = [gf for gf in FUNDAMENTALS if gf in seg_order]
            order += [gf for gf in seg_order if gf not in order]
            for gf in order:
                parts.append(f"--- {gf} ---")
                for (gc, gs) in seg_order[gf]:
                    parts.append(f"  -- {gc} {gs} --")
                    parts.extend(seg_blocks[(gf, gc, gs)])
                    parts.extend(attr_blocks.get((gf, gc, gs), []))
            if drv_error:
                parts.append(drv_error)
            if fundamental is None:
                # unfiltered call: include the cross-fundamental closing
                parts.append("--- CLOSING ---")
                parts.append(_unc_ranking(unc_rows_all))
                parts.append(_vs_actuals(old, actuals, msa, None))
        elif len(vintages) == 1:
            # only one vintage on disk, so there's nothing to compare against yet
            parts.append("Only one vintage on disk — run another vintage to get "
                         "forecast-over-forecast, attribution and accuracy.")
        else:
            # the assessed vintage is the oldest on disk, so there's no earlier one
            parts.append(f"{new_v} is the oldest vintage on disk — no earlier vintage "
                         "to compare against.")
        return "\n\n".join(parts)
    except ValueError as e:
        return f"ASSESSMENT ABORTED: {e}"
