#!/usr/bin/env python3
"""Runnable public-data scaffold for testing the Luongo yen-carry / collateral thesis.

The program intentionally separates three levels of evidence:

1. Raw diagnostics: sovereign yield spreads and EUR/JPY.
2. Better public proxies: CFTC yen positioning, Japanese portfolio flows,
   Treasury International Capital holdings, funding/liquidity indicators.
3. The decisive but usually premium/private inputs: cross-currency basis,
   executable hedge costs, repo/haircuts, bank CDS, and institution-level books.

It does not "prove" the thesis from endpoints alone.  It tests the required
causal sequence and reports missing links rather than filling them in.

Examples
--------
Offline parser smoke test using the bundled snapshots::

    python 03_luongo_model_test.py --offline-snapshots --output outputs/offline

Live public-data baseline (no API key required under normal source policies)::

    python 03_luongo_model_test.py --start 2006-01-01 --output outputs/live

Add candidate event dates and premium/custom market data::

    python 03_luongo_model_test.py \
        --events 05_events_template.csv \
        --optional-market-data 06_optional_market_data_template.csv \
        --output outputs/full

Research use only.  Source schemas and availability can change; every fetch is
cached and every failure is logged so the analysis remains auditable.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

try:
    import statsmodels.api as sm
except ImportError:  # pragma: no cover - handled at runtime
    sm = None


LOGGER = logging.getLogger("luongo_test")
USER_AGENT = "LuongoModelTest/1.0 (public-data research scaffold)"

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
CFTC_TFF_URL = "https://publicreporting.cftc.gov/resource/gpe5-46if.csv"
MOF_WEEKLY_URL = (
    "https://www.mof.go.jp/policy/international_policy/reference/"
    "itn_transactions_in_securities/week.csv"
)
TIC_TABLE3_URL = (
    "https://ticdata.treasury.gov/resource-center/data-chart-center/tic/"
    "Documents/slt_table3.txt"
)


# FRED series are deliberately listed in one place so Tom can add or replace
# proxies without touching the analysis functions.
FRED_SERIES: dict[str, str] = {
    # FX and hard collateral
    "jpy_per_usd": "DEXJPUS",
    "usd_per_eur": "DEXUSEU",
    "gold_usd": "GOLDAMGBD228NLBM",
    "broad_usd": "DTWEXBGS",
    # U.S. rates, liquidity and risk
    "us10": "DGS10",
    "us10_real": "DFII10",
    "us10_breakeven": "T10YIE",
    "sofr": "SOFR",
    "iorb": "IORB",
    "fed_assets": "WALCL",
    "m2": "M2SL",
    "fed_swap_lines": "SWPT",
    "vix": "VIXCLS",
    "hy_oas": "BAMLH0A0HYM2",
    # Monthly OECD long-term government-bond yields
    "jp10": "IRLTLT01JPM156N",
    "de10": "IRLTLT01DEM156N",
    "it10": "IRLTLT01ITM156N",
    "uk10": "IRLTLT01GBM156N",
    # ECB policy proxy
    "ecb_deposit_rate": "ECBDFR",
}


@dataclass(frozen=True)
class FetchResult:
    name: str
    frame: pd.DataFrame
    source: str
    cached_path: Path | None = None


class SourceUnavailable(RuntimeError):
    """Raised when a public source cannot be fetched or parsed."""


def configure_logging(output_dir: Path, verbose: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(output_dir / "run.log", encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def request_bytes(
    url: str,
    *,
    params: Mapping[str, str] | None = None,
    retries: int = 4,
    timeout: int = 45,
) -> bytes:
    """Fetch bytes with retries and an explicit user agent."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            sleep_s = 2**attempt
            LOGGER.warning(
                "Fetch failed (%s, attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                exc,
            )
            if attempt + 1 < retries:
                time.sleep(sleep_s)
    raise SourceUnavailable(f"Unable to fetch {url}: {last_error}")


def write_raw_cache(raw_dir: Path, name: str, content: bytes) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / name
    path.write_bytes(content)
    return path


def clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(".", "", regex=False),
        errors="coerce",
    )


def fetch_fred_series(
    alias: str,
    series_id: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> FetchResult:
    params = {"id": series_id, "cosd": start, "coed": end}
    content = request_bytes(FRED_GRAPH_URL, params=params)
    cached = write_raw_cache(raw_dir, f"fred_{series_id}.csv", content)
    frame = pd.read_csv(io.BytesIO(content))
    if frame.shape[1] < 2:
        raise SourceUnavailable(f"Unexpected FRED schema for {series_id}")
    date_col, value_col = frame.columns[:2]
    frame = frame.rename(columns={date_col: "date", value_col: alias})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return FetchResult(alias, frame[[alias]], f"FRED:{series_id}", cached)


def fetch_all_fred(
    start: str,
    end: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    status: dict[str, str] = {}
    for alias, series_id in FRED_SERIES.items():
        try:
            result = fetch_fred_series(alias, series_id, start, end, raw_dir)
            frames.append(result.frame)
            status[alias] = "downloaded"
            LOGGER.info("Downloaded FRED %s (%s)", alias, series_id)
        except SourceUnavailable as exc:
            status[alias] = f"unavailable: {exc}"
            LOGGER.warning("Skipping FRED %s: %s", alias, exc)
    if not frames:
        return pd.DataFrame(), status
    panel = pd.concat(frames, axis=1).sort_index()
    return panel, status


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {c.lower(): c for c in frame.columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]
    return None


def fetch_cftc_jpy(start: str, raw_dir: Path) -> FetchResult:
    # The Socrata TFF futures-only dataset normally accepts this no-token query.
    # Pull all rows from the start date, then filter market names locally to make
    # the code resilient to minor naming changes.
    params = {
        "$limit": "50000",
        "$order": "report_date_as_yyyy_mm_dd asc",
        "$where": f"report_date_as_yyyy_mm_dd >= '{start}T00:00:00.000'",
    }
    content = request_bytes(CFTC_TFF_URL, params=params)
    cached = write_raw_cache(raw_dir, "cftc_tff_futures_only.csv", content)
    frame = pd.read_csv(io.BytesIO(content), low_memory=False)

    market_col = _pick_column(frame, ["market_and_exchange_names"])
    date_col = _pick_column(
        frame,
        ["report_date_as_yyyy_mm_dd", "report_date_as_yyyy_mm_dd_1"],
    )
    if market_col is None or date_col is None:
        raise SourceUnavailable("Unexpected CFTC TFF schema")

    jpy = frame[
        frame[market_col].astype(str).str.contains("JAPANESE YEN", case=False, na=False)
    ].copy()
    if jpy.empty:
        raise SourceUnavailable("No Japanese yen row in CFTC TFF response")

    col_map = {
        "open_interest": ["open_interest_all"],
        "dealer_long": ["dealer_positions_long_all", "dealer_positions_long"],
        "dealer_short": ["dealer_positions_short_all", "dealer_positions_short"],
        "asset_mgr_long": ["asset_mgr_positions_long_all", "asset_mgr_positions_long"],
        "asset_mgr_short": ["asset_mgr_positions_short_all", "asset_mgr_positions_short"],
        "lev_long": ["lev_money_positions_long_all", "lev_money_positions_long"],
        "lev_short": ["lev_money_positions_short_all", "lev_money_positions_short"],
    }
    out = pd.DataFrame(index=pd.to_datetime(jpy[date_col], errors="coerce"))
    for alias, candidates in col_map.items():
        column = _pick_column(jpy, candidates)
        if column is not None:
            out[alias] = pd.to_numeric(jpy[column], errors="coerce").to_numpy()
    out = out[~out.index.isna()].sort_index()
    if {"lev_long", "lev_short"}.issubset(out.columns):
        out["lev_net"] = out["lev_long"] - out["lev_short"]
    if {"lev_net", "open_interest"}.issubset(out.columns):
        out["lev_net_share_oi"] = out["lev_net"] / out["open_interest"]
    return FetchResult("cftc_jpy", out, "CFTC:TFF futures-only", cached)


def parse_cftc_week_snapshot(path: Path) -> pd.DataFrame:
    """Parse the bundled current-week TFF text as an offline smoke test.

    The raw text has no header.  The positions below follow the official TFF
    futures-only layout.  This snapshot provides only one observation; live
    historical analysis uses the Socrata endpoint above.
    """
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = csv.reader(handle)
        for row in rows:
            if row and "JAPANESE YEN -" in row[0].upper():
                values = [item.strip() for item in row]
                try:
                    obs = {
                        "date": pd.to_datetime(values[2]),
                        "open_interest": float(values[7]),
                        "dealer_long": float(values[8]),
                        "dealer_short": float(values[9]),
                        "asset_mgr_long": float(values[11]),
                        "asset_mgr_short": float(values[12]),
                        "lev_long": float(values[14]),
                        "lev_short": float(values[15]),
                    }
                except (IndexError, ValueError) as exc:
                    raise SourceUnavailable(f"Unable to parse CFTC snapshot: {exc}") from exc
                frame = pd.DataFrame([obs]).set_index("date")
                frame["lev_net"] = frame["lev_long"] - frame["lev_short"]
                frame["lev_net_share_oi"] = frame["lev_net"] / frame["open_interest"]
                return frame
    raise SourceUnavailable("Japanese yen row not found in CFTC snapshot")


def _parse_mof_period_end(value: object) -> pd.Timestamp:
    text = str(value).strip()
    if not re.match(r"^\d{4}[．.]", text):
        return pd.NaT
    normalized = (
        text.replace("．", ".")
        .replace("～", "~")
        .replace("〜", "~")
        .replace(" ", "")
    )
    match = re.match(
        r"^(?P<year>\d{4})\.(?P<m1>\d{1,2})\.(?P<d1>\d{1,2})~"
        r"(?:(?P<m2>\d{1,2})\.)?(?P<d2>\d{1,2})$",
        normalized,
    )
    if not match:
        return pd.NaT
    year = int(match.group("year"))
    m1 = int(match.group("m1"))
    m2 = int(match.group("m2") or m1)
    d2 = int(match.group("d2"))
    # Weeks that cross Dec/Jan use the start year in the file.
    end_year = year + 1 if m1 == 12 and m2 == 1 else year
    try:
        return pd.Timestamp(date(end_year, m2, d2))
    except ValueError:
        return pd.NaT


def parse_mof_weekly_bytes(content: bytes) -> pd.DataFrame:
    decoded = content.decode("cp932", errors="replace")
    raw = pd.read_csv(io.StringIO(decoded), header=None, dtype=str)
    if raw.shape[1] < 23:
        raise SourceUnavailable("Unexpected MOF weekly CSV schema")
    dates = raw.iloc[:, 0].map(_parse_mof_period_end)
    data = raw.loc[dates.notna()].copy()
    data.index = pd.DatetimeIndex(dates[dates.notna()], name="date")

    # Unit: 100 million yen.  These are the most relevant net-flow columns.
    selected = pd.DataFrame(index=data.index)
    selected["resident_foreign_equity_net_100m_yen"] = clean_numeric(data.iloc[:, 3])
    selected["resident_foreign_long_debt_net_100m_yen"] = clean_numeric(data.iloc[:, 6])
    selected["resident_foreign_short_debt_net_100m_yen"] = clean_numeric(data.iloc[:, 10])
    selected["resident_foreign_total_net_100m_yen"] = clean_numeric(data.iloc[:, 11])
    selected["nonresident_japan_equity_net_100m_yen"] = clean_numeric(data.iloc[:, 14])
    selected["nonresident_japan_long_debt_net_100m_yen"] = clean_numeric(data.iloc[:, 17])
    selected["nonresident_japan_short_debt_net_100m_yen"] = clean_numeric(data.iloc[:, 21])
    selected["nonresident_japan_total_net_100m_yen"] = clean_numeric(data.iloc[:, 22])
    return selected.sort_index()


def fetch_mof_weekly(raw_dir: Path) -> FetchResult:
    content = request_bytes(MOF_WEEKLY_URL)
    cached = write_raw_cache(raw_dir, "mof_weekly.csv", content)
    frame = parse_mof_weekly_bytes(content)
    return FetchResult("mof_weekly", frame, "Japan MOF weekly securities flows", cached)


def parse_tic_table3_bytes(content: bytes) -> pd.DataFrame:
    decoded = content.decode("utf-8", errors="replace")
    lines = decoded.splitlines()
    header_index = None
    for i, line in enumerate(lines):
        if line.lower().startswith("country\tcountry_code\tdate"):
            header_index = i
            break
    if header_index is None:
        raise SourceUnavailable("TIC machine-readable header not found")
    frame = pd.read_csv(io.StringIO("\n".join(lines[header_index:])), sep="\t")
    expected = {"country", "date", "for_treas_pos", "for_treas_net"}
    if not expected.issubset(frame.columns):
        raise SourceUnavailable("Unexpected TIC table 3 schema")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m", errors="coerce")
    for column in frame.columns:
        if column not in {"country", "country_code", "date"}:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values(["country", "date"])


def fetch_tic_table3(raw_dir: Path) -> FetchResult:
    content = request_bytes(TIC_TABLE3_URL)
    cached = write_raw_cache(raw_dir, "tic_slt_table3.txt", content)
    frame = parse_tic_table3_bytes(content)
    return FetchResult("tic_table3", frame, "U.S. Treasury TIC SLT table 3", cached)


def load_optional_market_data(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError("Optional market-data CSV requires a 'date' column")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def build_daily_panel(fred: pd.DataFrame, optional: pd.DataFrame) -> pd.DataFrame:
    if fred.empty and optional.empty:
        return pd.DataFrame()
    panel = fred.copy()
    if not optional.empty:
        panel = panel.join(optional, how="outer")
    panel = panel.sort_index()

    if {"jpy_per_usd", "usd_per_eur"}.issubset(panel.columns):
        panel["jpy_per_eur"] = panel["jpy_per_usd"] * panel["usd_per_eur"]
        panel["eurjpy_return"] = np.log(panel["jpy_per_eur"]).diff()
        panel["jpy_appreciation_vs_eur"] = -panel["eurjpy_return"]
    if "gold_usd" in panel:
        panel["gold_return"] = np.log(panel["gold_usd"]).diff()
    if "broad_usd" in panel:
        panel["broad_usd_return"] = np.log(panel["broad_usd"]).diff()
    if {"sofr", "iorb"}.issubset(panel.columns):
        panel["sofr_minus_iorb_bp"] = (panel["sofr"] - panel["iorb"]) * 100
    if {"usdjpy_xccy_basis_bp", "eurusd_xccy_basis_bp"}.issubset(panel.columns):
        # A simple directional stress proxy.  Users should harmonize signs to
        # their data vendor's convention before interpreting levels.
        panel["dollar_basis_stress_proxy"] = -(
            panel["usdjpy_xccy_basis_bp"] + panel["eurusd_xccy_basis_bp"]
        ) / 2
    return panel


def build_monthly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    monthly = daily.resample("ME").last()
    flow_like = [
        "gold_return",
        "broad_usd_return",
        "eurjpy_return",
        "jpy_appreciation_vs_eur",
    ]
    for column in flow_like:
        if column in daily:
            monthly[column] = daily[column].resample("ME").sum(min_count=1)
    if {"de10", "jp10"}.issubset(monthly.columns):
        monthly["de_jp_10y_bp"] = (monthly["de10"] - monthly["jp10"]) * 100
    if {"us10", "jp10"}.issubset(monthly.columns):
        monthly["us_jp_10y_bp"] = (monthly["us10"] - monthly["jp10"]) * 100
    if {"us10", "de10"}.issubset(monthly.columns):
        monthly["us_de_10y_bp"] = (monthly["us10"] - monthly["de10"]) * 100
    if {"it10", "de10"}.issubset(monthly.columns):
        monthly["btp_bund_bp"] = (monthly["it10"] - monthly["de10"]) * 100
    if {"uk10", "de10"}.issubset(monthly.columns):
        monthly["uk_bund_bp"] = (monthly["uk10"] - monthly["de10"]) * 100

    # Better executable covered-carry proxy, available only when the user adds
    # cross-currency basis and European total-return/funding data.
    required = {"europe_asset_return_ann_pct", "eur_funding_ann_pct", "eurjpy_xccy_basis_bp"}
    if required.issubset(monthly.columns):
        costs = monthly.get("all_in_costs_ann_bp", 0.0)
        monthly["covered_carry_ann_pct"] = (
            monthly["europe_asset_return_ann_pct"]
            - monthly["eur_funding_ann_pct"]
            + monthly["eurjpy_xccy_basis_bp"] / 100
            - costs / 100
        )
    return monthly


def load_events(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    events = pd.read_csv(path, comment="#")
    if events.empty:
        return events
    if "event_date" not in events.columns:
        raise ValueError("Events CSV requires an 'event_date' column")
    events["event_date"] = pd.to_datetime(events["event_date"], errors="coerce")
    return events.dropna(subset=["event_date"]).sort_values("event_date")


def _nearest_position(index: pd.DatetimeIndex, target: pd.Timestamp) -> int | None:
    if index.empty:
        return None
    pos = index.searchsorted(target)
    if pos >= len(index):
        return len(index) - 1
    return int(pos)


def run_event_study(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    if daily.empty or events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    windows = {"m1_p1": (-1, 1), "d0_p5": (0, 5), "d0_p20": (0, 20)}
    panel = daily.sort_index()
    for _, event in events.iterrows():
        pos = _nearest_position(panel.index, pd.Timestamp(event["event_date"]))
        if pos is None:
            continue
        row: dict[str, object] = {
            "event_date": event["event_date"],
            "event_type": event.get("event_type", ""),
            "source_note": event.get("source_note", ""),
        }
        for variable in variables:
            if variable not in panel:
                continue
            series = panel[variable]
            for label, (left, right) in windows.items():
                i0 = max(0, pos + left)
                i1 = min(len(series) - 1, pos + right)
                start_value = series.iloc[i0]
                end_value = series.iloc[i1]
                row[f"{variable}__{label}"] = (
                    float(end_value - start_value)
                    if pd.notna(start_value) and pd.notna(end_value)
                    else np.nan
                )
        rows.append(row)
    return pd.DataFrame(rows)


def hac_regression(
    frame: pd.DataFrame,
    dependent: str,
    independents: Sequence[str],
    maxlags: int = 3,
) -> str:
    if sm is None:
        return "statsmodels is not installed; regression skipped."
    columns = [dependent, *independents]
    missing = [column for column in columns if column not in frame]
    if missing:
        return f"Skipped: missing columns {missing}"
    data = frame[columns].dropna()
    if len(data) < max(30, len(independents) * 8):
        return f"Skipped: only {len(data)} complete observations."
    y = data[dependent]
    x = sm.add_constant(data[list(independents)])
    model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return model.summary().as_text()


def run_regressions(monthly: pd.DataFrame, output_dir: Path) -> None:
    reports: dict[str, str] = {}
    work = monthly.copy()
    for column in [
        "btp_bund_bp",
        "de_jp_10y_bp",
        "vix",
        "us10",
        "gold_usd",
        "broad_usd",
        "us10_real",
        "hy_oas",
        "fed_swap_lines",
    ]:
        if column in work:
            work[f"d_{column}"] = work[column].diff()

    reports["europe_stress_on_carry_proxies"] = hac_regression(
        work,
        "d_btp_bund_bp",
        ["d_de_jp_10y_bp", "jpy_appreciation_vs_eur", "d_vix", "d_us10"],
    )
    reports["gold_repair_vs_debasement"] = hac_regression(
        work,
        "gold_return",
        ["broad_usd_return", "d_us10_real", "d_hy_oas", "d_fed_swap_lines"],
    )
    with (output_dir / "regression_report.txt").open("w", encoding="utf-8") as handle:
        for title, report in reports.items():
            handle.write("=" * 88 + "\n")
            handle.write(title + "\n")
            handle.write("=" * 88 + "\n")
            handle.write(report + "\n\n")


def zscore(series: pd.Series) -> pd.Series:
    std = series.std(skipna=True)
    if not std or pd.isna(std):
        return series * np.nan
    return (series - series.mean(skipna=True)) / std


def create_scorecard(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    cftc: pd.DataFrame,
    mof: pd.DataFrame,
) -> pd.DataFrame:
    """Create a transparent status sheet, not a fake probability model."""
    rows: list[dict[str, str]] = []

    def add(test: str, status: str, evidence: str, next_needed: str) -> None:
        rows.append(
            {
                "test": test,
                "status": status,
                "evidence_from_current_run": evidence,
                "next_data_or_decision": next_needed,
            }
        )

    if "de_jp_10y_bp" in monthly and monthly["de_jp_10y_bp"].notna().any():
        latest = monthly["de_jp_10y_bp"].dropna().iloc[-1]
        add(
            "Raw German-Japanese 10Y spread is thin",
            "DESCRIPTIVE ONLY",
            f"Latest monthly proxy: {latest:.1f} bp.",
            "Do not call this executable carry; add forward/basis/repo inputs.",
        )
    else:
        add(
            "Raw German-Japanese 10Y spread is thin",
            "NOT RUN",
            "Required yield series unavailable.",
            "Restore FRED/OECD series or provide vendor data.",
        )

    if "covered_carry_ann_pct" in monthly and monthly["covered_carry_ann_pct"].notna().any():
        latest = monthly["covered_carry_ann_pct"].dropna().iloc[-1]
        status = "PRESSURED" if latest <= 0.25 else "POSITIVE"
        add(
            "All-in covered carry approaches zero",
            status,
            f"Latest reconstructed carry: {latest:.2f}% annualized.",
            "Validate sign conventions and institution-specific hedge terms.",
        )
    else:
        add(
            "All-in covered carry approaches zero",
            "CRITICAL DATA GAP",
            "No basis/asset-return/funding-cost series supplied.",
            "Populate optional market-data CSV with executable inputs.",
        )

    if not cftc.empty and "lev_net_share_oi" in cftc:
        latest = cftc["lev_net_share_oi"].dropna().iloc[-1]
        add(
            "Leveraged funds reduce short-yen exposure after shocks",
            "PROXY AVAILABLE",
            f"Latest net share of OI: {latest:.3f}; event inference requires history.",
            "Run live CFTC history and event study; CFTC is not the full OTC book.",
        )
    else:
        add(
            "Leveraged funds reduce short-yen exposure after shocks",
            "NOT RUN",
            "No usable CFTC series.",
            "Fetch CFTC TFF history or provide positioning data.",
        )

    if "btp_bund_bp" in monthly and monthly["btp_bund_bp"].notna().any():
        latest = monthly["btp_bund_bp"].dropna().iloc[-1]
        add(
            "Europe bears more stress than U.S./Japan",
            "PARTIAL PROXY",
            f"Latest BTP-Bund proxy: {latest:.1f} bp.",
            "Add bank CDS, funding spreads, and comparable U.S./Japan stress indices.",
        )
    else:
        add(
            "Europe bears more stress than U.S./Japan",
            "NOT RUN",
            "No European fragmentation proxy.",
            "Provide sovereign and bank stress series.",
        )

    if not mof.empty:
        latest = mof.dropna(how="all").iloc[-1]
        val = latest.get("resident_foreign_total_net_100m_yen", np.nan)
        add(
            "Japanese capital repatriates",
            "FLOW PROXY AVAILABLE",
            f"Latest resident foreign-security net flow: {val:,.0f} x ¥100m.",
            "Decompose by country/currency and test around BOJ surprises.",
        )
    else:
        add(
            "Japanese capital repatriates",
            "NOT RUN",
            "No MOF flow data.",
            "Fetch MOF weekly portfolio flow series.",
        )

    if {"gold_return", "broad_usd_return", "hy_oas"}.issubset(daily.columns):
        sample = daily[["gold_return", "broad_usd_return", "hy_oas"]].dropna().copy()
        sample["d_hy"] = sample["hy_oas"].diff()
        repair = sample[
            (sample["gold_return"] > 0)
            & (sample["broad_usd_return"] >= 0)
            & (sample["d_hy"] < 0)
        ]
        share = len(repair) / max(1, len(sample))
        add(
            "Gold trades as system-repair collateral",
            "JOINT-SIGN PROXY",
            f"Repair-sign days: {share:.1%} of complete daily sample.",
            "Use identified intervention windows; unconditional frequency is not causal proof.",
        )
    else:
        add(
            "Gold trades as system-repair collateral",
            "NOT RUN",
            "Gold/USD/credit-spread joint panel incomplete.",
            "Restore FRED series or provide vendor data.",
        )

    add(
        "The transition is deliberately engineered",
        "NOT IDENTIFIED BY PRICE DATA",
        "Market sequencing cannot establish intent.",
        "Require documents, disclosed operations, or institution-specific transaction evidence.",
    )
    return pd.DataFrame(rows)


def save_plot(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_outputs(
    daily: pd.DataFrame,
    monthly: pd.DataFrame,
    cftc: pd.DataFrame,
    mof: pd.DataFrame,
    tic: pd.DataFrame,
    output_dir: Path,
) -> None:
    chart_dir = output_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    spread_cols = [c for c in ["de_jp_10y_bp", "us_jp_10y_bp", "us_de_10y_bp"] if c in monthly]
    if spread_cols:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        monthly[spread_cols].plot(ax=ax)
        ax.set_title("Sovereign-yield diagnostics (not executable carry)")
        ax.set_ylabel("Basis points")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "01_yield_spread_diagnostics.png")

    if {"jpy_per_eur", "de_jp_10y_bp"}.issubset(set(daily.columns) | set(monthly.columns)):
        joined = pd.concat(
            [daily["jpy_per_eur"].resample("ME").last(), monthly["de_jp_10y_bp"]], axis=1
        ).dropna()
        if not joined.empty:
            fig, ax1 = plt.subplots(figsize=(10, 5.5))
            ax2 = ax1.twinx()
            ax1.plot(joined.index, joined["jpy_per_eur"], label="JPY per EUR")
            ax2.plot(joined.index, joined["de_jp_10y_bp"], linestyle="--", label="DE-JP 10Y")
            ax1.set_title("EUR/JPY and raw German-Japanese yield spread")
            ax1.set_ylabel("JPY per EUR")
            ax2.set_ylabel("Basis points")
            ax1.set_xlabel("")
            save_plot(fig, chart_dir / "02_eurjpy_vs_dejp_spread.png")

    stress_cols = [c for c in ["btp_bund_bp", "uk_bund_bp"] if c in monthly]
    if stress_cols:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        monthly[stress_cols].plot(ax=ax)
        ax.set_title("European stress proxies")
        ax.set_ylabel("Basis points")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "03_europe_stress_proxies.png")

    if not cftc.empty and "lev_net_share_oi" in cftc:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        cftc["lev_net_share_oi"].plot(ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_title("CFTC leveraged-fund Japanese-yen net position")
        ax.set_ylabel("Net contracts / open interest")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "04_cftc_jpy_positioning.png")

    if not mof.empty:
        cols = [
            c
            for c in [
                "resident_foreign_long_debt_net_100m_yen",
                "resident_foreign_total_net_100m_yen",
            ]
            if c in mof
        ]
        if cols:
            fig, ax = plt.subplots(figsize=(10, 5.5))
            mof[cols].rolling(4).sum().plot(ax=ax)
            ax.axhline(0, linewidth=1)
            ax.set_title("Japan MOF: four-week resident foreign-security flows")
            ax.set_ylabel("100 million yen")
            ax.set_xlabel("")
            save_plot(fig, chart_dir / "05_mof_repatriation_proxy.png")

    if not tic.empty:
        selected_names = [
            "Japan",
            "United Kingdom",
            "Belgium",
            "Luxembourg",
            "Ireland",
            "Cayman Islands",
            "Switzerland",
            "China, Mainland",
        ]
        subset = tic[tic["country"].isin(selected_names)]
        if not subset.empty:
            pivot = subset.pivot(index="date", columns="country", values="for_treas_pos")
            fig, ax = plt.subplots(figsize=(10, 5.5))
            pivot.tail(72).plot(ax=ax)
            ax.set_title("TIC: Treasury holdings by residence (custody caveat applies)")
            ax.set_ylabel("USD millions")
            ax.set_xlabel("")
            save_plot(fig, chart_dir / "06_tic_selected_residences.png")

    if {"gold_return", "broad_usd_return"}.issubset(daily.columns):
        sample = daily[["gold_return", "broad_usd_return"]].dropna()
        if len(sample) > 20:
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(sample["broad_usd_return"], sample["gold_return"], s=12, alpha=0.45)
            ax.axhline(0, linewidth=1)
            ax.axvline(0, linewidth=1)
            ax.set_title("Gold return versus broad-dollar return")
            ax.set_xlabel("Broad-dollar log return")
            ax.set_ylabel("Gold log return")
            save_plot(fig, chart_dir / "07_gold_dollar_quadrants.png")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    source_status: Mapping[str, str],
) -> None:
    metadata = {
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "arguments": vars(args),
        "source_status": dict(source_status),
        "important_interpretation": [
            "Raw sovereign-yield spreads are diagnostics, not executable carry.",
            "Covered carry requires forward/basis/funding/repo/cost inputs.",
            "Price sequencing cannot by itself identify policymaker intent.",
            "TIC is primarily residence/custody based and does not always reveal ultimate owners.",
        ],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )


def offline_snapshot_run(package_dir: Path, output_dir: Path) -> None:
    raw_dir = package_dir / "data" / "raw"
    LOGGER.info("Running offline snapshot parser checks")

    cftc_path = raw_dir / "cftc_financial_futures_week_snapshot.txt"
    mof_path = raw_dir / "mof_week_snapshot_2026-08-09.csv"
    tic_path = raw_dir / "tic_slt_table3_snapshot.txt"

    cftc = parse_cftc_week_snapshot(cftc_path)
    mof = parse_mof_weekly_bytes(mof_path.read_bytes())
    tic = parse_tic_table3_bytes(tic_path.read_bytes())

    cftc.to_csv(output_dir / "cftc_jpy_snapshot_parsed.csv")
    mof.tail(24).to_csv(output_dir / "mof_latest_24_weeks_parsed.csv")

    selected = tic[
        tic["country"].isin(
            [
                "Japan",
                "United Kingdom",
                "Belgium",
                "Luxembourg",
                "Ireland",
                "Cayman Islands",
                "Switzerland",
                "China, Mainland",
            ]
        )
    ]
    latest_date = selected["date"].max()
    selected[selected["date"] == latest_date].sort_values("for_treas_pos", ascending=False).to_csv(
        output_dir / "tic_selected_latest_parsed.csv", index=False
    )

    scorecard = create_scorecard(pd.DataFrame(), pd.DataFrame(), cftc, mof)
    scorecard.to_csv(output_dir / "offline_scorecard.csv", index=False)
    plot_outputs(pd.DataFrame(), pd.DataFrame(), cftc, mof, tic, output_dir)

    summary = {
        "cftc_rows": len(cftc),
        "mof_rows": len(mof),
        "tic_rows": len(tic),
        "tic_latest_date": str(latest_date.date()) if pd.notna(latest_date) else None,
    }
    (output_dir / "offline_smoke_test.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info("Offline parser checks passed: %s", summary)


def live_run(args: argparse.Namespace, package_dir: Path, output_dir: Path) -> None:
    raw_dir = output_dir / "raw_cache"
    source_status: dict[str, str] = {}

    fred, fred_status = fetch_all_fred(args.start, args.end, raw_dir)
    source_status.update({f"fred:{k}": v for k, v in fred_status.items()})

    try:
        cftc = fetch_cftc_jpy(args.start, raw_dir).frame
        source_status["cftc"] = "downloaded"
    except SourceUnavailable as exc:
        LOGGER.warning("CFTC unavailable: %s", exc)
        cftc = pd.DataFrame()
        source_status["cftc"] = f"unavailable: {exc}"

    try:
        mof = fetch_mof_weekly(raw_dir).frame
        source_status["mof"] = "downloaded"
    except SourceUnavailable as exc:
        LOGGER.warning("MOF unavailable: %s", exc)
        mof = pd.DataFrame()
        source_status["mof"] = f"unavailable: {exc}"

    try:
        tic = fetch_tic_table3(raw_dir).frame
        source_status["tic"] = "downloaded"
    except SourceUnavailable as exc:
        LOGGER.warning("TIC unavailable: %s", exc)
        tic = pd.DataFrame()
        source_status["tic"] = f"unavailable: {exc}"

    optional = load_optional_market_data(args.optional_market_data)
    source_status["optional_market_data"] = (
        "loaded" if not optional.empty else "not supplied or empty"
    )

    daily = build_daily_panel(fred, optional)
    monthly = build_monthly_panel(daily)
    events = load_events(args.events)

    daily.to_csv(output_dir / "daily_panel.csv")
    monthly.to_csv(output_dir / "monthly_panel.csv")
    cftc.to_csv(output_dir / "cftc_jpy_positioning.csv")
    mof.to_csv(output_dir / "mof_weekly_flows.csv")
    tic.to_csv(output_dir / "tic_table3.csv", index=False)

    event_variables = [
        "jpy_per_eur",
        "eurjpy_return",
        "gold_usd",
        "gold_return",
        "broad_usd",
        "broad_usd_return",
        "sofr_minus_iorb_bp",
        "hy_oas",
        "vix",
        "eurjpy_xccy_basis_bp",
        "euro_bank_cds_bp",
        "us_bank_cds_bp",
    ]
    event_study = run_event_study(daily, events, event_variables)
    event_study.to_csv(output_dir / "event_study.csv", index=False)

    scorecard = create_scorecard(daily, monthly, cftc, mof)
    scorecard.to_csv(output_dir / "falsification_scorecard.csv", index=False)
    run_regressions(monthly, output_dir)
    plot_outputs(daily, monthly, cftc, mof, tic, output_dir)
    write_metadata(output_dir, args, source_status)

    LOGGER.info("Live baseline complete. Review %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the Luongo yen-carry/collateral model with public and optional data."
    )
    parser.add_argument("--start", default="2006-01-01", help="Start date (YYYY-MM-DD).")
    parser.add_argument(
        "--end", default=date.today().isoformat(), help="End date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/run"), help="Output directory."
    )
    parser.add_argument(
        "--events", type=Path, default=None, help="Candidate event-date CSV."
    )
    parser.add_argument(
        "--optional-market-data",
        type=Path,
        default=None,
        help="Optional cross-currency basis/CDS/repo/return CSV.",
    )
    parser.add_argument(
        "--offline-snapshots",
        action="store_true",
        help="Parse bundled public snapshots without internet access.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir, args.verbose)
    package_dir = Path(__file__).resolve().parent

    try:
        if args.offline_snapshots:
            offline_snapshot_run(package_dir, output_dir)
        else:
            live_run(args, package_dir, output_dir)
    except (SourceUnavailable, ValueError, OSError) as exc:
        LOGGER.exception("Run failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
