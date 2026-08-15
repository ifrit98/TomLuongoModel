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

    python luongo_model_test_v2.py --offline-snapshots --output outputs/offline

Live public-data baseline (no API key required under normal source policies)::

    python luongo_model_test_v2.py --start 2006-01-01 --output outputs/live

Add candidate event dates and premium/custom market data::

    python luongo_model_test_v2.py \
        --events events_v2_template.csv \
        --optional-market-data optional_market_data_v2_template.csv \
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
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
USER_AGENT = "LuongoModelTest/2.0 (public-data research scaffold)"

FRED_GRAPH_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FRED_API_URL = "https://api.stlouisfed.org/fred/series/observations"
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
    "usd_per_gbp": "DEXUSUK",
    "chf_per_usd": "DEXSZUS",
    # FRED no longer serves the old LBMA fixing series through the API.
    # Use the current daily Credit Suisse/Nasdaq gold price index as a RETURN proxy.
    # This is not a spot-gold level and should not be interpreted as $/oz.
    "gold_usd": "NASDAQQGLDI",
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
    "fr10": "IRLTLT01FRM156N",
    "ch10": "IRLTLT01CHM156N",
    # Short-term interbank rates: the non-tautological carry-capacity layer
    "jp3m": "IR3TIB01JPM156N",
    "ch3m": "IR3TIB01CHM156N",
    "us3m": "IR3TIB01USM156N",
    "uk3m": "IR3TIB01GBM156N",
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


class HostUnavailable(SourceUnavailable):
    """Raised for timeout/connection/server failures that can affect a whole host."""


class ResourceUnavailable(SourceUnavailable):
    """Raised when one requested resource/series is rejected or does not exist."""


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
    """Fetch bytes with retries while keeping credentials out of logs.

    4xx errors such as an invalid FRED series are resource-level failures and
    must NOT trip the host circuit breaker. Timeouts, connection failures,
    rate limiting, and 5xx responses may be transient and are retried.
    """
    last_error: str | None = None
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )

            status = response.status_code
            if 200 <= status < 300:
                return response.content

            # Do not retry permanent client/resource errors. Crucially, do not
            # stringify requests.HTTPError because it includes response.url and
            # therefore API keys passed in the query string.
            if 400 <= status < 500 and status != 429:
                detail = response.text.strip().replace("\n", " ")[:300]
                LOGGER.warning(
                    "Fetch rejected (%s, HTTP %d)%s",
                    url,
                    status,
                    f": {detail}" if detail else "",
                )
                raise ResourceUnavailable(
                    f"HTTP {status} for {url}" + (f": {detail}" if detail else "")
                )

            # 429 and 5xx are potentially transient service/host failures.
            last_error = f"HTTP {status}"
            LOGGER.warning(
                "Fetch failed (%s, attempt %d/%d): HTTP %d",
                url,
                attempt + 1,
                retries,
                status,
            )
        except ResourceUnavailable:
            raise
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            LOGGER.warning(
                "Fetch failed (%s, attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                last_error,
            )
        except requests.RequestException as exc:
            # Other requests-layer failures are logged without the prepared URL.
            last_error = f"{type(exc).__name__}: {exc.__class__.__name__}"
            LOGGER.warning(
                "Fetch failed (%s, attempt %d/%d): %s",
                url,
                attempt + 1,
                retries,
                last_error,
            )

        if attempt + 1 < retries:
            time.sleep(2**attempt)

    raise HostUnavailable(f"Unable to fetch {url}: {last_error}")


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


def _pick_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first matching column name from a list of schema candidates.

    Public-data APIs occasionally alter capitalization or punctuation in field
    names.  Prefer exact matches, then fall back to a conservative normalized
    comparison so minor schema changes do not crash the run.
    """
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate

    def normalize(value: object) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")

    normalized = {normalize(column): column for column in frame.columns}
    for candidate in candidates:
        match = normalized.get(normalize(candidate))
        if match is not None:
            return match
    return None


def _parse_fred_graph_csv(content: bytes, alias: str, series_id: str) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content))
    if frame.shape[1] < 2:
        raise SourceUnavailable(f"Unexpected FRED graph schema for {series_id}")
    date_col, value_col = frame.columns[:2]
    frame = frame.rename(columns={date_col: "date", value_col: alias})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date").sort_index()[[alias]]


def _parse_fred_api_json(content: bytes, alias: str, series_id: str) -> pd.DataFrame:
    try:
        payload = json.loads(content.decode("utf-8"))
        observations = payload["observations"]
    except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise SourceUnavailable(f"Unexpected FRED API response for {series_id}: {exc}") from exc
    frame = pd.DataFrame(observations)
    if frame.empty or not {"date", "value"}.issubset(frame.columns):
        raise SourceUnavailable(f"No observations returned by FRED API for {series_id}")
    frame = frame.rename(columns={"value": alias})
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[alias] = pd.to_numeric(frame[alias], errors="coerce")
    return frame.dropna(subset=["date"]).set_index("date").sort_index()[[alias]]


def fetch_fred_series(
    alias: str,
    series_id: str,
    start: str,
    end: str,
    raw_dir: Path,
    *,
    api_key: str | None = None,
    source: str = "auto",
    timeout: int = 15,
    retries: int = 2,
) -> FetchResult:
    """Fetch one FRED series.

    Preferred path is the official FRED API when an API key is supplied.
    The no-key fredgraph CSV endpoint remains as a convenience fallback.
    On network failure, an existing raw cache is used if available.
    """
    cache_csv = raw_dir / f"fred_{series_id}.csv"
    errors: list[str] = []

    use_api = source == "api" or (source == "auto" and bool(api_key))
    use_graph = source == "graph" or (source == "auto" and not api_key)

    if use_api:
        if not api_key:
            raise SourceUnavailable(
                "FRED API selected but no API key supplied. Set FRED_API_KEY or use --fred-source graph."
            )
        params = {
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start,
            "observation_end": end,
            "sort_order": "asc",
            "limit": "100000",
        }
        try:
            content = request_bytes(
                FRED_API_URL, params=params, retries=retries, timeout=timeout
            )
            frame = _parse_fred_api_json(content, alias, series_id)
            # Save a normalized CSV so later reruns can proceed from cache.
            raw_dir.mkdir(parents=True, exist_ok=True)
            frame.reset_index().to_csv(cache_csv, index=False)
            return FetchResult(alias, frame, f"FRED API:{series_id}", cache_csv)
        except ResourceUnavailable as exc:
            # A 4xx response is specific to this series/request. It is not a
            # FRED host outage, so do not fall through to the graph host and do
            # not trip the host-level circuit breaker.
            errors.append(f"API resource: {exc}")
            if cache_csv.exists():
                frame = _parse_fred_graph_csv(cache_csv.read_bytes(), alias, series_id)
                LOGGER.warning("Using cached FRED %s after API resource failure", series_id)
                return FetchResult(alias, frame, f"FRED cache:{series_id}", cache_csv)
            raise
        except HostUnavailable as exc:
            errors.append(f"API host: {exc}")
            if source == "api":
                if cache_csv.exists():
                    frame = _parse_fred_graph_csv(cache_csv.read_bytes(), alias, series_id)
                    LOGGER.warning("Using cached FRED %s after API host failure", series_id)
                    return FetchResult(alias, frame, f"FRED cache:{series_id}", cache_csv)
                raise
            # In auto mode, only a host/network problem merits trying the
            # graph endpoint as an alternate transport.
            use_graph = True
        except SourceUnavailable as exc:
            errors.append(f"API parse: {exc}")
            if cache_csv.exists():
                frame = _parse_fred_graph_csv(cache_csv.read_bytes(), alias, series_id)
                LOGGER.warning("Using cached FRED %s after API parse failure", series_id)
                return FetchResult(alias, frame, f"FRED cache:{series_id}", cache_csv)
            raise

    if use_graph:
        params = {"id": series_id, "cosd": start, "coed": end}
        try:
            content = request_bytes(
                FRED_GRAPH_URL, params=params, retries=retries, timeout=timeout
            )
            frame = _parse_fred_graph_csv(content, alias, series_id)
            cached = write_raw_cache(raw_dir, f"fred_{series_id}.csv", content)
            return FetchResult(alias, frame, f"FRED graph:{series_id}", cached)
        except ResourceUnavailable as exc:
            errors.append(f"graph resource: {exc}")
        except HostUnavailable as exc:
            errors.append(f"graph host: {exc}")
            raise
        except SourceUnavailable as exc:
            errors.append(f"graph parse: {exc}")

    if cache_csv.exists():
        try:
            frame = _parse_fred_graph_csv(cache_csv.read_bytes(), alias, series_id)
            LOGGER.warning("Using cached FRED %s after network failure", series_id)
            return FetchResult(alias, frame, f"FRED cache:{series_id}", cache_csv)
        except Exception as exc:  # cache corruption should not hide the network error
            errors.append(f"cache: {exc}")

    raise SourceUnavailable("; ".join(errors) or f"Unable to fetch FRED {series_id}")


def fetch_all_fred(
    start: str,
    end: str,
    raw_dir: Path,
    *,
    api_key: str | None = None,
    source: str = "auto",
    timeout: int = 15,
    retries: int = 2,
    fail_fast: bool = True,
) -> tuple[pd.DataFrame, dict[str, str]]:
    frames: list[pd.DataFrame] = []
    status: dict[str, str] = {}
    fred_host_failed = False

    if source == "auto":
        effective = "official API" if api_key else "no-key graph CSV fallback"
        LOGGER.info("FRED source: %s", effective)
    elif source == "api":
        LOGGER.info("FRED source: official API")
    else:
        LOGGER.info("FRED source: graph CSV fallback")

    aliases = list(FRED_SERIES.items())
    for idx, (alias, series_id) in enumerate(aliases):
        if fred_host_failed:
            status[alias] = "skipped after FRED host/network failure"
            continue
        try:
            result = fetch_fred_series(
                alias,
                series_id,
                start,
                end,
                raw_dir,
                api_key=api_key,
                source=source,
                timeout=timeout,
                retries=retries,
            )
            frames.append(result.frame)
            status[alias] = f"loaded ({result.source})"
            LOGGER.info("Loaded FRED %s (%s)", alias, series_id)
        except ResourceUnavailable as exc:
            status[alias] = f"resource unavailable: {exc}"
            LOGGER.warning(
                "Skipping FRED %s (%s): series/request unavailable; continuing with remaining series",
                alias,
                series_id,
            )
        except HostUnavailable as exc:
            status[alias] = f"host unavailable: {exc}"
            LOGGER.warning("Skipping FRED %s after host/network failure: %s", alias, exc)
            if fail_fast:
                fred_host_failed = True
                remaining = len(aliases) - idx - 1
                if remaining:
                    LOGGER.warning(
                        "FRED circuit breaker opened after host/network failure; "
                        "skipping %d remaining FRED requests. Use --no-fred-fail-fast "
                        "only if you intentionally want every series retried.",
                        remaining,
                    )
        except SourceUnavailable as exc:
            status[alias] = f"unavailable: {exc}"
            LOGGER.warning(
                "Skipping FRED %s (%s): parse/data failure; continuing with remaining series",
                alias,
                series_id,
            )

    if not frames:
        return pd.DataFrame(), status
    return pd.concat(frames, axis=1).sort_index(), status


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
    """Build the daily research panel and derived FX/carry-state variables.

    V2 separates: (1) funding/carry state, (2) weak-link stress, and
    (3) policy-response signals. EUR/JPY=JPY per EUR and EUR/GBP=GBP per EUR.
    Tom's 180 and 0.85 levels are treated as hypotheses to test, not fitted rules.
    """
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
        # .where preserves missingness: a bare `<` comparison evaluates NaN to False,
        # which would silently code every weekend and holiday as "not below threshold".
        panel["eurjpy_below_180"] = (
            (panel["jpy_per_eur"] < 180.0).astype(float).where(panel["jpy_per_eur"].notna())
        )
        panel["eurjpy_distance_to_180_pct"] = (panel["jpy_per_eur"] / 180.0 - 1.0) * 100

    if {"usd_per_eur", "usd_per_gbp"}.issubset(panel.columns):
        # USD/EUR divided by USD/GBP = GBP per EUR, the conventional EUR/GBP quote.
        panel["gbp_per_eur"] = panel["usd_per_eur"] / panel["usd_per_gbp"]
        panel["eurgbp_return"] = np.log(panel["gbp_per_eur"]).diff()
        panel["gbp_appreciation_vs_eur"] = -panel["eurgbp_return"]
        panel["eurgbp_below_085"] = (
            (panel["gbp_per_eur"] < 0.85).astype(float).where(panel["gbp_per_eur"].notna())
        )
        panel["eurgbp_distance_to_085_pct"] = (panel["gbp_per_eur"] / 0.85 - 1.0) * 100

    if {"chf_per_usd", "usd_per_eur"}.issubset(panel.columns):
        panel["chf_per_eur"] = panel["chf_per_usd"] * panel["usd_per_eur"]
        panel["eurchf_return"] = np.log(panel["chf_per_eur"]).diff()
        panel["chf_appreciation_vs_eur"] = -panel["eurchf_return"]
    if {"jpy_per_usd", "chf_per_usd"}.issubset(panel.columns):
        panel["jpy_per_chf"] = panel["jpy_per_usd"] / panel["chf_per_usd"]

    if {"eurjpy_below_180", "eurgbp_below_085"}.issubset(panel.columns):
        # No fillna: the panel is calendar-indexed, so weekends and holidays carry no
        # FX observation. Coercing those to zero would file an unobserved day as
        # "not below threshold" and contaminate every regime count downstream.
        panel["threshold_breach_count"] = (
            panel["eurjpy_below_180"] + panel["eurgbp_below_085"]
        )
        panel["threshold_joint_breach"] = np.where(
            panel["threshold_breach_count"].isna(),
            np.nan,
            (panel["threshold_breach_count"] >= 2).astype(float),
        )

    if "gold_usd" in panel:
        panel["gold_return"] = np.log(panel["gold_usd"]).diff()
    if "broad_usd" in panel:
        panel["broad_usd_return"] = np.log(panel["broad_usd"]).diff()
    if {"sofr", "iorb"}.issubset(panel.columns):
        panel["sofr_minus_iorb_bp"] = (panel["sofr"] - panel["iorb"]) * 100
    if {"usdjpy_xccy_basis_bp", "eurusd_xccy_basis_bp"}.issubset(panel.columns):
        panel["dollar_basis_stress_proxy"] = -(
            panel["usdjpy_xccy_basis_bp"] + panel["eurusd_xccy_basis_bp"]
        ) / 2
    return panel

def build_monthly_panel(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    monthly = daily.resample("ME").last()
    flow_like = [
        "gold_return", "broad_usd_return", "eurjpy_return", "jpy_appreciation_vs_eur",
        "eurgbp_return", "gbp_appreciation_vs_eur", "eurchf_return", "chf_appreciation_vs_eur",
    ]
    for column in flow_like:
        if column in daily:
            monthly[column] = daily[column].resample("ME").sum(min_count=1)

    # Long-end diagnostic geometry. These are identities/diagnostics, not executable carry.
    pairs = {
        "de_jp_10y_bp": ("de10", "jp10"),
        "us_jp_10y_bp": ("us10", "jp10"),
        "us_de_10y_bp": ("us10", "de10"),
        "uk_jp_10y_bp": ("uk10", "jp10"),
        "us_uk_10y_bp": ("us10", "uk10"),
        "fr_bund_bp": ("fr10", "de10"),
        "fr_jp_10y_bp": ("fr10", "jp10"),
        "btp_bund_bp": ("it10", "de10"),
        "uk_bund_bp": ("uk10", "de10"),
        "ch_jp_10y_bp": ("ch10", "jp10"),
        "uk_ch_10y_bp": ("uk10", "ch10"),
        "us_ch_10y_bp": ("us10", "ch10"),
    }
    for name, (a, b) in pairs.items():
        if {a, b}.issubset(monthly.columns):
            monthly[name] = (monthly[a] - monthly[b]) * 100

    # Short-end funding/carry capacity. Positive CHF-vs-JPY advantage means CHF is cheaper funding.
    short_pairs = {
        "us_jp_3m_bp": ("us3m", "jp3m"),
        "uk_jp_3m_bp": ("uk3m", "jp3m"),
        "us_ch_3m_bp": ("us3m", "ch3m"),
        "uk_ch_3m_bp": ("uk3m", "ch3m"),
        "jp_ch_3m_bp": ("jp3m", "ch3m"),
    }
    for name, (a, b) in short_pairs.items():
        if {a, b}.issubset(monthly.columns):
            monthly[name] = (monthly[a] - monthly[b]) * 100
    if "jp_ch_3m_bp" in monthly:
        monthly["chf_funding_advantage_vs_jpy_bp"] = monthly["jp_ch_3m_bp"]

    # Tom's v2 ontology: France = candidate weak link; Italy = ECB intervention/fragmentation signal.
    if "btp_bund_bp" in monthly:
        delta = monthly["btp_bund_bp"].diff()
        rolling_mean = delta.rolling(24, min_periods=12).mean()
        rolling_std = delta.rolling(24, min_periods=12).std()
        monthly["btp_bund_change_z24"] = (delta - rolling_mean) / rolling_std.replace(0, np.nan)
        monthly["ecb_fragmentation_signal"] = monthly["btp_bund_change_z24"]

    # Better executable covered-carry proxy, only when premium/private inputs are supplied.
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


PRICE_LEVEL_VARS = {
    "jpy_per_eur", "gbp_per_eur", "chf_per_eur", "jpy_per_chf", "gold_usd", "broad_usd"
}


def _observed_window_change(series: pd.Series, event_date: pd.Timestamp, left: int, right: int, pct: bool) -> float:
    """Change over available observations, not calendar rows.

    For price/index levels, returns 100*log(end/start). For spreads/rates, returns end-start.
    """
    s = series.dropna().sort_index()
    if s.empty:
        return np.nan
    pos = s.index.searchsorted(event_date)
    if pos >= len(s):
        return np.nan
    i0 = pos + left
    i1 = pos + right
    # A window that runs off either end of the sample is unobserved, not truncated:
    # clamping would silently report a shorter horizon under the longer label.
    if i0 < 0 or i1 > len(s) - 1:
        return np.nan
    start = s.iloc[i0]
    end = s.iloc[i1]
    if pd.isna(start) or pd.isna(end):
        return np.nan
    if pct:
        if start <= 0 or end <= 0:
            return np.nan
        return float(100.0 * np.log(end / start))
    return float(end - start)


def run_event_study(
    daily: pd.DataFrame,
    events: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """Event study using each variable's actual observed dates.

    This fixes v1's calendar-row issue and avoids subtracting daily-return columns.
    """
    if daily.empty or events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    windows = {"m1_p1": (-1, 1), "d0_p5": (0, 5), "d0_p20": (0, 20)}
    for _, event in events.iterrows():
        event_date = pd.Timestamp(event["event_date"])
        row: dict[str, object] = {
            "event_date": event_date,
            "event_type": event.get("event_type", ""),
            "source_note": event.get("source_note", ""),
        }
        for variable in variables:
            if variable not in daily:
                continue
            pct = variable in PRICE_LEVEL_VARS
            suffix = "pct" if pct else "delta"
            for label, (left, right) in windows.items():
                row[f"{variable}__{label}_{suffix}"] = _observed_window_change(
                    daily[variable], event_date, left, right, pct
                )
        rows.append(row)
    return pd.DataFrame(rows)


def _below_threshold_indicator(series: pd.Series, threshold: float) -> pd.Series:
    """Below-threshold indicator that preserves NaN on unobserved dates.

    Same idiom as build_daily_panel's eurjpy_below_180 / eurgbp_below_085: a bare
    `<` comparison evaluates NaN to False, which would silently code every
    weekend/holiday as "not below threshold" and mis-state regime counts.
    """
    return (series < threshold).astype(float).where(series.notna())


def _threshold_transitions(series: pd.Series, threshold: float) -> pd.Series:
    """Below-state transitions on observed dates only: +1 = crossed below, -1 = crossed above.

    Operates on series.dropna() so unobserved (weekend/holiday) dates neither count as
    a state nor register a spurious crossing when the series resumes.
    """
    s = series.dropna().sort_index()
    below = s < threshold
    return below.astype(int).diff().dropna()


def extract_threshold_crossings(daily: pd.DataFrame) -> pd.DataFrame:
    specs = [("jpy_per_eur", 180.0, "EURJPY_180"), ("gbp_per_eur", 0.85, "EURGBP_085")]
    rows: list[dict[str, object]] = []
    for col, threshold, label in specs:
        if col not in daily:
            continue
        change = _threshold_transitions(daily[col], threshold)
        for dt, value in change.items():
            if value == 1:
                rows.append({"event_date": dt, "event_type": "threshold_crossing", "source_note": f"{label}: crossed below", "threshold": label, "direction": "below"})
            elif value == -1:
                rows.append({"event_date": dt, "event_type": "threshold_crossing", "source_note": f"{label}: crossed above", "threshold": label, "direction": "above"})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)


def threshold_regime_summary(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    if not {"eurjpy_below_180", "eurgbp_below_085"}.issubset(daily.columns):
        return pd.DataFrame()
    # Keep only days where BOTH thresholds are actually observed. The calendar-indexed
    # panel has ~2,150 weekend/holiday rows with no FX print; including them would
    # inflate the above-threshold buckets with non-observations.
    work = daily.dropna(subset=["eurjpy_below_180", "eurgbp_below_085"]).copy()
    if work.empty:
        return pd.DataFrame()
    a = work["eurjpy_below_180"].astype(int)
    b = work["eurgbp_below_085"].astype(int)
    # Tom's framing: the ECB is said to fear a DROP below these levels, so
    # "below" is the stressed state and "above" is the comfortable one.
    work["threshold_state"] = np.select(
        [(a == 0) & (b == 0), (a == 1) & (b == 0), (a == 0) & (b == 1), (a == 1) & (b == 1)],
        ["both_above", "eurjpy_below_only", "eurgbp_below_only", "both_below"], default="unknown"
    )
    # Tom's central question is what the ECB is afraid of below these levels, so the
    # European stress and intervention spreads have to be conditioned on the state too.
    # Those live at monthly frequency, so label each month by its dominant daily state.
    monthly_state = pd.Series(dtype=object)
    if not monthly.empty:
        monthly_state = (
            work["threshold_state"]
            .resample("ME")
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
        )
    european_cols = ["fr_bund_bp", "btp_bund_bp", "uk_bund_bp", "de_jp_10y_bp",
                     "chf_funding_advantage_vs_jpy_bp"]

    rows = []
    for state, g in work.groupby("threshold_state"):
        # Report the span each state occupies. These regimes are contiguous historical
        # episodes, not randomized draws, so any cross-state difference is confounded
        # with era; the reader needs to see that without opening the daily panel.
        row = {
            "threshold_state": state,
            "daily_observations": int(len(g)),
            "first_date": g.index.min().date().isoformat(),
            "last_date": g.index.max().date().isoformat(),
            "distinct_years": int(g.index.year.nunique()),
        }
        for c in ["vix", "hy_oas", "sofr_minus_iorb_bp"]:
            if c in g:
                row[f"median_{c}"] = float(g[c].median(skipna=True)) if g[c].notna().any() else np.nan
        for c in ["gold_return", "broad_usd_return"]:
            if c in g:
                row[f"mean_{c}_bp"] = float(g[c].mean(skipna=True) * 10000) if g[c].notna().any() else np.nan

        if not monthly_state.empty:
            months = monthly_state.index[monthly_state == state]
            gm = monthly.reindex(months)
            row["monthly_observations"] = int(len(gm))
            for c in european_cols:
                if c in gm and gm[c].notna().any():
                    row[f"median_{c}"] = float(gm[c].median(skipna=True))
                else:
                    row[f"median_{c}"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def placebo_threshold_tests(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """Do 180 (EUR/JPY) and 0.85 (EUR/GBP) behave any differently from neighbouring
    levels, or would any nearby threshold produce the same picture?

    Tests each pre-specified threshold against four placebo levels on either side,
    using the same NaN-preserving below/above indicator and crossing-transition
    logic as extract_threshold_crossings / threshold_regime_summary.
    """
    specs = [
        ("jpy_per_eur", [170.0, 175.0, 180.0, 185.0, 190.0], 180.0),
        ("gbp_per_eur", [0.83, 0.84, 0.85, 0.86, 0.87], 0.85),
    ]
    spread_cols = ["btp_bund_bp", "fr_bund_bp"]

    rows: list[dict[str, object]] = []
    for col, thresholds, prespecified in specs:
        if col not in daily:
            continue
        series = daily[col]
        vix = daily["vix"] if "vix" in daily else pd.Series(dtype=float, index=daily.index)
        for threshold in thresholds:
            below_ind = _below_threshold_indicator(series, threshold)
            # Restrict to days where the FX rate is actually observed; unobserved
            # (weekend/holiday) days must be excluded, not silently coded as "above".
            observed = below_ind.dropna()
            below_mask = observed == 1.0
            above_mask = observed == 0.0

            transitions = _threshold_transitions(series, threshold)
            below_dates = observed.index[below_mask]

            row: dict[str, object] = {
                "pair": col,
                "threshold": threshold,
                "is_prespecified": bool(threshold == prespecified),
                "days_below": int(below_mask.sum()),
                "days_above": int(above_mask.sum()),
                "crossings_below": int((transitions == 1).sum()),
                "crossings_above": int((transitions == -1).sum()),
                "first_date_below": below_dates.min().date().isoformat() if len(below_dates) else None,
                "last_date_below": below_dates.max().date().isoformat() if len(below_dates) else None,
            }

            vix_below = vix.reindex(observed.index[below_mask])
            vix_above = vix.reindex(observed.index[above_mask])
            row["median_vix_below"] = (
                float(vix_below.median(skipna=True)) if vix_below.notna().any() else np.nan
            )
            row["median_vix_above"] = (
                float(vix_above.median(skipna=True)) if vix_above.notna().any() else np.nan
            )

            # The spread columns are monthly-only, so label each month by its modal
            # daily state at this threshold, exactly as threshold_regime_summary does,
            # then take medians over the labelled months.
            monthly_state = pd.Series(dtype=object)
            if not monthly.empty:
                state = below_ind.map({1.0: "below", 0.0: "above"})
                monthly_state = state.resample("ME").agg(
                    lambda s: s.mode().iloc[0] if not s.dropna().empty and not s.mode().empty else np.nan
                )
            below_months = monthly_state.index[monthly_state == "below"] if not monthly_state.empty else pd.DatetimeIndex([])
            above_months = monthly_state.index[monthly_state == "above"] if not monthly_state.empty else pd.DatetimeIndex([])

            for spread_col in spread_cols:
                if spread_col in monthly.columns:
                    below_vals = monthly.reindex(below_months)[spread_col]
                    above_vals = monthly.reindex(above_months)[spread_col]
                    med_below = float(below_vals.median(skipna=True)) if below_vals.notna().any() else np.nan
                    med_above = float(above_vals.median(skipna=True)) if above_vals.notna().any() else np.nan
                else:
                    med_below = np.nan
                    med_above = np.nan
                row[f"median_{spread_col}_below"] = med_below
                row[f"median_{spread_col}_above"] = med_above
                row[f"{spread_col}_gap"] = (
                    med_below - med_above if pd.notna(med_below) and pd.notna(med_above) else np.nan
                )

            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["pair", "threshold"]).reset_index(drop=True)


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


def local_projection_btp_on_jpy(monthly: pd.DataFrame, max_horizon: int = 6) -> pd.DataFrame:
    """Jordà-style local projections of BTP-Bund on yen appreciation vs EUR.

    Negative horizons are a pre-trend diagnostic, not a "pre-trend check that
    passes": dep = y_{t+h} - y_{t-1}, so a negative coefficient at h<0 means
    the level *before* t-1 was lower than the t-1 baseline, i.e. the spread
    was already rising into the shock. h=-1 is a normalization point, not an
    estimate: dep = y_{t-1} - y_{t-1} is identically zero by construction, so
    that row is emitted with coef=0.0 (true, structural) but std_err/z/
    pvalue/ci_low/ci_high/nobs all NaN and estimable=False, so it cannot be
    mistaken for a tight, significant null. This is a deliberate, documented
    deviation from a literal reading of the brief's dep-var formula, made
    because a fully-populated zero-CI row at h=-1 would misrepresent the
    normalization point as the most confident finding in the chart.
    """
    required = {"btp_bund_bp", "jpy_appreciation_vs_eur", "vix"}
    if sm is None or not required.issubset(monthly.columns):
        return pd.DataFrame()

    work = monthly.copy()
    work["d_vix"] = work["vix"].diff()
    regressor_sd = float(work["jpy_appreciation_vs_eur"].std(skipna=True))

    rows: list[dict[str, object]] = []
    for h in range(-max_horizon, max_horizon + 1):
        if h == -1:
            # dep = y_{t-1} - y_{t-1} is identically zero by construction: this
            # is the normalization point of the local projection, not a null
            # estimate. Emitting a fitted std_err=0 / ci width=0 row here would
            # render as the single most authoritative point in the chart,
            # sitting exactly on zero one month before the shock -- precisely
            # where a pre-trend claim would be made or refuted. Mark it
            # explicitly as not estimable instead.
            rows.append(
                {
                    "horizon": h,
                    "coef": 0.0,
                    "std_err": float("nan"),
                    "z": float("nan"),
                    "pvalue": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "nobs": float("nan"),
                    "coef_per_1sd_bp": 0.0,
                    "estimable": False,
                }
            )
            continue

        dep = work["btp_bund_bp"].shift(-h) - work["btp_bund_bp"].shift(1)
        frame = pd.DataFrame(
            {
                "dep": dep,
                "jpy_appreciation_vs_eur": work["jpy_appreciation_vs_eur"],
                "d_vix": work["d_vix"],
            }
        ).dropna()
        if len(frame) < 30:
            LOGGER.info(
                "Skipping local projection horizon %d: only %d usable observations.",
                h, len(frame),
            )
            continue

        y = frame["dep"]
        x = sm.add_constant(frame[["jpy_appreciation_vs_eur", "d_vix"]])
        maxlags = max(3, abs(h) + 1)
        model = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})

        coef = float(model.params["jpy_appreciation_vs_eur"])
        std_err = float(model.bse["jpy_appreciation_vs_eur"])
        z = float(model.tvalues["jpy_appreciation_vs_eur"])
        pvalue = float(model.pvalues["jpy_appreciation_vs_eur"])
        ci_low, ci_high = model.conf_int().loc["jpy_appreciation_vs_eur"]

        rows.append(
            {
                "horizon": h,
                "coef": coef,
                "std_err": std_err,
                "z": z,
                "pvalue": pvalue,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "nobs": int(len(frame)),
                # coef is bp per 1.0 (i.e. 100%) log-return move in
                # jpy_appreciation_vs_eur; rescale to bp per 1 in-sample
                # standard deviation of the regressor, which is the size of
                # move that actually occurs month to month.
                "coef_per_1sd_bp": coef * regressor_sd,
                "estimable": True,
            }
        )

    if not rows:
        return pd.DataFrame()
    columns = [
        "horizon", "coef", "std_err", "z", "pvalue", "ci_low", "ci_high",
        "nobs", "coef_per_1sd_bp", "estimable",
    ]
    return pd.DataFrame(rows)[columns].sort_values("horizon").reset_index(drop=True)


def run_regressions(monthly: pd.DataFrame, output_dir: Path) -> None:
    reports: dict[str, str] = {}
    work = monthly.copy()
    for column in [
        "fr_bund_bp", "btp_bund_bp", "uk_bund_bp", "de_jp_10y_bp", "uk_jp_10y_bp",
        "us_jp_10y_bp", "us_uk_10y_bp", "chf_funding_advantage_vs_jpy_bp",
        "vix", "us10", "gold_usd", "broad_usd", "us10_real", "hy_oas", "fed_swap_lines",
    ]:
        if column in work:
            work[f"d_{column}"] = work[column].diff()

    # France is the candidate continental weak link in v2.
    reports["france_weak_link_on_carry_proxies"] = hac_regression(
        work,
        "d_fr_bund_bp",
        ["d_de_jp_10y_bp", "jpy_appreciation_vs_eur", "gbp_appreciation_vs_eur", "d_vix", "d_us10"],
    )
    # Italy/BTP-Bund is treated as a fragmentation/intervention-state signal, not the victim.
    reports["ecb_fragmentation_signal"] = hac_regression(
        work,
        "d_btp_bund_bp",
        ["d_fr_bund_bp", "d_uk_bund_bp", "jpy_appreciation_vs_eur", "gbp_appreciation_vs_eur", "d_vix"],
    )
    # Same as above but excluding France to isolate the yen coefficient.
    reports["ecb_fragmentation_signal_ex_france"] = hac_regression(
        work,
        "d_btp_bund_bp",
        ["d_uk_bund_bp", "jpy_appreciation_vs_eur", "gbp_appreciation_vs_eur", "d_vix"],
    )
    # US/UK/JP long-end triangle: diagnostic only; shared-yield algebra creates mechanical linkage.
    reports["uk_vs_us_relative_long_end"] = hac_regression(
        work,
        "d_us_uk_10y_bp",
        ["jpy_appreciation_vs_eur", "gbp_appreciation_vs_eur", "d_vix"],
    )
    # Does cheaper CHF funding expand as JPY funding normalizes?
    reports["chf_substitute_carry_capacity"] = hac_regression(
        work,
        "d_chf_funding_advantage_vs_jpy_bp",
        ["jpy_appreciation_vs_eur", "gbp_appreciation_vs_eur", "d_vix"],
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
    """Transparent v2 status sheet. No fake probability score."""
    rows: list[dict[str, str]] = []
    def add(test: str, status: str, evidence: str, next_needed: str) -> None:
        rows.append({"test": test, "status": status, "evidence_from_current_run": evidence, "next_data_or_decision": next_needed})

    if "de_jp_10y_bp" in monthly and monthly["de_jp_10y_bp"].notna().any():
        latest = monthly["de_jp_10y_bp"].dropna().iloc[-1]
        add("Raw German-Japanese 10Y spread is thin", "DESCRIPTIVE ONLY", f"Latest monthly proxy: {latest:.1f} bp.", "Add forwards/basis/repo; long-end spread is not executable carry.")
    else:
        add("Raw German-Japanese 10Y spread is thin", "NOT RUN", "Required yield series unavailable.", "Restore FRED/OECD series.")

    if "covered_carry_ann_pct" in monthly and monthly["covered_carry_ann_pct"].notna().any():
        latest = monthly["covered_carry_ann_pct"].dropna().iloc[-1]
        add("All-in covered carry approaches zero", "PRESSURED" if latest <= 0.25 else "POSITIVE", f"Latest reconstructed carry: {latest:.2f}% annualized.", "Validate signs and institution-specific hedge terms.")
    else:
        add("All-in covered carry approaches zero", "CRITICAL DATA GAP", "No executable basis/asset-return/funding-cost series supplied.", "Populate optional market-data CSV.")

    if "fr_bund_bp" in monthly and monthly["fr_bund_bp"].notna().any():
        latest = monthly["fr_bund_bp"].dropna().iloc[-1]
        add("France is the continental weak link", "PARTIAL PROXY", f"Latest OAT-Bund proxy: {latest:.1f} bp.", "Add French sovereign/bank CDS, OAT repo specialness, auction and wholesale-funding data.")
    else:
        add("France is the continental weak link", "NOT RUN", "No OAT-Bund proxy.", "Fetch France 10Y and premium French stress data.")

    if "btp_bund_bp" in monthly and monthly["btp_bund_bp"].notna().any():
        latest = monthly["btp_bund_bp"].dropna().iloc[-1]
        z = monthly.get("btp_bund_change_z24", pd.Series(dtype=float)).dropna()
        ztxt = f"; latest 24m change-z {z.iloc[-1]:.2f}" if not z.empty else ""
        add("BTP-Bund signals ECB fragmentation/intervention pressure", "STATE PROXY", f"Latest BTP-Bund: {latest:.1f} bp{ztxt}.", "Map spikes to PEPP/TPI/reinvestment and Bund/OAT/BTP flows; do not treat Italy as the weak link.")
    else:
        add("BTP-Bund signals ECB fragmentation/intervention pressure", "NOT RUN", "No BTP-Bund proxy.", "Restore Italy/Germany yields.")

    if "uk_bund_bp" in monthly and monthly["uk_bund_bp"].notna().any():
        latest = monthly["uk_bund_bp"].dropna().iloc[-1]
        add("UK/London is a separate weak-link channel", "PARTIAL PROXY", f"Latest UK-Bund proxy: {latest:.1f} bp.", "Add gilt repo, pension LDI, SONIA/basis, bank CDS and dealer funding data.")

    if "chf_funding_advantage_vs_jpy_bp" in monthly and monthly["chf_funding_advantage_vs_jpy_bp"].notna().any():
        latest = monthly["chf_funding_advantage_vs_jpy_bp"].dropna().iloc[-1]
        add("CHF/SNB substitutes for lost JPY carry capacity", "PUBLIC SHORT-RATE PROXY", f"CHF funding cheaper than JPY by {latest:.1f} bp on latest common 3m observation." if latest > 0 else f"CHF funding advantage proxy: {latest:.1f} bp.", "Add CHF FX forwards/cross-currency basis and actual SNB/SARON funding terms.")
    else:
        add("CHF/SNB substitutes for lost JPY carry capacity", "NOT RUN", "Short-rate pair incomplete.", "Fetch JP/CH 3m rates and CHF basis.")

    if {"jpy_per_eur", "gbp_per_eur"}.issubset(daily.columns):
        x = daily[["jpy_per_eur", "gbp_per_eur"]].dropna()
        if not x.empty:
            last=x.iloc[-1]
            add("EUR/JPY 180 and EUR/GBP 0.85 are regime boundaries", "EXPLICIT HYPOTHESIS", f"Latest common FX: EUR/JPY {last['jpy_per_eur']:.2f}; EUR/GBP {last['gbp_per_eur']:.4f}.", "Use crossing studies, local projections, and intervention-state conditioning; thresholds must earn predictive power out of sample.")

    if not cftc.empty and "lev_net_share_oi" in cftc:
        latest = cftc["lev_net_share_oi"].dropna().iloc[-1]
        add("Leveraged funds reduce short-yen exposure after shocks", "PROXY AVAILABLE", f"Latest net share of OI: {latest:.3f}; history required for event inference.", "Use full CFTC history; OTC book remains missing.")
    else:
        add("Leveraged funds reduce short-yen exposure after shocks", "NOT RUN", "No usable CFTC series.", "Fetch CFTC TFF history.")

    if not mof.empty:
        latest = mof.dropna(how="all").iloc[-1]
        val = latest.get("resident_foreign_total_net_100m_yen", np.nan)
        add("Japanese capital repatriates", "FLOW PROXY AVAILABLE", f"Latest resident foreign-security net flow: {val:,.0f} x ¥100m.", "Make sign convention explicit and decompose by security/country/currency.")

    if {"gold_return", "broad_usd_return", "hy_oas"}.issubset(daily.columns):
        sample = daily[["gold_return", "broad_usd_return", "hy_oas"]].dropna().copy(); sample["d_hy"] = sample["hy_oas"].diff()
        repair = sample[(sample["gold_return"] > 0) & (sample["broad_usd_return"] >= 0) & (sample["d_hy"] < 0)]
        add("Gold trades as system-repair collateral", "JOINT-SIGN PROXY", f"Repair-sign days: {len(repair)/max(1,len(sample)):.1%} of complete daily sample.", "Condition on identified interventions and use real spot/futures gold data.")

    add("The transition is deliberately engineered", "NOT IDENTIFIED BY PRICE DATA", "Market sequencing cannot establish intent.", "Require documents, disclosed operations, or institution-specific transaction evidence.")
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
    local_projection: pd.DataFrame | None = None,
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

    stress_cols = [c for c in ["fr_bund_bp", "btp_bund_bp", "uk_bund_bp"] if c in monthly]
    if stress_cols:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        monthly[stress_cols].plot(ax=ax)
        ax.set_title("V2: France weak-link, Italy intervention-state, UK/London stress proxies")
        ax.set_ylabel("Basis points")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "03_europe_stress_proxies.png")

    if {"chf_funding_advantage_vs_jpy_bp", "de_jp_10y_bp"}.issubset(monthly.columns):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        monthly[["chf_funding_advantage_vs_jpy_bp", "de_jp_10y_bp"]].plot(ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_title("CHF substitution versus German-Japanese long-end compression")
        ax.set_ylabel("Basis points")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "04_chf_substitution_capacity.png")

    if {"jpy_per_eur", "gbp_per_eur"}.issubset(daily.columns):
        joined = daily[["jpy_per_eur", "gbp_per_eur"]].dropna()
        if not joined.empty:
            fig, ax1 = plt.subplots(figsize=(10, 5.5))
            ax2 = ax1.twinx()
            ax1.plot(joined.index, joined["jpy_per_eur"], label="EUR/JPY")
            ax2.plot(joined.index, joined["gbp_per_eur"], linestyle="--", label="EUR/GBP")
            ax1.axhline(180.0, linewidth=1, linestyle=":")
            ax2.axhline(0.85, linewidth=1, linestyle=":")
            ax1.set_title("Tom's hypothesized FX regime boundaries")
            ax1.set_ylabel("JPY per EUR (threshold 180)")
            ax2.set_ylabel("GBP per EUR (threshold 0.85)")
            ax1.set_xlabel("")
            save_plot(fig, chart_dir / "05_fx_thresholds.png")

    if {"chf_funding_advantage_vs_jpy_bp", "de_jp_10y_bp"}.issubset(monthly.columns):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        monthly[["chf_funding_advantage_vs_jpy_bp", "de_jp_10y_bp"]].plot(ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_title("CHF substitution versus German-Japanese long-end compression")
        ax.set_ylabel("Basis points")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "04_chf_substitution_capacity.png")

    if {"jpy_per_eur", "gbp_per_eur"}.issubset(daily.columns):
        joined = daily[["jpy_per_eur", "gbp_per_eur"]].dropna()
        if not joined.empty:
            fig, ax1 = plt.subplots(figsize=(10, 5.5))
            ax2 = ax1.twinx()
            ax1.plot(joined.index, joined["jpy_per_eur"], label="EUR/JPY")
            ax2.plot(joined.index, joined["gbp_per_eur"], linestyle="--", label="EUR/GBP")
            ax1.axhline(180.0, linewidth=1, linestyle=":")
            ax2.axhline(0.85, linewidth=1, linestyle=":")
            ax1.set_title("Tom's hypothesized FX regime boundaries")
            ax1.set_ylabel("JPY per EUR (threshold 180)")
            ax2.set_ylabel("GBP per EUR (threshold 0.85)")
            ax1.set_xlabel("")
            save_plot(fig, chart_dir / "05_fx_thresholds.png")

    if not cftc.empty and "lev_net_share_oi" in cftc:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        cftc["lev_net_share_oi"].plot(ax=ax)
        ax.axhline(0, linewidth=1)
        ax.set_title("CFTC leveraged-fund Japanese-yen net position")
        ax.set_ylabel("Net contracts / open interest")
        ax.set_xlabel("")
        save_plot(fig, chart_dir / "06_cftc_jpy_positioning.png")

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
            save_plot(fig, chart_dir / "07_mof_repatriation_proxy.png")

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
            save_plot(fig, chart_dir / "08_tic_selected_residences.png")

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
            save_plot(fig, chart_dir / "09_gold_dollar_quadrants.png")

    if local_projection is None:
        local_projection = local_projection_btp_on_jpy(monthly)
    if not local_projection.empty:
        lp = local_projection.sort_values("horizon")
        estimable = lp["estimable"].astype(bool)
        fig, ax = plt.subplots(figsize=(10, 5.5))
        # Coefficient path stays continuous through h=-1 (coef=0.0 there is a
        # true structural value, not an estimate), but the CI band uses
        # ci_low/ci_high which are NaN at h=-1 -- matplotlib breaks both the
        # line and the fill at NaN, so the band shows a gap rather than
        # pinching to a false zero-width interval at that point.
        ax.plot(lp["horizon"], lp["coef"], linewidth=1.5, color="tab:blue", zorder=2)
        ax.scatter(
            lp.loc[estimable, "horizon"], lp.loc[estimable, "coef"],
            marker="o", color="tab:blue", zorder=3,
        )
        ax.scatter(
            lp.loc[~estimable, "horizon"], lp.loc[~estimable, "coef"],
            marker="o", facecolors="white", edgecolors="tab:blue",
            linewidths=1.5, s=60, zorder=4,
        )
        for _, row in lp.loc[~estimable].iterrows():
            ax.annotate(
                "normalization (0 by construction)",
                xy=(row["horizon"], row["coef"]),
                xytext=(row["horizon"], row["coef"] - 0.12 * max(1.0, lp["coef"].abs().max())),
                ha="center", va="top", fontsize=8, color="tab:blue",
            )
        ax.fill_between(
            lp["horizon"], lp["ci_low"], lp["ci_high"],
            alpha=0.2, color="tab:blue",
        )
        ax.axhline(0, linewidth=1)
        ax.axvline(0, linewidth=1, linestyle="--")
        ax.set_title("Local projection: BTP-Bund response to yen appreciation vs EUR")
        ax.set_ylabel("BTP-Bund response, bp per 1.0 (100%) JPY/EUR log-return appreciation")
        ax.set_xlabel("Months from yen appreciation")
        save_plot(fig, chart_dir / "10_local_projection_btp_on_jpy.png")


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    source_status: Mapping[str, str],
) -> None:
    # Outputs are shared with collaborators, so never serialize the API key.
    safe_arguments = dict(vars(args))
    if safe_arguments.get("fred_api_key"):
        safe_arguments["fred_api_key"] = "<redacted>"

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "arguments": safe_arguments,
        "source_status": dict(source_status),
        "important_interpretation": [
            "Raw sovereign-yield spreads are diagnostics, not executable carry.",
            "Covered carry requires forward/basis/funding/repo/cost inputs.",
            "Price sequencing cannot by itself identify policymaker intent.",
            "TIC is primarily residence/custody based and does not always reveal ultimate owners.",
            "V2 treats France as candidate continental weak link and BTP-Bund as an ECB fragmentation/intervention-state signal.",
            "EUR/JPY 180 and EUR/GBP 0.85 are hypotheses to falsify, not pre-validated boundaries.",
            "Threshold convention follows Tom's question: a DROP BELOW 180/0.85 is the feared state, so 'below' is coded as the stressed regime and 'above' as the comfortable one.",
            "Regime states use only days where both FX rates actually printed; weekends and holidays are excluded rather than treated as above-threshold.",
            "Threshold regimes are contiguous historical episodes, not randomized samples: the both-above state is confined to 2025-11 onward, so cross-state differences in French and Italian spreads are confounded with era and are not causal evidence.",
            "US/UK/JP long-end spreads are algebraically linked; nontrivial carry tests require short rates, FX and basis.",
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

    fred, fred_status = fetch_all_fred(
        args.start,
        args.end,
        raw_dir,
        api_key=args.fred_api_key,
        source=args.fred_source,
        timeout=args.fred_timeout,
        retries=args.fred_retries,
        fail_fast=args.fred_fail_fast,
    )
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
        "jpy_per_eur", "gbp_per_eur", "chf_per_eur", "jpy_per_chf",
        "gold_usd", "broad_usd", "sofr_minus_iorb_bp", "hy_oas", "vix",
        "eurjpy_xccy_basis_bp", "eurgbp_xccy_basis_bp", "usdchf_xccy_basis_bp",
        "fr_sovereign_cds_bp", "fr_bank_cds_bp", "uk_bank_cds_bp",
        "bund_repo_specialness_bp", "oat_repo_specialness_bp", "gilt_repo_specialness_bp",
    ]
    event_study = run_event_study(daily, events, event_variables)
    event_study.to_csv(output_dir / "event_study.csv", index=False)

    crossings = extract_threshold_crossings(daily)
    crossings.to_csv(output_dir / "threshold_crossings.csv", index=False)
    if not crossings.empty:
        crossing_study = run_event_study(daily, crossings, event_variables)
    else:
        crossing_study = pd.DataFrame()
    crossing_study.to_csv(output_dir / "threshold_crossing_event_study.csv", index=False)
    threshold_regime_summary(daily, monthly).to_csv(output_dir / "threshold_regime_summary.csv", index=False)

    placebo = placebo_threshold_tests(daily, monthly)
    placebo.to_csv(output_dir / "placebo_threshold_tests.csv", index=False)
    LOGGER.info("Wrote placebo threshold tests (%d rows) to %s", len(placebo), output_dir / "placebo_threshold_tests.csv")

    scorecard = create_scorecard(daily, monthly, cftc, mof)
    scorecard.to_csv(output_dir / "falsification_scorecard.csv", index=False)
    run_regressions(monthly, output_dir)

    local_projection = local_projection_btp_on_jpy(monthly)
    if not local_projection.empty:
        local_projection.to_csv(output_dir / "local_projection_btp_on_jpy.csv", index=False)
        LOGGER.info(
            "Wrote local projection of BTP-Bund on yen appreciation (%d rows) to %s",
            len(local_projection), output_dir / "local_projection_btp_on_jpy.csv",
        )
    else:
        LOGGER.warning(
            "Local projection of BTP-Bund on yen appreciation produced no rows "
            "(missing columns or statsmodels unavailable); skipping CSV write."
        )

    plot_outputs(daily, monthly, cftc, mof, tic, output_dir, local_projection=local_projection)
    write_metadata(output_dir, args, source_status)

    LOGGER.info("Live baseline complete. Review %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="V2: test the Luongo carry-regime model with France, UK/London, CHF substitution, and FX thresholds."
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
        "--fred-api-key",
        default=os.getenv("FRED_API_KEY"),
        help="FRED API key. Defaults to the FRED_API_KEY environment variable.",
    )
    parser.add_argument(
        "--fred-source",
        choices=["auto", "api", "graph"],
        default="auto",
        help="FRED transport: auto prefers official API when a key is available; otherwise graph CSV.",
    )
    parser.add_argument(
        "--fred-timeout",
        type=int,
        default=15,
        help="Read timeout in seconds for each FRED request (default: 15).",
    )
    parser.add_argument(
        "--fred-retries",
        type=int,
        default=2,
        help="Attempts per FRED request (default: 2).",
    )
    parser.add_argument(
        "--no-fred-fail-fast",
        dest="fred_fail_fast",
        action="store_false",
        help="Retry every FRED series even after the first host/network failure.",
    )
    parser.set_defaults(fred_fail_fast=True)
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
