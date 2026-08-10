# Source Notes and Reproducibility

## Source transcripts

The research protocol is a formalization of the author's arguments in:

1. `Episode #255 - Larry Lepard and the Paths to the Big Print.txt` (August 6, 2026)
2. `Market Report 8 9 2026.txt` (August 9, 2026)

The protocol preserves the following source-derived claims as hypotheses rather than established facts:

- Japanese monetary normalization is compressing a load-bearing carry structure.
- A roughly 30–35 bp German-Japanese residual is near a critical threshold.
- the Bank of England / ECB / European complex is the terminal weak link;
- U.S.–Japan support can force deleveraging while returning dollar liquidity;
- gold's appreciation can repair the collateral structure rather than merely signal fiat collapse;
- stablecoins and Treasury collateral may form a lower-leverage replacement rail.

The package adds external mathematical and accounting distinctions so those claims can be tested. These additions are not attributed to the transcripts.

## Public-data snapshots

The files in `data/raw/` were downloaded from official public endpoints and are included only so the parsers can be verified without network access. They are dated snapshots, not automatically refreshed datasets.

- `cftc_financial_futures_week_snapshot.txt`: CFTC Traders in Financial Futures, futures-only current-week text.
- `mof_week_snapshot_2026-08-09.csv`: Japan MOF weekly international securities transactions, last updated August 6, 2026 in the source file.
- `tic_slt_table3_snapshot.txt`: Treasury TIC SLT table 3, with data through May 2026 in this snapshot.

## Reproducibility rules

- Keep raw downloads unchanged in `raw_cache/`.
- Record every event date and sign prediction before running the event study.
- Do not overwrite the optional vendor-data file without preserving its source, timestamp, timezone and sign conventions.
- Treat a source outage or missing field as missing, never as zero.
- Report the strongest result against the thesis alongside the strongest result in its favor.
