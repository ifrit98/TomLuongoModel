# Luongo Yen-Carry / Collateral-Regime Testing Package

This package turns the model discussed in **Episode #255 with Larry Lepard (August 6, 2026)** and the **August 9, 2026 market report** into a falsifiable research protocol and a runnable public-data baseline.

It is designed to answer a stricter question than “did Europe weaken while gold rose?”:

> Did the market move through the specific causal chain the thesis requires?

The target sequence is:

```text
BOJ surprise / yen appreciation
  -> effective JPY-funded European carry falls
  -> relevant positions and dealer balance sheets contract
  -> Europe bears disproportionate financial stress
  -> capital shifts toward the U.S. and/or Japan
  -> policy support eases dollar-funding stress
  -> gold rises with the dollar firm and credit conditions stabilizing
```

Correct endpoints without the intermediate links are **not** treated as confirmation.

## Contents

- `From_Thesis_to_Test.docx` — the working paper and research protocol.
- `README.md` — this file.
- `luongo_model_test.py` — public-data collection, parsing, panels, charts, event study, regressions, and scorecard.
- `requirements.txt` — Python dependencies.
- `events_template.csv` — candidate event dates; Tom should verify, add, or delete before formal testing.
- `optional_market_data_template.csv` — schema for cross-currency basis, CDS, repo, actual return and funding inputs.
- `data_dictionary.csv` — variable definitions, sources, roles, and warnings.
- `data/raw/` — public snapshots used to verify the MOF, CFTC, and TIC parsers offline.
- `outputs_example/offline/` — parser outputs from the bundled snapshots. These are **not model results**.

## The key mathematical distinction

The transcript's raw diagnostic is approximately:

```text
(US 10Y - Japan 10Y) - (US 10Y - Germany 10Y)
= Germany 10Y - Japan 10Y
≈ 30–35 bp
```

That may be a useful alarm bell, but it is not the all-in trade.

For a fully hedged JPY-funded European position, using the sign convention in the paper:

```text
covered carry
≈ European asset total return
  - EUR funding rate
  + JPY/EUR cross-currency basis
  - repo / hedge / capital / transaction costs
```

Under covered interest parity, cheap JPY cash funding is largely offset by forward points. The mechanism therefore needs one or more of the following:

- partial or absent FX hedging;
- short-dated hedges rolled against long-dated assets;
- adverse cross-currency basis;
- dealer balance-sheet scarcity;
- repo, collateral, or margin pressure;
- duration or European-credit losses;
- or a mismatch between the legal entities holding the asset and the hedge.

The optional-data template is built around those decisive variables.

## Install

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## First run: verify the bundled parsers offline

```bash
python luongo_model_test.py \
  --offline-snapshots \
  --output outputs/offline
```

This checks the current-week CFTC TFF text, Japan MOF weekly flow CSV, and Treasury TIC table 3 snapshot. Expected output includes:

- parsed Japanese-yen CFTC positioning;
- the latest 24 weeks of Japanese securities flows;
- selected-country Treasury holdings;
- a deliberately incomplete scorecard marking the missing causal data.

## Public-data baseline

```bash
python luongo_model_test.py \
  --start 2006-01-01 \
  --events 05_events_template.csv \
  --output outputs/public_baseline
```

The live run attempts to download:

- FRED FX, rates, gold, dollar, liquidity and credit series;
- CFTC Traders in Financial Futures positioning for Japanese yen;
- Japan Ministry of Finance weekly international securities flows;
- Treasury International Capital table 3 holdings and flows.

No API key is normally required for these endpoints. Public sites can rate-limit, rename fields, or change schemas; failures are logged and do not silently become zeroes.

## Full test with basis, CDS, repo and actual trade inputs

Copy the optional template, populate it from Bloomberg, Refinitiv, ICE, CME, dealer runs, or another consistent vendor, and preserve the units and sign conventions:

```bash
cp optional_market_data_template.csv data/optional_market_data.csv
```

Then run:

```bash
python luongo_model_test.py \
  --start 2006-01-01 \
  --events 05_events_template.csv \
  --optional-market-data data/optional_market_data.csv \
  --output outputs/full_test
```

The minimum decisive inputs are:

```text
date
EUR/JPY cross-currency basis
European asset total return or a clearly specified bond/credit leg
matched EUR funding rate
all-in repo / hedge / capital costs
European and U.S. bank CDS or comparable funding stress
```

Before interpreting basis data, verify the vendor's sign convention. A sign error can reverse the economic conclusion.

## Main outputs

- `daily_panel.csv` and `monthly_panel.csv`
- `cftc_jpy_positioning.csv`
- `mof_weekly_flows.csv`
- `tic_table3.csv`
- `event_study.csv`
- `regression_report.txt`
- `falsification_scorecard.csv`
- `charts/*.png`
- `run_metadata.json`
- `run.log`

## What the first regression does

A basic monthly proxy regression asks whether changes in European fragmentation are associated with changes in the raw German–Japanese yield spread and yen appreciation, controlling for global volatility and U.S. rates:

```text
Δ(BTP - Bund)
  ~ Δ(Germany 10Y - Japan 10Y)
  + yen appreciation versus EUR
  + ΔVIX
  + ΔUS 10Y
```

This is a baseline, not causal proof. The paper proposes a stronger BOJ-surprise first stage and a mediated test:

```text
BOJ surprise -> reconstructed effective carry -> European stress
```

## What would support the thesis

The strongest result is an ordered, repeated sequence in which:

1. exogenous BOJ hawkish surprises appreciate the yen and reduce **all-in** carry;
2. the relevant short-yen / European-risk positions contract;
3. vulnerable European institutions and sovereigns weaken more than comparable U.S. and Japanese entities;
4. Japanese repatriation and Europe-to-U.S./Japan capital flows follow;
5. identified U.S.–Japan support operations reduce dollar-funding stress;
6. gold appreciates while the dollar remains firm and credit spreads stabilize.

## What would weaken or reject it

The core carry channel is weakened if:

- forward points and basis offset the raw yield differential;
- reconstructed all-in carry stays comfortably positive after BOJ shocks;
- the actual books are fully hedged with matched tenors and no collateral stress;
- positioning does not fall;
- U.S. or Japanese markets absorb more stress than Europe;
- no capital-flow migration follows;
- or gold rises only in the classic debasement pattern of a weaker dollar, expanding Fed balance sheet, lower real yields, and wider inflation expectations.

The deliberate-policy thesis is a separate claim. Price data can support the mechanism but cannot establish intent without documents, disclosed operations, or institution-specific transaction evidence.

## Public source endpoints

- FRED graph CSV: `https://fred.stlouisfed.org/graph/fredgraph.csv`
- CFTC TFF Socrata dataset: `https://publicreporting.cftc.gov/resource/gpe5-46if.csv`
- CFTC current TFF text: `https://www.cftc.gov/dea/newcot/FinFutWk.txt`
- Japan MOF weekly securities flows: `https://www.mof.go.jp/policy/international_policy/reference/itn_transactions_in_securities/week.csv`
- Treasury TIC SLT table 3: `https://ticdata.treasury.gov/resource-center/data-chart-center/tic/Documents/slt_table3.txt`
- BIS data portal / SDMX API: `https://data.bis.org/`
- ECB Data Portal API: `https://data-api.ecb.europa.eu/`

## Important limitations

1. **CFTC is a proxy, not the OTC book.** It captures exchange-traded positions by trader category, not the complete FX-swap, bank, pension, or hedge-fund exposure.
2. **TIC is residence/custody oriented.** A Treasury held in the Cayman Islands, Belgium, Luxembourg, or the United Kingdom does not automatically reveal the ultimate beneficial owner or political principal.
3. **MOF weekly flows are aggregate.** They do not fully identify destination, currency, hedge, or investor.
4. **Cross-currency swaps can be off balance sheet.** Gross notional, rollover and margin needs can be large even when current marked-to-market value is small.
5. **Gold revaluation needs a legal/accounting path.** U.S. Treasury owns the official gold; the Fed holds gold certificates at a statutory value. Market appreciation alone does not create reserves or fund fiscal cash flow.
6. **Endpoints are not mechanisms.** Europe weakening and gold rising can occur under several competing models.

## Recommended workflow with Tom

1. Freeze his exact representative trade and counterparties.
2. Freeze candidate event dates and expected signs before running the study.
3. Run the free public baseline.
4. Replace weak public proxies with basis, repo, CDS and institution-specific data.
5. Review every failure condition rather than moving the target after the result.
6. Publish both the strongest supporting result and the strongest disconfirming result.

That process should make the model more useful even if several of its current components fail.
