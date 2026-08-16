# Luongo Carry-Regime Model V2

V2 incorporates Tom Luongo's feedback after reviewing the first public-data pass.

## What changed

1. **France is the candidate continental weak link.** OAT-Bund is the first public stress proxy; premium inputs should add French sovereign/bank CDS, OAT repo specialness, auctions, and wholesale funding.
2. **Italy is no longer treated as the weak link.** BTP-Bund is reclassified as an ECB fragmentation/intervention-state signal. Spikes should be mapped to PEPP/TPI/reinvestment or other balance-sheet responses.
3. **UK/London remains a separate weak-link channel.** Gilt/SONIA/repo/LDI/dealer funding should be modeled separately from the Eurozone.
4. **US/UK/JP long-end spreads are diagnostic only.** Because they share the same yields, much of their linkage is algebraic. V2 adds 3-month public rates so the economically meaningful carry-capacity test moves to short rates + FX + basis/hedging.
5. **CHF/SNB is added as a substitute funding source.** V2 tracks CHF FX, Swiss 10Y, Swiss 3M, and the CHF funding advantage versus JPY.
6. **EUR/JPY 180 and EUR/GBP 0.85 become explicit hypotheses.** The script identifies every crossing, runs event windows using actual observed dates, and produces regime summaries. These levels are not assumed to be causal or optimal.
7. **The event-study bug from V1 is fixed.** V2 uses each variable's actual observation dates. Price/index variables use log percentage changes; spreads/rates use level changes.

## Run

```bash
export FRED_API_KEY='YOUR_KEY'
python luongo_model_test_v2.py \
  --start 2006-01-01 \
  --events events_v2_template.csv \
  --optional-market-data optional_market_data_v2_template.csv \
  --output outputs/v2_public
```

The optional market-data file may be omitted for the public-only baseline.

## New public FRED inputs

- `DEXUSUK` USD per GBP
- `DEXSZUS` CHF per USD
- `IRLTLT01FRM156N` France 10Y
- `IRLTLT01CHM156N` Switzerland 10Y
- `IR3TIB01JPM156N` Japan 3M interbank
- `IR3TIB01CHM156N` Switzerland 3M interbank
- `IR3TIB01USM156N` United States 3M interbank
- `IR3TIB01GBM156N` United Kingdom 3M interbank (note: public series may be stale; check metadata)

## New derived variables

- `gbp_per_eur` = EUR/GBP
- `chf_per_eur` = EUR/CHF
- `jpy_per_chf` = CHF/JPY expressed as JPY per CHF
- `fr_bund_bp`
- `fr_jp_10y_bp`
- `uk_jp_10y_bp`, `us_uk_10y_bp`
- `ch_jp_10y_bp`, `uk_ch_10y_bp`, `us_ch_10y_bp`
- `us_jp_3m_bp`, `uk_jp_3m_bp`, `us_ch_3m_bp`, `uk_ch_3m_bp`
- `chf_funding_advantage_vs_jpy_bp`
- `btp_bund_change_z24` / `ecb_fragmentation_signal`
- threshold distances and breach states for EUR/JPY 180 and EUR/GBP 0.85

## New outputs

- `threshold_crossings.csv`
- `threshold_crossing_event_study.csv`
- `threshold_regime_summary.csv`
- `placebo_threshold_tests.csv` — EUR/JPY 180 and EUR/GBP 0.85 tested against four
  neighbouring placebo levels each. `days_below`/`days_above` are daily observation
  counts; `months_below`/`months_above` report the (usually much thinner) number of
  labelled months that actually have BTP-Bund/OAT-Bund data behind the monthly
  median/gap columns — always check these before quoting a gap figure.
- `local_projection_btp_on_jpy.csv` — Jordà-style local projections of BTP-Bund on
  yen appreciation vs EUR, horizons h=-6..+6. `coef`/`std_err`/`ci_low`/`ci_high` are
  bp per 1.0 (100%) monthly log-return move; `coef_per_1sd_bp`/`ci_low_per_1sd_bp`/
  `ci_high_per_1sd_bp` are the same estimates rescaled to bp per 1 in-sample standard
  deviation of the regressor (the size of move that actually occurs month to month).
  Do not mix the two unit families — the unscaled and per-SD confidence intervals are
  roughly a 34x scale apart. h=0 is contemporaneous same-month co-movement, not a
  post-shock response. The 95% CIs are pointwise across the 13 tested horizons with
  no multiplicity adjustment: at a Bonferroni-corrected 0.05/13 = 0.0038 threshold
  only h=0 (p=6.2e-4) survives, and h=+1 (p=0.0039) fails by a hair. h=-1 is a
  normalization point (dep is identically zero by construction), not an estimate;
  its `std_err`/`ci_low`/`ci_high`/`nobs` are emitted as null and `estimable=False`.
- `charts/10_local_projection_btp_on_jpy.png` — chart form of the above, with the
  same h=0/multiplicity caveat printed as a subtitle on the chart itself.
- revised `event_study.csv`
- revised `falsification_scorecard.csv`
- revised `regression_report.txt`
- charts for France/Italy/UK, CHF substitution, and Tom's FX boundaries

## Interpretation rules

- Do not call a long-yield residual executable carry.
- Do not treat correlation among US/UK/JP long spreads as independent evidence; the yields are algebraically linked.
- Do not infer ECB intervention from BTP-Bund alone. It is a state/signal variable to align against actual ECB operations.
- Do not optimize the 180/0.85 thresholds on the same sample and then call them predictive. Treat them as Tom's pre-specified priors and test out-of-sample/event-wise.
- Price data cannot establish policymaker intent.
