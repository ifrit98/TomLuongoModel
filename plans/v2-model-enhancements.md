# V2 Carry-Regime Model — Enhancement Plan

Three enhancements to the V2 harness that can be done on existing public data,
while awaiting practitioner data from Matt Dines and Vince Lanci.

## Context

`Luongo_Model_V2_Package/luongo_model_test_v2.py` tests Tom Luongo's carry-regime
thesis. V2 separates three roles: carry condition (Germany–Japan geometry, EUR/JPY,
CHF funding), weak-link stress (France OAT–Bund, UK Gilt–Bund), and policy response
(Italy BTP–Bund as an ECB intervention signal).

The current headline result is that yen appreciation vs EUR widens BTP–Bund
(coef +162.99, p<0.001) while sterling appreciation does not (p=0.92). Two problems
with how that is currently reported motivate this work:

1. `V2_CHANGELOG.md` claims the 180 / 0.85 thresholds get "crossing/event/**placebo**
   tests." Placebo tests were never implemented. This is a documentation/code gap.
2. French OAT–Bund enters the BTP–Bund regression at z=8.5 and absorbs much of the
   explanatory power, so the reported R² overstates what the carry variables explain.
3. The yen→BTP relationship is contemporaneous only, so it establishes co-movement
   rather than sequence.

## Global Constraints

- **Single target file:** `Luongo_Model_V2_Package/luongo_model_test_v2.py`. Do not
  create new modules.
- **Interpreter:** use `../.venv/bin/python` relative to the package directory
  (Python 3.14, pandas 3.0.5, numpy 2.5.2, statsmodels 0.14.6).
- **NaN semantics are load-bearing.** A bare `series < threshold` comparison
  evaluates NaN to False; `.astype(float)` then freezes it as 0. This exact bug
  previously mis-stated regime counts by 14×. Any new threshold indicator MUST
  preserve missingness with `.where(series.notna())`. Never `fillna(0)` an
  observation indicator.
- **Never write secrets to any output.** `write_metadata` redacts `fred_api_key`;
  keep it that way.
- **Match existing style:** module-level `LOGGER`, type hints on function
  signatures, functions return `pd.DataFrame`, callers write CSVs into `output_dir`.
- **Both run paths must still pass** after every task:
  - `python luongo_model_test_v2.py --offline-snapshots --output /tmp/smoke`
    must print `Offline parser checks passed` with cftc_rows 1, mof_rows 1109,
    tic_rows 7007.
  - The live run requires a network + `FRED_API_KEY`. If unavailable in your
    environment, say so in your report rather than skipping verification silently.
- **Do not rename or remove any existing output file.** Additive changes only.
- **Guard for missing columns.** Every new function must return an empty
  `pd.DataFrame()` when its required columns are absent, matching how
  `threshold_regime_summary` and `extract_threshold_crossings` already behave.

---

## Task 1: Placebo threshold tests

Close the gap between `V2_CHANGELOG.md`'s claim and the code. The question being
answered: do 180 and 0.85 behave any differently from neighbouring levels, or would
any nearby threshold produce the same picture?

**Add function:**

```python
def placebo_threshold_tests(daily: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
```

**Behaviour:**

- Test EUR/JPY (`jpy_per_eur`) at thresholds `[170.0, 175.0, 180.0, 185.0, 190.0]`
  and EUR/GBP (`gbp_per_eur`) at `[0.83, 0.84, 0.85, 0.86, 0.87]`.
- For each (pair, threshold), compute over days where the FX rate is observed:
  - `days_below`, `days_above`
  - `crossings_below`, `crossings_above` — reuse the transition logic already in
    `extract_threshold_crossings` rather than reimplementing it
  - `median_btp_bund_bp_below`, `median_btp_bund_bp_above`, and their difference
    `btp_bund_bp_gap` (below minus above)
  - the same three for `fr_bund_bp`
  - `median_vix_below`, `median_vix_above`
- The spread columns live on the **monthly** frame. Label each month by the modal
  daily state for that threshold, exactly as `threshold_regime_summary` does, then
  take medians over the labelled months.
- Include a boolean column `is_prespecified`, True only for 180.0 and 0.85.
- Include `first_date_below` and `last_date_below` so the reader can see whether a
  given threshold's below-state is era-confined (180 is; this is the point).
- Sort output by pair, then threshold.

**Wire into `live_run`:** write to `placebo_threshold_tests.csv` in `output_dir`,
alongside the existing `threshold_crossings.csv` write. Log a line at INFO.

**Verification:** run the offline smoke test, then confirm on the committed
`baseline_v1/daily_panel.csv` and `baseline_v1/monthly_panel.csv` (or a live run if
you have network) that the function returns 10 rows, that the 180.0 row's
`days_below` matches the sum of the `both_below` and `eurjpy_below_only` daily
observations from `threshold_regime_summary`, and that no row has `days_below +
days_above` equal to the raw calendar row count (which would indicate the NaN bug
has been reintroduced).

---

## Task 2: BTP–Bund regression without France

`d_fr_bund_bp` enters `ecb_fragmentation_signal` at +1.83 (z=8.54) and dominates its
R² of 0.465. We need to see the yen coefficient without France absorbing the variance,
because the headline claim is about yen→Italy, not France→Italy.

**Change:** in `run_regressions`, add a second specification immediately after the
existing `ecb_fragmentation_signal` block, named
`ecb_fragmentation_signal_ex_france`, with the same dependent variable
(`d_btp_bund_bp`) and the same independents **minus** `d_fr_bund_bp`.

Keep the original specification exactly as it is — both must appear in
`regression_report.txt`, original first.

**Verification:** confirm both blocks appear in the generated `regression_report.txt`
with distinct R² values, and report both the R² and the `jpy_appreciation_vs_eur`
coefficient, standard error and p-value for the new specification in your report.
This is the number the controller needs.

---

## Task 3: Local projections of BTP–Bund on yen appreciation

Turn the contemporaneous correlation into a path, so we can see whether yen strength
leads Italian spread widening or merely accompanies it. Negative horizons act as a
pre-trend check: significant response before the shock undermines a causal reading.

**Add function:**

```python
def local_projection_btp_on_jpy(monthly: pd.DataFrame, max_horizon: int = 6) -> pd.DataFrame:
```

**Behaviour:**

- Jordà-style local projections at horizons `h` from `-max_horizon` to `+max_horizon`.
- For each `h`, the dependent variable is the cumulative change in `btp_bund_bp`
  from `t-1` to `t+h`:
  `dep = monthly["btp_bund_bp"].shift(-h) - monthly["btp_bund_bp"].shift(1)`
- Regress `dep` on `jpy_appreciation_vs_eur` at time `t`, with `d_vix` as a control
  and a constant. Use the existing HAC machinery; set `maxlags = max(3, abs(h) + 1)`.
- Return one row per horizon with columns: `horizon`, `coef`, `std_err`, `z`,
  `pvalue`, `ci_low`, `ci_high`, `nobs`. `coef` etc. refer to the
  `jpy_appreciation_vs_eur` coefficient, not the constant.
- Skip a horizon (do not crash) if fewer than 30 usable observations remain.
- `hac_regression` currently returns a formatted summary string. Do not change its
  signature or behaviour — other callers depend on it. Fit the model directly with
  `sm.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": ...})` inside the new
  function so you can read numeric attributes.

**Wire into `live_run`:** write `local_projection_btp_on_jpy.csv` to `output_dir`.

**Also add a chart** in `plot_outputs`, saved as
`charts/10_local_projection_btp_on_jpy.png`: coefficient by horizon as a line with a
shaded 95% confidence band, a horizontal line at zero, and a vertical line at
horizon 0. Label the y-axis "BTP–Bund response (bp)" and the x-axis
"Months from yen appreciation". Follow the existing `save_plot` pattern.

**Verification:** run the offline smoke test, then produce the CSV from
`baseline_v1/monthly_panel.csv` (or a live run) and report the coefficient path —
specifically whether coefficients at negative horizons are distinguishable from zero,
since that is what determines whether the result survives as evidence of sequence.
