# Validation

- `luongo_model_test_v2.py` passes Python compilation.
- Offline snapshot parser smoke test completed against bundled CFTC, MOF and TIC source snapshots.
- Full live V2 data pull is intentionally left to the user environment because it requires network access and a FRED API key.
- Companion DOCX was rendered to 14 PNG pages and every page was visually inspected after the final chart fix.
- DOCX accessibility audit after final regeneration: 0 high, 0 medium, 0 low findings.
- The V1 raw `run.log` is excluded because an earlier FRED exception exposed a credential in its URL.
