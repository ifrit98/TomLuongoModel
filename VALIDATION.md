# Build and Validation Record

Validated on 2026-08-10 UTC in the packaging environment.

- `03_luongo_model_test.py` passes Python bytecode compilation.
- The bundled offline parser test completed successfully:
  - CFTC Japanese-yen rows parsed: 1 current-week observation.
  - Japan MOF weekly rows parsed: 1,109.
  - Treasury TIC table 3 rows parsed: 7,007.
  - Latest TIC month in the bundled snapshot: May 2026.
- `01_From_Thesis_to_Test.docx` rendered cleanly to 9 pages and every page was visually reviewed.
- The DOCX accessibility audit reports no high-, medium-, or low-severity findings after adding figure alt text and table-header metadata.

The live public-data download was not completed in the build environment because outbound DNS/network access is disabled there. The live endpoints, retry logic, raw caching, and failure logging are included; run the public-baseline command in `02_README.md` on a normal internet-connected machine. The decisive basis, CDS, repo, hedge-ratio, margin, and actual-return inputs remain intentionally external because they generally require a consistent vendor or dealer source.
