# CVM (Creator Variable Machine) — Paper Code Package

This package contains all code underlying the paper "Detecting missing
variables from residual structure: a validated four-stage framework with
application to reinforced concrete deep beam shear capacity."

## Structure

```
CVM_package/
├── src/                    All runnable Python code (flat layout, see note below)
│   ├── cvm_core.py               Stage 1 + selection-stability diagnostic
│   ├── equation_search.py        Stage 2-3
│   ├── residual_diagnostics.py   Stage 4
│   ├── property_inference.py     Hypothesis Card generation
│   ├── experiment_beam.py                          synthetic, positive control
│   ├── experiment_negative_control.py               synthetic, negative control
│   ├── experiment_gas_law.py                         synthetic, positive control
│   ├── experiment_gas_law_negative_control.py         synthetic, negative control
│   ├── experiment_real_deep_beam.py                    real data, WOR subset
│   ├── experiment_real_deep_beam_wwr.py                 real data, WWR subset
│   ├── experiment_holdout_kfold.py                       5-fold out-of-fold validation (WOR + WWR)
│   ├── experiment_holdout_validation.py                   single-split holdout (superseded by k-fold; kept for reference)
│   ├── check_wor_downstream_stability.py                   400-resample WOR robustness check
│   ├── fig_common_setup.py                                  shared style/data-loading for figure scripts
│   ├── gen_fig1_flowchart.py … gen_fig7 (via gen_figs_2_3_6_7.py, gen_fig4_bootstrap.py, gen_fig5_wor_stability.py)
│   └── deep_beam_raw.xlsx        working copy (so scripts run unmodified with relative paths)
├── data/
│   └── deep_beam_raw.xlsx        canonical copy of the dataset (Megahed, 2024)
├── figures_output/               pre-generated PNG (300dpi) + PDF (vector) for all 7 figures
├── tests/                        pytest suite (43 tests, see below)
├── docs/
│   ├── original_project_README.md   the original project README
│   ├── VALIDATION_FINDINGS.md        full write-up of the out-of-sample/stability investigation
│   └── FIGURE_CAPTIONS.md            captions + manuscript placement for each figure
├── requirements.txt
└── pytest.ini
```

**Note on `src/` layout:** all modules use flat, sibling-style imports
(e.g. `from cvm_core import CVM`), exactly as originally written and
validated. Rather than restructure into a nested package (which would
require editing every import statement in already-validated code), all
runnable scripts live together in `src/`. Run any script directly from
inside `src/`:

```bash
cd src
python3 experiment_beam.py
python3 experiment_real_deep_beam.py
python3 experiment_holdout_kfold.py
python3 gen_fig1_flowchart.py    # writes into ../figures_output or ./figures depending on script — see script header
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the experiments

```bash
cd src

# Synthetic validation (Section 3.1-3.2 of the paper)
python3 experiment_beam.py
python3 experiment_negative_control.py
python3 experiment_gas_law.py
python3 experiment_gas_law_negative_control.py

# Real-data validation (Section 3.3-3.4)
python3 experiment_real_deep_beam.py
python3 experiment_real_deep_beam_wwr.py

# Out-of-sample / stability validation (Section 3.3-3.5, 4.2)
python3 experiment_holdout_kfold.py
python3 check_wor_downstream_stability.py
```

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

43 tests covering:
- `test_cvm_core.py` — Stage 1 candidate generation, reciprocal
  canonicalization, deduplication, and the selection-stability
  diagnostic (`CVM.assess_stability()`).
- `test_equation_search.py` — Stage 2-3 linear vs. power-law model
  selection and complexity-penalized scoring.
- `test_residual_diagnostics.py` — Stage 4's three-way classification
  (sufficient / model insufficient / representation insufficient), the
  mutual information estimator's scale-robustness (quantile vs.
  equal-width binning), and the Bonferroni correction.
- `test_property_inference.py` — unit parsing, SI base-unit
  decomposition, direction/interaction/smoothness inference, and
  Hypothesis Card assembly.
- `test_integration_synthetic.py` — end-to-end regression tests tying
  pipeline behavior to the specific results reported in the paper
  (e.g., withholding $I$ in the beam experiment must classify as
  representation insufficient and recover unit power $[m]^4$).

All 43 tests pass as of this package's creation (verified without
network access using a local pytest-compatible shim; a fresh
`pip install pytest && pytest tests/` is expected to reproduce this).

## Regenerating figures

```bash
cd src
python3 gen_fig1_flowchart.py      # schematic, no data
python3 gen_figs_2_3_6_7.py        # Figs. 2, 3, 6, 7 (WOR/WWR real-data plots)
python3 gen_fig4_bootstrap.py      # Fig. 4 (WWR bootstrap coefficient distribution, 2000 resamples)
python3 gen_fig5_wor_stability.py  # Fig. 5 (WOR 400-resample stability boxplot)
```

Figures are written to a local `figures/` subdirectory relative to
wherever the script is run; pre-generated copies (matching the numbers
reported in the paper) are already provided in `figures_output/`.

## Data

`deep_beam_raw.xlsx` is the reinforced-concrete deep-beam shear-strength
dataset from Megahed (2024), *Scientific Reports*, 14, 14590.
https://doi.org/10.1038/s41598-024-64386-w. It is included here for
reproducibility; see the source paper for licensing/attribution terms
before redistribution.

## Known limitation of this package

The Stage 2-3 selection-stability diagnostic (analogous to
`CVM.assess_stability()` but operating on `fit_simplest_equation`'s
complexity-penalized output rather than Stage 1's raw correlation
ranking) is described in the paper as a planned extension and is **not**
implemented in this codebase — see `docs/VALIDATION_FINDINGS.md`,
Section 5, for the manual analysis that was performed in its place.
