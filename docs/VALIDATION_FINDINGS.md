# CVM Validation Findings — Out-of-Sample Robustness & Selection Stability

This document records a second round of validation performed on top of the
original CVM results (see `README.md`). The original README results are all
**in-sample**: Stage 1 candidate search, Stage 2-3 equation fitting, and the
confirmatory "does adding the flagged variable improve fit" tests were all
run on the same rows. This document reports what happens when that
circularity is removed, and documents a new diagnostic capability added to
the codebase as a result.

All numbers below were produced by re-running the actual project code
(not re-derived by hand); scripts referenced are included alongside this
document.

---

## 1. Motivation

For publication, an in-sample "the missing variable we searched
hundreds of candidates for turns out to explain leftover variance" result
is not, by itself, convincing — reviewers will expect out-of-sample or
cross-validated evidence, especially for the real-data (deep beam) results
that are the paper's central claim. We therefore re-ran both real-data
experiments (WOR and WWR subsets of the Megahed 2024 deep-beam dataset)
using genuine out-of-fold validation, and used the results to stress-test
the pipeline itself.

---

## 2. Out-of-fold validation: WOR (arch action / a-d ratio)

**Method:** 5-fold cross-validation. For each fold, Stage 1 (candidate
search) and Stage 2-3 (equation fitting) were fit on the other 4 folds
only; the held-out fold's residual was recorded. Out-of-fold residuals
for all N=322 rows were then pooled and Stage 4 (residual diagnostics)
was run once on the full-size, genuinely out-of-sample residual vector —
this preserves statistical power for Stage 4's permutation-MI test while
keeping every residual unbiased.

**Result, across 5 different fold-assignment seeds:**

| seed | classification | flagged variable | corr(resid, a/d) | held-out R² gain from adding a/d |
|---|---|---|---|---|
| 0–4 | `model_insufficient` (all 5) | `a` (all 5) | −0.74 to −0.75 | +0.176 to +0.185 |

This is essentially identical to the in-sample numbers reported in the
README (corr=−0.75, R² gain +0.186 in-sample). **The arch-action finding
is robust to genuine out-of-sample validation.**

Script: `experiment_holdout_kfold.py::run_wor_kfold`

---

## 3. Out-of-fold validation: WWR (web reinforcement / Ψvh confound)

Same 5-fold out-of-fold procedure, N=518.

**Result, across 5 seeds:**
- Stage 4 stays `model_insufficient` every time (consistent with README).
- The flagged max-MI variable is **not** Ψvh-related in any fold — it is
  consistently a different creator variable (`h⁻¹·b⁻¹·fck⁻¹`).
- The sign of the fck-controlled Ψvh coefficient, when the base model is
  refit per training fold (as the original in-sample script does),
  **flips essentially at random** across folds/seeds — roughly 50/50
  positive vs. negative.
- The out-of-fold R² gain from adding Ψvh is tiny (0.0001–0.0035) versus
  0.0041 in-sample.

**Initial conclusion (before Section 4):** the confound-corrected sign
result reported in the README does not survive naive out-of-fold
cross-validation. This looked, initially, like the WWR result was an
artifact of using the full dataset.

Script: `experiment_holdout_kfold.py::run_wwr_kfold`

---

## 4. Root-cause investigation: is the Ψvh effect itself unstable, or is something else going on?

To distinguish "the effect is genuinely weak/noisy" from "something in the
*pipeline* is injecting instability," two further checks were run.

### 4.1 Full-sample statistical significance

OLS on the full N=518 WWR dataset, confound-controlled model
(target ~ phi + fck + Ψvh), with correct standard errors:

```
psi_vh   coef = 660.3   se = 182.9   t = 3.61   p = 0.0003
VIF (phi, fck, psi_vh): 1.05, 1.27, 1.25   →  no multicollinearity problem
```

The effect is strongly statistically significant on the full data — not a
marginal or borderline result.

### 4.2 Case-resampling bootstrap with the functional form held fixed

2000 bootstrap resamples (with replacement, N=518 each), using the SAME
base creator variable (`h * b`, the full-data Stage 2-3 winner) in every
resample — i.e., isolating coefficient sampling variability from
creator-variable *selection* variability:

```
Positive sign: 2000 / 2000 bootstrap replicates
95% CI for the Ψvh coefficient: [402.7, 958.6]  →  clearly excludes zero
```

**Conclusion: the physical effect (web reinforcement increases capacity,
once fck is controlled for) is real, statistically significant, and
robust under resampling — when the base model is held fixed.**

### 4.3 So why did Section 3's fold-by-fold test show random sign flips?

Because that test let Stage 1/2-3 **re-select** the base creator variable
independently on each training fold. On the full data, several candidates
are near-tied in Stage 1 correlation and/or Stage 2-3 fitted score:

```
d * b * rho   corr = 0.908
h * b * rho   corr = 0.905
h * b         corr = 0.898   ← full-data Stage 2-3 winner
d * b * fck   corr = 0.898
```

These are close enough that resampling ~80% of the data changes which one
wins. Since the Ψvh coefficient is a *partial* coefficient relative to
whichever base variable was selected, a flipped base-variable selection
can flip the sign of the downstream Ψvh coefficient — even though the
underlying effect never moved. **The instability lives in the pipeline's
model-selection step, not in the physical signal being tested.**

Scripts: full-sample OLS / VIF / fixed-formula bootstrap, run ad hoc (see
chat log); logic is straightforward to re-materialize as a standalone
script if needed for the paper's supplementary material.

---

## 5. New diagnostic added to the codebase: `CVM.assess_stability()`

Given Section 4's finding, a Stage 1 selection-stability diagnostic was
added to `cvm_core.py` (pure addition — no existing logic was modified).

**What it does:** after `CVM.fit()`, `assess_stability(y, n_boot=...)`
case-resample-bootstraps the rows and re-scores the *already-found* top-k
candidates (no re-search of the combinatorial space — cheap) using the
same scoring rule as `fit()`. It tracks which candidate wins each
resample and returns a `StabilityReport`: win rate of the full-data
winner, the runner-up, the margin between them, and a `stable` flag
(win rate ≥ 50% AND margin ≥ 15%, both configurable).

**Validated on:**
- **Synthetic beam case:** flagged `STABLE` (top candidate wins 81.7% of
  resamples; the runner-up is its own reciprocal-sign twin, so this is
  expected and harmless).
- **WOR:** flagged `UNSTABLE` (top candidate wins only 43.2%, runner-up
  33.4%, margin 9.8%).
- **WWR:** Stage 1 (raw correlation) ranking itself flagged `STABLE`
  (`d*b*rho` wins 54.8%) — note this differs from the Stage 2-3 winner
  (`h*b`), because `fit_simplest_equation` re-ranks by a
  complexity-penalized score, not raw correlation. **This means the
  Stage 1 check alone does not fully capture the instability found in
  Section 4** — a Stage 2-3-level check would be needed to catch it
  directly. (An initial implementation, `assess_equation_stability` in
  `equation_search.py`, was drafted and then reverted at the user's
  request pending this documentation step; it is not currently in the
  codebase.)

---

## 6. Does WOR's flagged Stage 1 instability actually matter?

Section 5 flagged WOR's Stage 1 selection as "unstable" (43.2% win rate,
near-tied with `d⁻¹·b⁻¹·fck⁻¹` and others). This raised the question:
does that instability threaten the arch-action conclusion the way the
analogous instability threatened WWR's Ψvh sign?

**Method:** 400 bootstrap resamples of WOR; on each, Stage 2-3 selection
was re-run among the fixed top-k candidates, and for that resample's
winning model, `corr(residual, a/d)` and the R² gain from adding a/d were
recorded, grouped by which candidate won.

**Result:**

| winner (n out of 400) | mean corr(resid, a/d) | mean R² gain from a/d |
|---|---|---|
| h·b·fck (194) | −0.72 | +0.167 |
| d⁻¹·b⁻¹·fck⁻¹ (149) | −0.74 | +0.184 |
| d·b·fck (13) | −0.63 | +0.134 |
| d·b·ρ (18) | −0.53 | +0.099 |
| h·b·ρ (8) | −0.53 | +0.082 |
| h⁻¹·b⁻¹·fck⁻¹ (18) | −0.74 | +0.177 |

**Across all 400 resamples: the sign of corr(resid, a/d) was negative in
100% of resamples, and the R² gain from adding a/d was positive in 100%
of resamples.** Only the *magnitude* of the effect varies with which
candidate wins (roughly −0.53 to −0.81 for the correlation), never its
direction or qualitative conclusion.

**Conclusion: WOR's Stage 1/2-3 selection instability is real but
inconsequential — the scientific conclusion (a/d is the dominant missing
signal) does not depend on which near-tied candidate happens to be
selected.** This is the key qualitative difference from WWR, where the
analogous instability *did* change the sign of the downstream conclusion
before the base model was held fixed (Section 4.2).

Script: `check_wor_downstream_stability.py`

---

## 7. Summary table

| Question | WOR | WWR |
|---|---|---|
| Out-of-fold Stage 4 classification | `model_insufficient`, stable across seeds | `model_insufficient`, stable across seeds |
| Out-of-fold recovery of the target signal | Yes (a/d, robust) | Not directly (Ψvh never top-flagged out-of-fold) |
| Is the underlying physical effect real? | Yes (never in question) | Yes — confirmed via full-sample OLS (p=0.0003) and fixed-formula bootstrap (2000/2000 positive) |
| Is Stage 1/2-3 candidate selection stable? | No (43% win rate, near 3-way tie) | No (Stage 2-3 winner differs from Stage 1 top-correlation winner) |
| Does that instability change the qualitative conclusion? | **No** — sign/direction never flips across 400 resamples | **Yes** — sign of the confound-controlled coefficient flips when the base model is re-selected per resample |

---

## 8. Open items / recommended next steps

1. **Decide whether to formally implement `assess_equation_stability`**
   (Stage 2-3-level stability check, analogous to `CVM.assess_stability`
   but operating on `fit_simplest_equation`'s output) — this is the check
   that would have caught the WWR instability directly, rather than
   requiring the manual Section 4 investigation.
2. **Wire `assess_stability` (and its Stage 2-3 counterpart, once built)
   into `experiment_real_deep_beam.py` and `experiment_real_deep_beam_wwr.py`**
   so the stability report is part of the standard experiment output, not
   a one-off side analysis.
3. **Decide the WWR narrative for the paper.** Recommended framing per
   the earlier discussion: report the confound-controlled effect as real
   and statistically robust (Section 4), but explicitly discuss the
   selection-instability finding as a genuine, precisely-characterized
   methodological limitation of greedy top-1 candidate selection — turning
   it into a contribution (motivates the stability-check addition) rather
   than a hidden flaw.
4. **Paper structure:** WOR as the primary, fully out-of-sample-validated
   real-data result; WWR as a secondary case study demonstrating both
   correct qualitative diagnosis (`model_insufficient`) and the
   stability-check contribution.
