# Figure Captions & Placement Guide

All figures are provided as both PNG (300 dpi, for review/Word) and PDF
(vector, for final typesetting) in `/figures/`. Every number shown was
regenerated directly from `deep_beam_raw.xlsx` and the actual pipeline
code (`cvm_core.py`, `equation_search.py`) and cross-checked against the
values already stated in the manuscript text — see the confirmation
lines in each generation script's output, reproduced below each caption.

---

**Fig. 1** (`fig1_pipeline_flowchart`) — *Place in Section 2.1 (Methodology
overview).* 
> Fig. 1. CVM four-stage workflow. Observed variables pass through Stage
> 1 (Creator Variable generation), Stage 2–3 (equation search and model
> selection), and Stage 4 (residual sufficiency classification), which
> branches into sufficient / model insufficient / representation
> insufficient. The last triggers Hypothesis Card generation. The
> selection-stability diagnostic (Section 2.6) audits the Stage 2–3
> model choice that all downstream inference is conditioned on.

**Fig. 2** (`fig2_wor_residual_vs_ad`) — *Place in Section 3.3 (WOR
results), alongside the in-sample result paragraph.*
> Fig. 2. WOR base-model residual (log scale) versus shear span-to-depth
> ratio $a/d$, $N=322$. corr = −0.75, confirming $a/d$ as the dominant
> unexplained signal flagged by Stage 4.
> *(script output: `corr(resid, a/d) = -0.7528`)*

**Fig. 3** (`fig3_wor_pred_vs_actual`) — *Place in Section 3.3, immediately
after Fig. 2.*
> Fig. 3. WOR predicted vs. actual shear capacity. Left: base model
> ($h\cdot b\cdot f_{ck}$, $R^2=0.675$). Right: extended model with
> $a/d$ and $1/(a/d)$ added ($R^2=0.861$).
> *(script output: `R2 base=0.6753  R2 ext=0.8611`)*

**Fig. 4** (`fig4_wwr_bootstrap_coef`) — *Place in Section 3.4 (WWR
root-cause analysis), alongside the fixed-formula bootstrap paragraph.*
> Fig. 4. Distribution of the fck-controlled $\Psi_{vh}$ coefficient
> across 2000 case-resampling bootstrap replicates with the base model
> formula held fixed. 100% of replicates are positive; 95% CI excludes
> zero.
> *(script output: `frac_positive=1.0000  95% CI=[402.7, 958.5]  mean=654.5`
> — matches the manuscript's reported [402.7, 958.6])*

**Fig. 5** (`fig5_wor_stability_boxplot`) — *Place in Section 3.5 (WOR
downstream-stability check), replacing or alongside the results table.*
> Fig. 5. Distribution of corr(residual, $a/d$) across 400 bootstrap
> resamples of the WOR dataset, grouped by which near-tied Stage 1–3
> candidate won the resample. The correlation is negative in 100% of
> resamples regardless of which candidate wins; only its magnitude
> varies.
> *(script output: winner counts match Results Table 3.5 exactly;
> 100.0% of resamples negative)*

**Fig. 6** (`fig6_wwr_confound_scatter`) — *Place in Section 3.4, near
the confound description.*
> Fig. 6. The confound underlying WWR's naive sign reversal:
> $\Psi_{vh}$ vs. concrete strength $f_{ck}$, $N=518$. corr = −0.42.
> *(script output: `corr(psi_vh, fck) = -0.4176`)*

**Fig. 7** (`fig7_aci_comparison`) — *Place in Section 3.3, near the ACI
318 baseline comparison paragraph.*
> Fig. 7. Ratio of test shear capacity to a simplified slender-beam ACI
> 318 $V_c$ estimate for the WOR subset. Mean ratio 4.75× (i.e., ACI
> underestimates capacity by this factor on average); ACI $R^2=-0.49$.
> *(script output: `mean ratio=4.7513  ACI R2=-0.4899`)*

---

## Notes on style

- All figures use a shared color palette and font (`fig_common_setup.py`,
  included in the code deliverables) for visual consistency across the
  manuscript.
- Fig. 1 is schematic (no underlying data); Figs. 2–7 are generated
  directly from `deep_beam_raw.xlsx` via the actual project code, not
  illustrative mockups.
- PDF versions are vector graphics and should be used for final
  typesetting; PNGs (300 dpi) are provided for convenience during review
  and Word-based drafting.
- Engineering Structures (Elsevier) typically expects figures to be
  submitted as separate files at submission, with captions in the
  manuscript text rather than embedded in the figure file itself — the
  captions above are written in that format (caption text only, no title
  baked into the image) for Figs. 2, 3, 4, 6, 7. Figs. 1 and 5 currently
  have their caption text rendered inside the image for readability
  during drafting; strip the in-image title before final submission if
  the journal's figure-file requirements prohibit embedded captions
  (verify against the current guide for authors).
