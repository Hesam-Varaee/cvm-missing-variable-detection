# Figure Captions

**Fig. 1** (`fig1_pipeline_flowchart`) 
> Fig. 1. CVM four-stage workflow. Observed variables pass through Stage
> 1 (Creator Variable generation), Stage 2–3 (equation search and model
> selection), and Stage 4 (residual sufficiency classification), which
> branches into sufficient / model insufficient / representation
> insufficient. The last triggers Hypothesis Card generation. The
> selection-stability diagnostic (Section 2.6) audits the Stage 2–3
> model choice that all downstream inference is conditioned on.

**Fig. 2** (`fig2_wor_residual_vs_ad`)
> Fig. 2. WOR base-model residual (log scale) versus shear span-to-depth
> ratio $a/d$, $N=322$. corr = −0.75, confirming $a/d$ as the dominant
> unexplained signal flagged by Stage 4.
> *(script output: `corr(resid, a/d) = -0.7528`)*

**Fig. 3** (`fig3_wor_pred_vs_actual`)
> Fig. 3. WOR predicted vs. actual shear capacity. Left: base model
> ($h\cdot b\cdot f_{ck}$, $R^2=0.675$). Right: extended model with
> $a/d$ and $1/(a/d)$ added ($R^2=0.861$).
> *(script output: `R2 base=0.6753  R2 ext=0.8611`)*

**Fig. 4** (`fig4_wwr_bootstrap_coef`)
> Fig. 4. Distribution of the fck-controlled $\Psi_{vh}$ coefficient
> across 2000 case-resampling bootstrap replicates with the base model
> formula held fixed. 100% of replicates are positive; 95% CI excludes
> zero.
> *(script output: `frac_positive=1.0000  95% CI=[402.7, 958.5]  mean=654.5`
> — matches the manuscript's reported [402.7, 958.6])*

**Fig. 5** (`fig5_wor_stability_boxplot`)
> Fig. 5. Distribution of corr(residual, $a/d$) across 400 bootstrap
> resamples of the WOR dataset, grouped by which near-tied Stage 1–3
> candidate won the resample. The correlation is negative in 100% of
> resamples regardless of which candidate wins; only its magnitude
> varies.
> *(script output: winner counts match Results Table 3.5 exactly;
> 100.0% of resamples negative)*

**Fig. 6** (`fig6_wwr_confound_scatter`)
> Fig. 6. The confound underlying WWR's naive sign reversal:
> $\Psi_{vh}$ vs. concrete strength $f_{ck}$, $N=518$. corr = −0.42.
> *(script output: `corr(psi_vh, fck) = -0.4176`)*

**Fig. 7** (`fig7_aci_comparison`)
> Fig. 7. Ratio of test shear capacity to a simplified slender-beam ACI
> 318 $V_c$ estimate for the WOR subset. Mean ratio 4.75× (i.e., ACI
> underestimates capacity by this factor on average); ACI $R^2=-0.49$.
> *(script output: `mean ratio=4.7513  ACI R2=-0.4899`)*
