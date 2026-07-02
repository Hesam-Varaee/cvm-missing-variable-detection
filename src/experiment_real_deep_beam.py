"""
REAL-DATA VALIDATION: RC Deep Beam Shear Strength (WOR subset)
==================================================================
Dataset: Megahed (2024), Scientific Reports, "Prediction and reliability
analysis of shear strength of RC deep beams"
https://doi.org/10.1038/s41598-024-64386-w
Raw data: https://github.com/kmegahed/Deep-beam-ML-models (deep_beam2222.xlsx)

This is the central real-world validation for the paper: unlike the
synthetic beam deflection and ideal gas law experiments (where we KNOW
the ground truth because we generated the data), here we apply the CVM
pipeline to genuine experimental test data with NO known ground truth
"missing variable" -- the question is whether the residual diagnostics
produce a scientifically defensible, interpretable signal rather than
noise, and whether that signal is consistent with known but typically
under-modeled shear-transfer mechanisms in deep beams without web
reinforcement (aggregate interlock, size effect, arch action).

Subset used: 322 specimens WITHOUT web reinforcement (rho_v = rho_h = 0),
input variables {h, d, b, a, fck, rho, fy}, target V (ultimate shear
strength, kN).

Baseline for comparison: ACI 318 shear provisions, which on this exact
subset are reported in the source paper as achieving R^2=0.671 against
test results -- i.e. a known, code-relevant degree of unexplained
variance that domain engineers already recognize and discuss in terms
of specific physical mechanisms (aggregate interlock, arch action,
size effect) that the base ACI formula does not explicitly include.
"""

import numpy as np
import pandas as pd
from cvm_core import CVM
from equation_search import fit_simplest_equation
from residual_diagnostics import test_residual_sufficiency
from property_inference import build_hypothesis_card


def load_wor_subset(path="deep_beam_raw.xlsx"):
    df = pd.read_excel(path, sheet_name="all")
    wor = df[(df["rho_v"] == 0) & (df["rho_h"] == 0)].copy().reset_index(drop=True)
    return wor


def aci_318_shear_estimate(wor):
    """
    Simplified ACI 318 SLENDER-BEAM concrete shear term:
        Vc = 0.17 * sqrt(fc') * b * d   [N, mm, MPa units]

    IMPORTANT CONTEXT: this formula is well documented in the structural
    engineering literature to substantially UNDERESTIMATE the shear
    capacity of deep beams (a/d below ~2.5), because it does not account
    for arch action -- the direct strut mechanism that develops once a
    diagonal crack forms, which becomes dominant as a/d decreases. This
    is not a bug in our baseline; it is the well-known, code-recognized
    reason ACI 318 mandates strut-and-tie methods (not this simple Vc
    formula) for deep beam design. We retain this comparison deliberately
    because it gives a real, literature-grounded illustration of exactly
    the kind of 'missing mechanism' scenario CVM's residual diagnostics
    are designed to surface.
    """
    Vc_N = 0.17 * np.sqrt(wor["fck"]) * wor["b"] * wor["d"]
    return Vc_N / 1000.0  # convert N to kN to match V column


def run_real_data_experiment(verbose=True):
    wor = load_wor_subset()
    N = len(wor)

    h = wor["h"].values.astype(float)
    d = wor["d"].values.astype(float)
    b = wor["b"].values.astype(float)
    a = wor["a"].values.astype(float)
    fck = wor["fck"].values.astype(float)
    rho = wor["rho"].values.astype(float)
    fy = wor["fy"].values.astype(float)
    V = wor["V"].values.astype(float)

    X_observed = {"h": h, "d": d, "b": b, "a": a, "fck": fck, "rho": rho, "fy": fy}
    units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}
    y_units = "[kN]"

    if verbose:
        print(f"WOR subset: N={N} specimens")
        print("Variables:", list(X_observed.keys()))

    cvm = CVM(variable_names=list(X_observed.keys()), variable_units=units,
              max_power=1, max_terms=3)
    cvm.fit(X_observed, V, top_k=15)

    if verbose:
        print("\n--- STAGE 1: top Creator Variables ---")
        for c in cvm.top(8):
            print(f"  {c}")

    best_model, pareto = fit_simplest_equation(cvm.candidates_, V, lam=0.01)

    if verbose:
        print("\n--- STAGE 2-3: best symbolic model ---")
        print(f"  {best_model}")
        print(f"  R2 = {best_model.r2:.4f}  (form={best_model.form})")

    creator_var_values = {c.name: c.values for c in cvm.top(8)}
    report = test_residual_sufficiency(
        residuals=best_model.residuals,
        observed_vars=X_observed,
        creator_vars=creator_var_values,
        exclude_from_creator_vars=[best_model.creator_var_name],
    )

    if verbose:
        print("\n--- STAGE 4: residual sufficiency test ---")
        print(f"  {report}")

    best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
    card = build_hypothesis_card(
        residuals=best_model.residuals,
        best_creator_var=best_cv.values if best_model.form == "linear" else np.log(best_cv.values),
        best_creator_var_units=best_cv.units,
        y_units=y_units,
        observed_vars=X_observed,
        fitted_coef=best_model.coef,
        is_log_space=(best_model.form == "power_law"),
    )

    if verbose:
        print("\n--- HYPOTHESIS CARD ---")
        print(card.display())

    V_aci = aci_318_shear_estimate(wor)
    aci_resid = V - V_aci
    aci_r2 = 1 - np.sum(aci_resid**2) / np.sum((V - V.mean())**2)
    aci_ratio = V / V_aci
    aci_mean_ratio = np.mean(aci_ratio)
    aci_cov_ratio = np.std(aci_ratio) / aci_mean_ratio

    if verbose:
        print("\n--- BASELINE: simplified ACI 318 concrete shear term ---")
        print(f"  R2 = {aci_r2:.4f}")
        print(f"  Mean(V_test/V_ACI) = {aci_mean_ratio:.3f}, CoV = {aci_cov_ratio:.3f}")

    a_d = wor["a_d"].values.astype(float)
    resid_for_check = best_model.residuals
    corr_resid_ad = np.corrcoef(resid_for_check, a_d)[0, 1]

    if verbose:
        print("\n--- DIAGNOSTIC: residual vs a/d ratio (arch action proxy) ---")
        print(f"  corr(CVM residual, a/d) = {corr_resid_ad:.4f}")

    # ---- Confirmatory test: does adding a/d to the model substantially ----
    # ---- improve fit, validating that the flagged residual signal is ----
    # ---- real and actionable, not spurious? ----
    from sklearn.linear_model import LinearRegression
    log_V = np.log(V)
    log_phi = np.log(best_cv.values)
    X_with_ad = np.column_stack([log_phi, a_d, 1.0 / a_d])
    reg_extended = LinearRegression().fit(X_with_ad, log_V)
    r2_extended = reg_extended.score(X_with_ad, log_V)

    if verbose:
        print("\n--- CONFIRMATORY TEST: adding a/d (and 1/a_d) to the model ---")
        print(f"  R2 without a/d: {best_model.r2:.4f}")
        print(f"  R2 with a/d, 1/a_d added: {r2_extended:.4f}")
        print(f"  Improvement: {r2_extended - best_model.r2:.4f}")

    return {
        "wor": wor, "cvm": cvm, "best_model": best_model, "report": report,
        "card": card, "aci_r2": aci_r2, "aci_mean_ratio": aci_mean_ratio,
        "aci_cov_ratio": aci_cov_ratio, "corr_resid_ad": corr_resid_ad,
        "r2_extended_with_ad": r2_extended,
    }


if __name__ == "__main__":
    results = run_real_data_experiment(verbose=True)
