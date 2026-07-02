"""
OUT-OF-SAMPLE VALIDATION: Real Deep Beam Experiments (WOR + WWR)
====================================================================
The in-sample versions (experiment_real_deep_beam.py,
experiment_real_deep_beam_wwr.py) fit CVM's candidate search AND the
confirmatory "does adding the flagged variable improve fit" test on the
SAME data. That's circular: CVM searches hundreds of candidates for the
one that best explains leftover residual variance, then "confirms" it
explains variance, on the same rows. This script re-runs both real-data
experiments with a genuine train/test split:

  1. Fit CVM (Stage 1) + equation search (Stage 2-3) on TRAIN only.
  2. Evaluate the chosen model's R^2 on the held-out TEST set.
  3. Compute Stage 4 residual diagnostics on TEST residuals only
     (using creator variable values evaluated on test rows, not
     re-fit on them).
  4. Confirmatory test: fit the extended model (base + flagged
     variable, e.g. a/d or Psi_vh) on TRAIN only, then evaluate
     R^2 improvement on TEST.

Repeated across multiple random splits to get stable estimates given
the modest sample sizes (N=322 / N=518).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from cvm_core import CVM
from equation_search import fit_simplest_equation
from residual_diagnostics import test_residual_sufficiency


def load_wor(path="deep_beam_raw.xlsx"):
    df = pd.read_excel(path, sheet_name="all")
    return df[(df["rho_v"] == 0) & (df["rho_h"] == 0)].copy().reset_index(drop=True)


def load_wwr(path="deep_beam_raw.xlsx"):
    df = pd.read_excel(path, sheet_name="all")
    return df[(df["rho_v"] > 0) | (df["rho_h"] > 0)].copy().reset_index(drop=True)


def _split(df, test_frac, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(len(df) * test_frac))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _predict(best_model, best_cv, X_dict):
    """Apply the train-fitted model (coef/intercept/form) to new X."""
    vals = best_cv.formula(X_dict)
    if best_model.form == "power_law":
        log_pred = best_model.coef * np.log(vals) + best_model.intercept
        return np.exp(log_pred), vals
    else:
        pred = best_model.coef * vals + best_model.intercept
        return pred, vals


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def run_wor_holdout(seed=0, test_frac=0.2, verbose=False):
    wor = load_wor()
    train, test = _split(wor, test_frac, seed)

    vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
    units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}

    X_train = {v: train[v].values.astype(float) for v in vars_}
    V_train = train["V"].values.astype(float)
    X_test = {v: test[v].values.astype(float) for v in vars_}
    V_test = test["V"].values.astype(float)

    # ---- Fit on TRAIN only ----
    cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
    cvm.fit(X_train, V_train, top_k=15)
    best_model, pareto = fit_simplest_equation(cvm.candidates_, V_train, lam=0.01)
    best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)

    # ---- Evaluate on TEST ----
    pred_test, cv_test_vals = _predict(best_model, best_cv, X_test)
    r2_test_raw = _r2(V_test, pred_test)

    if best_model.form == "power_law":
        log_V_test = np.log(V_test)
        log_pred_test = best_model.coef * np.log(cv_test_vals) + best_model.intercept
        resid_test = log_V_test - log_pred_test
        r2_test_scale = _r2(log_V_test, log_pred_test)  # log-space R2, comparable to train R2
    else:
        resid_test = V_test - pred_test
        r2_test_scale = r2_test_raw

    # ---- Stage 4 on TEST residuals, using top creator vars evaluated on TEST rows ----
    creator_var_test_values = {c.name: c.formula(X_test) for c in cvm.top(8)}
    report_test = test_residual_sufficiency(
        residuals=resid_test,
        observed_vars=X_test,
        creator_vars=creator_var_test_values,
        exclude_from_creator_vars=[best_model.creator_var_name],
    )

    a_d_test = test["a_d"].values.astype(float)
    corr_resid_ad_test = np.corrcoef(resid_test, a_d_test)[0, 1]

    # ---- Confirmatory test: fit extended model on TRAIN, evaluate on TEST ----
    a_d_train = train["a_d"].values.astype(float)
    log_phi_train = np.log(best_cv.values)  # best_cv.values are TRAIN values (from cvm.fit)
    log_V_train = np.log(V_train)
    X_ext_train = np.column_stack([log_phi_train, a_d_train, 1.0 / a_d_train])
    reg_ext = LinearRegression().fit(X_ext_train, log_V_train)

    log_phi_test = np.log(cv_test_vals)
    log_V_test = np.log(V_test)
    X_ext_test = np.column_stack([log_phi_test, a_d_test, 1.0 / a_d_test])
    r2_ext_test = _r2(log_V_test, reg_ext.predict(X_ext_test))
    r2_base_test_log = _r2(log_V_test, log_pred_test) if best_model.form == "power_law" else None

    if verbose:
        print(f"  [seed={seed}] train N={len(train)}, test N={len(test)}")
        print(f"  best model: {best_model.creator_var_name} (form={best_model.form}), train R2={best_model.r2:.4f}")
        print(f"  TEST R2 (raw)        = {r2_test_raw:.4f}")
        print(f"  TEST R2 (fit scale)  = {r2_test_scale:.4f}")
        print(f"  Stage4 on TEST resid : {report_test.classification} (max MI={report_test.max_mi_with_observed:.4f} with '{report_test.max_mi_variable}')")
        print(f"  corr(TEST resid, a/d) = {corr_resid_ad_test:.4f}")
        print(f"  TEST R2 base (log)    = {r2_base_test_log:.4f}" if r2_base_test_log is not None else "")
        print(f"  TEST R2 + a/d,1/a_d   = {r2_ext_test:.4f}")
        print(f"  Improvement (held out)= {r2_ext_test - (r2_base_test_log or r2_test_scale):.4f}")

    return {
        "seed": seed, "r2_test_raw": r2_test_raw, "r2_test_scale": r2_test_scale,
        "classification_test": report_test.classification,
        "max_mi_var_test": report_test.max_mi_variable,
        "max_mi_test": report_test.max_mi_with_observed,
        "corr_resid_ad_test": corr_resid_ad_test,
        "r2_base_test_log": r2_base_test_log,
        "r2_ext_test": r2_ext_test,
        "improvement_test": r2_ext_test - (r2_base_test_log if r2_base_test_log is not None else r2_test_scale),
        "form": best_model.form,
    }


def run_wwr_holdout(seed=0, test_frac=0.2, verbose=False):
    wwr = load_wwr()
    train, test = _split(wwr, test_frac, seed)

    vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
    units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}

    X_train = {v: train[v].values.astype(float) for v in vars_}
    V_train = train["V"].values.astype(float)
    X_test = {v: test[v].values.astype(float) for v in vars_}
    V_test = test["V"].values.astype(float)

    def psi(df_):
        return (df_["rho_h"].values.astype(float) * (df_["fyh"].values.astype(float) / df_["fck"].values.astype(float))
                + df_["rho_v"].values.astype(float) * (df_["fyv"].values.astype(float) / df_["fck"].values.astype(float)))

    psi_train = psi(train)
    psi_test = psi(test)
    fck_train = train["fck"].values.astype(float)
    fck_test = test["fck"].values.astype(float)

    # ---- Fit on TRAIN only ----
    cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
    cvm.fit(X_train, V_train, top_k=15)
    best_model, pareto = fit_simplest_equation(cvm.candidates_, V_train, lam=0.01)
    best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)

    # ---- Evaluate on TEST ----
    pred_test, cv_test_vals = _predict(best_model, best_cv, X_test)

    if best_model.form == "power_law":
        target_train = np.log(V_train)
        target_test = np.log(V_test)
        phi_train = np.log(best_cv.values)
        phi_test = np.log(cv_test_vals)
    else:
        target_train = V_train
        target_test = V_test
        phi_train = best_cv.values
        phi_test = cv_test_vals

    base_pred_test = best_model.coef * phi_test + best_model.intercept
    r2_base_test = _r2(target_test, base_pred_test)
    resid_test = target_test - base_pred_test

    creator_var_test_values = {c.name: c.formula(X_test) for c in cvm.top(8)}
    report_test = test_residual_sufficiency(
        residuals=resid_test,
        observed_vars=X_test,
        creator_vars=creator_var_test_values,
        exclude_from_creator_vars=[best_model.creator_var_name],
    )

    corr_resid_psi_test_naive = np.corrcoef(resid_test, psi_test)[0, 1]

    # ---- Confound-controlled confirmatory test: fit on TRAIN, eval on TEST ----
    X_base_train = np.column_stack([phi_train, fck_train])
    X_psi_train = np.column_stack([phi_train, fck_train, psi_train])
    reg_base = LinearRegression().fit(X_base_train, target_train)
    reg_ext = LinearRegression().fit(X_psi_train, target_train)

    X_base_test = np.column_stack([phi_test, fck_test])
    X_psi_test = np.column_stack([phi_test, fck_test, psi_test])
    r2_base_test_fck = _r2(target_test, reg_base.predict(X_base_test))
    r2_ext_test = _r2(target_test, reg_ext.predict(X_psi_test))
    psi_coef_sign_train = np.sign(reg_ext.coef_[2])

    naive_X_train = np.column_stack([phi_train, psi_train])
    reg_naive = LinearRegression().fit(naive_X_train, target_train)
    naive_psi_sign_train = np.sign(reg_naive.coef_[1])

    if verbose:
        print(f"  [seed={seed}] train N={len(train)}, test N={len(test)}")
        print(f"  best model form={best_model.form}, train R2={best_model.r2:.4f}")
        print(f"  Stage4 on TEST resid : {report_test.classification} (max MI={report_test.max_mi_with_observed:.4f} with '{report_test.max_mi_variable}')")
        print(f"  naive corr(TEST resid, Psi_vh) = {corr_resid_psi_test_naive:.4f}")
        print(f"  TEST R2 base+fck   = {r2_base_test_fck:.4f}")
        print(f"  TEST R2 +Psi_vh    = {r2_ext_test:.4f}  (improvement={r2_ext_test - r2_base_test_fck:.4f})")
        print(f"  Psi_vh coef sign (TRAIN-fitted, fck-controlled) = {'positive' if psi_coef_sign_train>0 else 'negative'}")
        print(f"  Psi_vh coef sign (TRAIN-fitted, naive)          = {'positive' if naive_psi_sign_train>0 else 'negative'}")

    return {
        "seed": seed,
        "classification_test": report_test.classification,
        "max_mi_test": report_test.max_mi_with_observed,
        "corr_resid_psi_test_naive": corr_resid_psi_test_naive,
        "r2_base_test_fck": r2_base_test_fck,
        "r2_ext_test": r2_ext_test,
        "improvement_test": r2_ext_test - r2_base_test_fck,
        "psi_sign_fck_controlled": psi_coef_sign_train,
        "psi_sign_naive": naive_psi_sign_train,
        "form": best_model.form,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("WOR HOLD-OUT VALIDATION (single split, verbose)")
    print("=" * 70)
    run_wor_holdout(seed=0, test_frac=0.2, verbose=True)

    print("\n" + "=" * 70)
    print("WOR HOLD-OUT VALIDATION across 20 random 80/20 splits")
    print("=" * 70)
    results = [run_wor_holdout(seed=s, test_frac=0.2, verbose=False) for s in range(20)]
    r2_raw = [r["r2_test_raw"] for r in results]
    corr_ad = [r["corr_resid_ad_test"] for r in results]
    improv = [r["improvement_test"] for r in results]
    classes = [r["classification_test"] for r in results]
    ad_flagged = sum(1 for r in results if r["max_mi_var_test"] == "a")
    from collections import Counter
    print(f"  Test R2 (raw):        mean={np.mean(r2_raw):.4f}  std={np.std(r2_raw):.4f}  "
          f"min={min(r2_raw):.4f}  max={max(r2_raw):.4f}")
    print(f"  corr(test resid,a/d): mean={np.mean(corr_ad):.4f}  std={np.std(corr_ad):.4f}")
    print(f"  Held-out improvement from adding a/d: mean={np.mean(improv):.4f}  std={np.std(improv):.4f}  "
          f"min={min(improv):.4f}")
    print(f"  Stage4 classification counts: {Counter(classes)}")
    print(f"  'a' was the max-MI variable in {ad_flagged}/20 splits")

    print("\n" + "=" * 70)
    print("WWR HOLD-OUT VALIDATION (single split, verbose)")
    print("=" * 70)
    run_wwr_holdout(seed=0, test_frac=0.2, verbose=True)

    print("\n" + "=" * 70)
    print("WWR HOLD-OUT VALIDATION across 20 random 80/20 splits")
    print("=" * 70)
    results_w = [run_wwr_holdout(seed=s, test_frac=0.2, verbose=False) for s in range(20)]
    improv_w = [r["improvement_test"] for r in results_w]
    sign_correct = [r["psi_sign_fck_controlled"] > 0 for r in results_w]
    sign_flip = [r["psi_sign_fck_controlled"] != r["psi_sign_naive"] for r in results_w]
    classes_w = [r["classification_test"] for r in results_w]
    print(f"  Held-out improvement from adding Psi_vh (fck-controlled): "
          f"mean={np.mean(improv_w):.4f}  std={np.std(improv_w):.4f}  min={min(improv_w):.4f}")
    print(f"  Psi_vh sign correct (positive) in {sum(sign_correct)}/20 splits")
    print(f"  Naive vs fck-controlled sign differs in {sum(sign_flip)}/20 splits")
    print(f"  Stage4 classification counts: {Counter(classes_w)}")
