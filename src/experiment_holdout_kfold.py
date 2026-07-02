"""
K-FOLD OUT-OF-FOLD VALIDATION: Real Deep Beam Experiments (WOR + WWR)
========================================================================
Fixes a statistical-power problem in the single-split holdout version:
running Stage 4 on a small held-out fold (N~64) starves the permutation-
MI test of power even when the underlying signal (a/d, Psi_vh) is real
and stable.

Correct design: K-fold CV. For each fold, fit CVM (Stage 1) + equation
search (Stage 2-3) on the OTHER folds (training), predict the held-out
fold, and record its residual. Stitch together out-of-fold residuals
for ALL N rows -- genuinely never fit on the model that produced them,
but at FULL sample size. Then run Stage 4 diagnostics once on the
complete out-of-fold residual vector.

This answers the real question: does the pipeline's automatic
diagnostic reliably rediscover a/d / Psi_vh when Stage 1-3's model
choice is never allowed to see the row it's being evaluated on?
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


def _kfold_indices(n, k, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def _r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def run_wor_kfold(k=5, seed=0, verbose=True):
    wor = load_wor()
    N = len(wor)
    vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
    units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}
    V_all = wor["V"].values.astype(float)
    a_d_all = wor["a_d"].values.astype(float)

    folds = _kfold_indices(N, k, seed)

    oof_resid = np.full(N, np.nan)       # out-of-fold residual, on fit-scale (log if power_law chosen in that fold)
    oof_resid_form = [None] * N          # which form was used for that row's fold
    oof_pred_log = np.full(N, np.nan)    # for improvement test
    oof_phi_log = np.full(N, np.nan)
    oof_pred_ext_log = np.full(N, np.nan)  # extended (base + a/d, 1/a_d) out-of-fold prediction
    forms_chosen = []
    train_r2s = []

    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(N), test_idx)
        train = wor.iloc[train_idx]
        test = wor.iloc[test_idx]

        X_train = {v: train[v].values.astype(float) for v in vars_}
        V_train = train["V"].values.astype(float)
        X_test = {v: test[v].values.astype(float) for v in vars_}
        V_test = test["V"].values.astype(float)

        cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
        cvm.fit(X_train, V_train, top_k=15)
        best_model, _ = fit_simplest_equation(cvm.candidates_, V_train, lam=0.01)
        best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
        forms_chosen.append(best_model.form)
        train_r2s.append(best_model.r2)

        cv_test_vals = best_cv.formula(X_test)
        if best_model.form == "power_law":
            log_pred = best_model.coef * np.log(cv_test_vals) + best_model.intercept
            resid = np.log(V_test) - log_pred
            oof_pred_log[test_idx] = log_pred
            oof_phi_log[test_idx] = np.log(cv_test_vals)
        else:
            pred = best_model.coef * cv_test_vals + best_model.intercept
            resid = V_test - pred
            oof_pred_log[test_idx] = np.log(np.clip(pred, 1e-6, None))
            oof_phi_log[test_idx] = np.log(np.clip(cv_test_vals, 1e-6, None))

        oof_resid[test_idx] = resid
        for i in test_idx:
            oof_resid_form[i] = best_model.form

        # Confirmatory extended model computed in this SAME fold pass
        # (reuses this fold's already-fitted best_cv/best_model rather
        # than re-fitting CVM from scratch a second time).
        log_phi_train = np.log(best_cv.values)
        log_V_train = np.log(V_train)
        a_d_train = train["a_d"].values.astype(float)
        X_ext_train = np.column_stack([log_phi_train, a_d_train, 1.0 / a_d_train])
        reg_ext = LinearRegression().fit(X_ext_train, log_V_train)

        a_d_test = test["a_d"].values.astype(float)
        X_ext_test = np.column_stack([np.log(cv_test_vals), a_d_test, 1.0 / a_d_test])
        oof_pred_ext_log[test_idx] = reg_ext.predict(X_ext_test)

    # ---- Stage 4 on the FULL out-of-fold residual vector (N=322) ----
    X_full = {v: wor[v].values.astype(float) for v in vars_}
    # Use a fresh CVM fit on the FULL data just to generate creator-variable
    # candidates to test residuals against (Stage 4 needs *some* candidate
    # set; using the full-data candidates here is standard practice for
    # the diagnostic step and does not leak into the residuals themselves,
    # which were generated out-of-fold).
    cvm_full = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
    cvm_full.fit(X_full, V_all, top_k=15)
    creator_var_full_values = {c.name: c.values for c in cvm_full.top(8)}

    report_oof = test_residual_sufficiency(
        residuals=oof_resid,
        observed_vars=X_full,
        creator_vars=creator_var_full_values,
        exclude_from_creator_vars=[],  # no single "excluded" var since form varies by fold
    )

    corr_resid_ad_oof = np.corrcoef(oof_resid, a_d_all)[0, 1]

    log_V_all = np.log(V_all)
    r2_oof_base_log = _r2(log_V_all, oof_pred_log)
    r2_oof_ext_log = _r2(log_V_all, oof_pred_ext_log)

    if verbose:
        print(f"  k={k} folds, seed={seed}")
        print(f"  form chosen per fold: {forms_chosen}")
        print(f"  train R2 per fold: {[f'{r:.3f}' for r in train_r2s]}")
        print(f"  Stage4 on FULL out-of-fold residual (N={N}): {report_oof.classification} "
              f"(max MI={report_oof.max_mi_with_observed:.4f} with '{report_oof.max_mi_variable}')")
        print(f"  corr(out-of-fold resid, a/d) = {corr_resid_ad_oof:.4f}")
        print(f"  Out-of-fold R2 (log scale), base model      = {r2_oof_base_log:.4f}")
        print(f"  Out-of-fold R2 (log scale), + a/d, 1/a_d     = {r2_oof_ext_log:.4f}")
        print(f"  Out-of-fold improvement from a/d              = {r2_oof_ext_log - r2_oof_base_log:.4f}")

    return {
        "classification_oof": report_oof.classification,
        "max_mi_var_oof": report_oof.max_mi_variable,
        "max_mi_oof": report_oof.max_mi_with_observed,
        "corr_resid_ad_oof": corr_resid_ad_oof,
        "r2_oof_base_log": r2_oof_base_log,
        "r2_oof_ext_log": r2_oof_ext_log,
        "improvement_oof": r2_oof_ext_log - r2_oof_base_log,
    }


def run_wwr_kfold(k=5, seed=0, verbose=True):
    wwr = load_wwr()
    N = len(wwr)
    vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
    units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}
    V_all = wwr["V"].values.astype(float)

    def psi(df_):
        return (df_["rho_h"].values.astype(float) * (df_["fyh"].values.astype(float) / df_["fck"].values.astype(float))
                + df_["rho_v"].values.astype(float) * (df_["fyv"].values.astype(float) / df_["fck"].values.astype(float)))

    psi_all = psi(wwr)
    fck_all = wwr["fck"].values.astype(float)

    folds = _kfold_indices(N, k, seed)
    oof_resid = np.full(N, np.nan)
    oof_target = np.full(N, np.nan)
    oof_pred = np.full(N, np.nan)
    oof_phi = np.full(N, np.nan)
    oof_pred_base_fck = np.full(N, np.nan)
    oof_pred_ext_psi = np.full(N, np.nan)
    forms_chosen = []

    for fi, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(N), test_idx)
        train = wwr.iloc[train_idx]
        test = wwr.iloc[test_idx]
        X_train = {v: train[v].values.astype(float) for v in vars_}
        V_train = train["V"].values.astype(float)
        X_test = {v: test[v].values.astype(float) for v in vars_}
        V_test = test["V"].values.astype(float)

        cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
        cvm.fit(X_train, V_train, top_k=15)
        best_model, _ = fit_simplest_equation(cvm.candidates_, V_train, lam=0.01)
        best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
        forms_chosen.append(best_model.form)

        cv_test_vals = best_cv.formula(X_test)
        if best_model.form == "power_law":
            target_test = np.log(V_test)
            phi_test = np.log(cv_test_vals)
            target_train = np.log(V_train)
            phi_train = np.log(best_cv.values)
        else:
            target_test = V_test
            phi_test = cv_test_vals
            target_train = V_train
            phi_train = best_cv.values
        pred_test = best_model.coef * phi_test + best_model.intercept

        oof_resid[test_idx] = target_test - pred_test
        oof_target[test_idx] = target_test
        oof_pred[test_idx] = pred_test
        oof_phi[test_idx] = phi_test

        # Confound-controlled confirmatory test, computed in this SAME
        # fold pass (reuses this fold's already-fitted best_cv/best_model).
        fck_train = train["fck"].values.astype(float)
        psi_train = psi(train)

        X_base_train = np.column_stack([phi_train, fck_train])
        X_psi_train = np.column_stack([phi_train, fck_train, psi_train])
        reg_base = LinearRegression().fit(X_base_train, target_train)
        reg_ext = LinearRegression().fit(X_psi_train, target_train)

        fck_test = test["fck"].values.astype(float)
        psi_test = psi(test)

        X_base_test = np.column_stack([phi_test, fck_test])
        X_psi_test = np.column_stack([phi_test, fck_test, psi_test])
        oof_pred_base_fck[test_idx] = reg_base.predict(X_base_test)
        oof_pred_ext_psi[test_idx] = reg_ext.predict(X_psi_test)

    r2_oof_base_fck = _r2(oof_target, oof_pred_base_fck)
    r2_oof_ext_psi = _r2(oof_target, oof_pred_ext_psi)

    # ---- Stage 4 on the FULL out-of-fold residual vector (N=518) ----
    X_full = {v: wwr[v].values.astype(float) for v in vars_}
    cvm_full = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
    cvm_full.fit(X_full, V_all, top_k=15)
    creator_var_full_values = {c.name: c.values for c in cvm_full.top(8)}

    report_oof = test_residual_sufficiency(
        residuals=oof_resid,
        observed_vars=X_full,
        creator_vars=creator_var_full_values,
        exclude_from_creator_vars=[],
    )
    corr_resid_psi_naive_oof = np.corrcoef(oof_resid, psi_all)[0, 1]

    if verbose:
        print(f"  k={k} folds, seed={seed}")
        print(f"  form chosen per fold: {forms_chosen}")
        print(f"  Stage4 on FULL out-of-fold residual (N={N}): {report_oof.classification} "
              f"(max MI={report_oof.max_mi_with_observed:.4f} with '{report_oof.max_mi_variable}')")
        print(f"  naive corr(out-of-fold resid, Psi_vh) = {corr_resid_psi_naive_oof:.4f}")
        print(f"  Out-of-fold R2, base+fck   = {r2_oof_base_fck:.4f}")
        print(f"  Out-of-fold R2, +Psi_vh    = {r2_oof_ext_psi:.4f}")
        print(f"  Out-of-fold improvement from Psi_vh (fck-controlled) = {r2_oof_ext_psi - r2_oof_base_fck:.4f}")

    return {
        "classification_oof": report_oof.classification,
        "max_mi_oof": report_oof.max_mi_with_observed,
        "max_mi_var_oof": report_oof.max_mi_variable,
        "corr_resid_psi_naive_oof": corr_resid_psi_naive_oof,
        "r2_oof_base_fck": r2_oof_base_fck,
        "r2_oof_ext_psi": r2_oof_ext_psi,
        "improvement_oof": r2_oof_ext_psi - r2_oof_base_fck,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("WOR: 5-FOLD OUT-OF-FOLD VALIDATION")
    print("=" * 70)
    wor_results = []
    for seed in range(5):
        print(f"\n--- seed={seed} ---")
        r = run_wor_kfold(k=5, seed=seed, verbose=True)
        wor_results.append(r)

    print("\nSummary across 5 different fold assignments:")
    for r in wor_results:
        print(f"  classification={r['classification_oof']:<25} max_mi_var={r['max_mi_var_oof']:<8} "
              f"corr(resid,a/d)={r['corr_resid_ad_oof']:.3f}  improvement={r['improvement_oof']:.4f}")

    print("\n\n" + "=" * 70)
    print("WWR: 5-FOLD OUT-OF-FOLD VALIDATION")
    print("=" * 70)
    wwr_results = []
    for seed in range(5):
        print(f"\n--- seed={seed} ---")
        r = run_wwr_kfold(k=5, seed=seed, verbose=True)
        wwr_results.append(r)

    print("\nSummary across 5 different fold assignments:")
    for r in wwr_results:
        print(f"  classification={r['classification_oof']:<25} max_mi_var={r['max_mi_var_oof']:<25} "
              f"naive_corr(resid,psi)={r['corr_resid_psi_naive_oof']:.3f}  improvement={r['improvement_oof']:.4f}")
