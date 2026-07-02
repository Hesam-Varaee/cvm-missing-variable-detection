"""
Check #2: does WOR's Stage-1 near-tie (h*b*fck vs d*b*fck vs d^-1*b^-1*fck^-1
vs h^-1*b^-1*fck^-1, all within ~0.01 correlation of each other) actually
change the DOWNSTREAM conclusion (a/d as the flagged missing signal, and
the R^2 improvement from adding it) -- or is it harmless because h and d
are highly correlated with each other regardless of which one gets picked?

Method: bootstrap-resample WOR rows. On each resample, re-run Stage 2-3
selection among the already-found top-k candidates (same logic as
fit_simplest_equation), record which candidate wins, then compute
corr(residual, a/d) and the R^2 improvement from adding a/d for THAT
resample's winning model. Group results by which candidate won to see if
the downstream conclusion actually depends on the Stage-1/2-3 selection.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from cvm_core import CVM
from equation_search import fit_simplest_equation

df = pd.read_excel("deep_beam_raw.xlsx", sheet_name="all")
wor = df[(df["rho_v"] == 0) & (df["rho_h"] == 0)].copy().reset_index(drop=True)
N = len(wor)

vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}
X = {v: wor[v].values.astype(float) for v in vars_}
V = wor["V"].values.astype(float)
a_d = wor["a_d"].values.astype(float)

cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
cvm.fit(X, V, top_k=15)
print("Top candidates (full data):")
for c in cvm.top(6):
    print(" ", c)

n_boot = 400
rng = np.random.default_rng(0)
results = []

for b in range(n_boot):
    idx = rng.integers(0, N, N)
    V_b = V[idx]
    a_d_b = a_d[idx]

    # Re-run Stage 2-3 selection among the fixed top-k candidates on this resample
    best_score, best = -np.inf, None
    for cv in cvm.candidates_:
        vals_b = cv.values[idx]
        if np.std(vals_b) < 1e-12:
            continue
        Xb = vals_b.reshape(-1, 1)
        reg_lin = LinearRegression().fit(Xb, V_b)
        r2_lin = reg_lin.score(Xb, V_b)
        form, r2, coef, intercept = "linear", r2_lin, reg_lin.coef_[0], reg_lin.intercept_
        if np.all(vals_b > 0) and np.all(V_b > 0):
            log_xb = np.log(vals_b).reshape(-1, 1)
            log_Vb = np.log(V_b)
            reg_log = LinearRegression().fit(log_xb, log_Vb)
            r2_log = reg_log.score(log_xb, log_Vb)
            if r2_log > r2:
                form, r2, coef, intercept = "power_law", r2_log, reg_log.coef_[0], reg_log.intercept_
        score = r2 - 0.01 * cv.complexity
        if score > best_score:
            best_score = score
            best = (cv, form, r2, coef, intercept, vals_b)

    cv, form, r2, coef, intercept, vals_b = best
    if form == "power_law":
        log_pred = coef * np.log(vals_b) + intercept
        resid = np.log(V_b) - log_pred
        target = np.log(V_b)
        phi = np.log(vals_b)
    else:
        pred = coef * vals_b + intercept
        resid = V_b - pred
        target = V_b
        phi = vals_b

    corr_resid_ad = np.corrcoef(resid, a_d_b)[0, 1]

    X_ext = np.column_stack([phi, a_d_b, 1.0 / a_d_b])
    reg_ext = LinearRegression().fit(X_ext, target)
    r2_ext = reg_ext.score(X_ext, target)
    improvement = r2_ext - r2

    results.append({"winner": cv.name, "form": form, "corr_resid_ad": corr_resid_ad,
                     "improvement": improvement, "r2_base": r2})

rdf = pd.DataFrame(results)
print(f"\nn_boot={n_boot}")
print("\nWinner distribution:")
print(rdf["winner"].value_counts())

print("\nGrouped by winner: corr(resid,a/d) and improvement stats")
print(rdf.groupby("winner").agg(
    n=("winner", "size"),
    corr_mean=("corr_resid_ad", "mean"),
    corr_std=("corr_resid_ad", "std"),
    improvement_mean=("improvement", "mean"),
    improvement_std=("improvement", "std"),
))

print("\nOverall (all winners pooled):")
print(f"  corr(resid,a/d): mean={rdf['corr_resid_ad'].mean():.4f} std={rdf['corr_resid_ad'].std():.4f} "
      f"min={rdf['corr_resid_ad'].min():.4f} max={rdf['corr_resid_ad'].max():.4f}")
print(f"  improvement from a/d: mean={rdf['improvement'].mean():.4f} std={rdf['improvement'].std():.4f} "
      f"min={rdf['improvement'].min():.4f} max={rdf['improvement'].max():.4f}")
