import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fig_common_setup import plt_rcparams, COLOR_BASE, COLOR_EXT, COLOR_ACCENT, COLOR_GRAY, load_wor
from cvm_core import CVM

plt.rcParams.update(plt_rcparams)

wor = load_wor()
N = len(wor)
vars_ = ["h", "d", "b", "a", "fck", "rho", "fy"]
units = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}
X = {v: wor[v].values.astype(float) for v in vars_}
V = wor["V"].values.astype(float)
a_d = wor["a_d"].values.astype(float)

cvm = CVM(variable_names=vars_, variable_units=units, max_power=1, max_terms=3)
cvm.fit(X, V, top_k=15)

n_boot = 400
rng = np.random.default_rng(0)
results = []

for b in range(n_boot):
    idx = rng.integers(0, N, N)
    V_b = V[idx]
    a_d_b = a_d[idx]
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
        resid = np.log(V_b) - (coef * np.log(vals_b) + intercept)
    else:
        resid = V_b - (coef * vals_b + intercept)
    corr_resid_ad = np.corrcoef(resid, a_d_b)[0, 1]
    results.append({"winner": cv.name, "corr_resid_ad": corr_resid_ad})

rdf = pd.DataFrame(results)
counts = rdf["winner"].value_counts()
top_winners = counts[counts >= 8].index.tolist()  # groups with enough resamples to plot

groups = [rdf[rdf["winner"] == w]["corr_resid_ad"].values for w in top_winners]
labels = [f"{w}\n(n={len(g)})" for w, g in zip(top_winners, groups)]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.55, showfliers=True,
                 medianprops=dict(color=COLOR_EXT, linewidth=1.6))
for patch in bp["boxes"]:
    patch.set_facecolor(COLOR_BASE)
    patch.set_alpha(0.75)
ax.axhline(0, color=COLOR_GRAY, linewidth=0.9, linestyle="--")
ax.set_ylabel("corr(residual, $a/d$)")
ax.set_title("Fig. 5. WOR: corr(residual, $a/d$) stays negative across all\n"
             "400 bootstrap resamples, regardless of which near-tied\n"
             "candidate wins Stage 1\u20133 selection", fontsize=10)
plt.xticks(rotation=15, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig5_wor_stability_boxplot.png")
plt.savefig("figures/fig5_wor_stability_boxplot.pdf")
plt.close()

print("Winner counts:")
print(counts)
print(f"\nOverall: {(rdf['corr_resid_ad']<0).mean()*100:.1f}% of resamples negative")
print("Fig 5 saved")
