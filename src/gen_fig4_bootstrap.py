import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fig_common_setup import plt_rcparams, COLOR_BASE, COLOR_EXT, COLOR_ACCENT, COLOR_GRAY, fit_wwr

plt.rcParams.update(plt_rcparams)

wwr, cvm_wwr, best_model_wwr, best_cv_wwr, V_wwr = fit_wwr()
fck = wwr["fck"].values.astype(float)
rho_v = wwr["rho_v"].values.astype(float); fyv = wwr["fyv"].values.astype(float)
rho_h = wwr["rho_h"].values.astype(float); fyh = wwr["fyh"].values.astype(float)
psi_vh = rho_h * (fyh / fck) + rho_v * (fyv / fck)

if best_model_wwr.form == "power_law":
    target = np.log(V_wwr); phi = np.log(best_cv_wwr.values)
else:
    target = V_wwr; phi = best_cv_wwr.values

N = len(V_wwr)
rng = np.random.default_rng(0)
n_boot = 2000
coefs = np.empty(n_boot)
for b in range(n_boot):
    idx = rng.integers(0, N, N)
    Xb = np.column_stack([np.ones(N), phi[idx], fck[idx], psi_vh[idx]])
    yb = target[idx]
    beta, _, _, _ = np.linalg.lstsq(Xb, yb, rcond=None)
    coefs[b] = beta[3]

frac_pos = np.mean(coefs > 0)
ci_lo, ci_hi = np.percentile(coefs, [2.5, 97.5])

fig, ax = plt.subplots(figsize=(5.2, 3.8))
ax.hist(coefs, bins=40, color=COLOR_ACCENT, alpha=0.85, edgecolor="white")
ax.axvline(0, color=COLOR_GRAY, linewidth=1.4, linestyle="--", label="Zero effect")
ax.axvline(ci_lo, color=COLOR_EXT, linewidth=1.2, linestyle=":", label=f"95% CI [{ci_lo:.0f}, {ci_hi:.0f}]")
ax.axvline(ci_hi, color=COLOR_EXT, linewidth=1.2, linestyle=":")
ax.set_xlabel("Bootstrap $\\Psi_{vh}$ coefficient (fck-controlled, formula fixed)")
ax.set_ylabel("Count (of 2000 resamples)")
ax.set_title(f"Fig. 4. WWR: $\\Psi_{{vh}}$ coefficient is robustly positive\n"
             f"({frac_pos*100:.0f}% of resamples positive)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig("figures/fig4_wwr_bootstrap_coef.png")
plt.savefig("figures/fig4_wwr_bootstrap_coef.pdf")
plt.close()
print(f"Fig 4 saved. frac_positive={frac_pos:.4f}  95% CI=[{ci_lo:.1f}, {ci_hi:.1f}]  mean={coefs.mean():.1f}")
