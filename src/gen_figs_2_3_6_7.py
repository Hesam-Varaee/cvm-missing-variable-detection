import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from fig_common_setup import (plt_rcparams, COLOR_BASE, COLOR_EXT, COLOR_ACCENT, COLOR_GRAY,
                                fit_wor, fit_wwr, aci_318_shear_estimate)

plt.rcParams.update(plt_rcparams)

# ---------- Fit WOR ----------
wor, cvm_wor, best_model_wor, best_cv_wor, V_wor = fit_wor()
a_d = wor["a_d"].values.astype(float)

log_V = np.log(V_wor)
log_phi = np.log(best_cv_wor.values)
resid = log_V - (best_model_wor.coef * log_phi + best_model_wor.intercept)  # power_law form

corr_ad = np.corrcoef(resid, a_d)[0, 1]

# extended model
X_ext = np.column_stack([log_phi, a_d, 1.0 / a_d])
reg_ext = LinearRegression().fit(X_ext, log_V)
log_V_pred_ext = reg_ext.predict(X_ext)
V_pred_ext = np.exp(log_V_pred_ext)
V_pred_base = np.exp(best_model_wor.coef * log_phi + best_model_wor.intercept)
r2_base = best_model_wor.r2
r2_ext = reg_ext.score(X_ext, log_V)

# ---------- FIGURE 2: residual vs a/d ----------
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.scatter(a_d, resid, s=18, alpha=0.65, color=COLOR_BASE, edgecolor="white", linewidth=0.3)
z = np.polyfit(a_d, resid, 1)
xs = np.linspace(a_d.min(), a_d.max(), 100)
ax.plot(xs, np.polyval(z, xs), color=COLOR_EXT, linewidth=1.8, label=f"linear fit")
ax.axhline(0, color=COLOR_GRAY, linewidth=0.8, linestyle="--")
ax.set_xlabel("Shear span-to-depth ratio, $a/d$")
ax.set_ylabel("Base-model residual (log scale)")
ax.set_title(f"Fig. 2. WOR residual vs. $a/d$\n(corr = {corr_ad:.2f}, $N$={len(wor)})", fontsize=10)
ax.legend(frameon=False, loc="upper right")
plt.tight_layout()
plt.savefig("figures/fig2_wor_residual_vs_ad.png")
plt.savefig("figures/fig2_wor_residual_vs_ad.pdf")
plt.close()
print(f"Fig 2 saved. corr(resid, a/d) = {corr_ad:.4f}")

# ---------- FIGURE 3: predicted vs actual, base vs extended ----------
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.8), sharex=True, sharey=True)
lims = [min(V_wor.min(), V_pred_base.min(), V_pred_ext.min()) * 0.9,
        max(V_wor.max(), V_pred_base.max(), V_pred_ext.max()) * 1.05]

axes[0].scatter(V_wor, V_pred_base, s=16, alpha=0.6, color=COLOR_BASE, edgecolor="white", linewidth=0.3)
axes[0].plot(lims, lims, color=COLOR_GRAY, linewidth=1.0, linestyle="--")
axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("Actual $V$ (kN)")
axes[0].set_ylabel("Predicted $V$ (kN)")
axes[0].set_title(f"Base model ($h{{\\cdot}}b{{\\cdot}}f_{{ck}}$)\n$R^2$={r2_base:.3f}", fontsize=9.5)

axes[1].scatter(V_wor, V_pred_ext, s=16, alpha=0.6, color=COLOR_EXT, edgecolor="white", linewidth=0.3)
axes[1].plot(lims, lims, color=COLOR_GRAY, linewidth=1.0, linestyle="--")
axes[1].set_xlim(lims); axes[1].set_ylim(lims)
axes[1].set_xlabel("Actual $V$ (kN)")
axes[1].set_title(f"Extended model (+ $a/d$, $1/(a/d)$)\n$R^2$={r2_ext:.3f}", fontsize=9.5)

fig.suptitle("Fig. 3. WOR predicted vs. actual shear capacity, base vs. extended model", fontsize=10, y=1.03)
plt.tight_layout()
plt.savefig("figures/fig3_wor_pred_vs_actual.png")
plt.savefig("figures/fig3_wor_pred_vs_actual.pdf")
plt.close()
print(f"Fig 3 saved. R2 base={r2_base:.4f}  R2 ext={r2_ext:.4f}")

# ---------- FIGURE 7: ACI 318 vs CVM comparison ----------
V_aci = aci_318_shear_estimate(wor)
ratio = V_wor / V_aci
aci_r2 = 1 - np.sum((V_wor - V_aci)**2) / np.sum((V_wor - V_wor.mean())**2)

fig, ax = plt.subplots(figsize=(4.8, 3.6))
ax.hist(ratio, bins=30, color=COLOR_GRAY, alpha=0.85, edgecolor="white")
ax.axvline(1.0, color=COLOR_BASE, linewidth=1.6, linestyle="--", label="Perfect prediction ($V_{test}/V_{ACI}=1$)")
ax.axvline(ratio.mean(), color=COLOR_EXT, linewidth=1.8, label=f"Mean = {ratio.mean():.2f}")
ax.set_xlabel("$V_{test} / V_{ACI}$ (simplified slender-beam estimate)")
ax.set_ylabel("Count")
ax.set_title(f"Fig. 7. ACI 318 slender-beam $V_c$ underestimates\ndeep-beam capacity (ACI $R^2$={aci_r2:.2f})", fontsize=9.5)
ax.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig7_aci_comparison.png")
plt.savefig("figures/fig7_aci_comparison.pdf")
plt.close()
print(f"Fig 7 saved. mean ratio={ratio.mean():.4f}  ACI R2={aci_r2:.4f}")

# ---------- Fit WWR (for Figure 6) ----------
wwr, cvm_wwr, best_model_wwr, best_cv_wwr, V_wwr = fit_wwr()
fck_wwr = wwr["fck"].values.astype(float)
rho_v = wwr["rho_v"].values.astype(float); fyv = wwr["fyv"].values.astype(float)
rho_h = wwr["rho_h"].values.astype(float); fyh = wwr["fyh"].values.astype(float)
psi_vh = rho_h * (fyh / fck_wwr) + rho_v * (fyv / fck_wwr)
corr_psi_fck = np.corrcoef(psi_vh, fck_wwr)[0, 1]

# ---------- FIGURE 6: Psi_vh vs fck confound ----------
fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.scatter(fck_wwr, psi_vh, s=16, alpha=0.6, color=COLOR_ACCENT, edgecolor="white", linewidth=0.3)
z2 = np.polyfit(fck_wwr, psi_vh, 1)
xs2 = np.linspace(fck_wwr.min(), fck_wwr.max(), 100)
ax.plot(xs2, np.polyval(z2, xs2), color=COLOR_EXT, linewidth=1.8)
ax.set_xlabel("Concrete strength, $f_{ck}$ (MPa)")
ax.set_ylabel("Web reinforcement factor, $\\Psi_{vh}$")
ax.set_title(f"Fig. 6. WWR confound: $\\Psi_{{vh}}$ vs. $f_{{ck}}$\n(corr = {corr_psi_fck:.2f}, $N$={len(wwr)})", fontsize=10)
plt.tight_layout()
plt.savefig("figures/fig6_wwr_confound_scatter.png")
plt.savefig("figures/fig6_wwr_confound_scatter.pdf")
plt.close()
print(f"Fig 6 saved. corr(psi_vh, fck) = {corr_psi_fck:.4f}")

print("\nAll of figs 2,3,6,7 done.")
