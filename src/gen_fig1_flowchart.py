import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from fig_common_setup import plt_rcparams, COLOR_BASE, COLOR_EXT, COLOR_ACCENT, COLOR_GRAY

plt.rcParams.update(plt_rcparams)

fig, ax = plt.subplots(figsize=(8.4, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis("off")

def box(x, y, w, h, text, color, fontsize=9, textcolor="white"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.92)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize,
             color=textcolor, weight="bold", linespacing=1.5)

def varrow(x, y1, y2, color=COLOR_GRAY):
    ax.add_patch(FancyArrowPatch((x, y1), (x, y2), arrowstyle="-|>", mutation_scale=14,
                                   linewidth=1.4, color=color))

def diag_arrow(p1, p2, color=COLOR_GRAY, style="-|>"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12,
                                   linewidth=1.2, color=color,
                                   connectionstyle="arc3,rad=0.0"))

cx = 4.6  # center x of main column
w_main = 7.2
x0 = cx - w_main/2

box(x0, 10.3, w_main, 1.1, "Observed variables  {$x_1,\\ldots,x_m$}, $y$", COLOR_GRAY)
box(x0, 8.6, w_main, 1.1, "Stage 1: Creator Variable Generation\nalgebraic search  $\\phi=\\prod x_i^{e_i}$ (raw + log-space correlation)", COLOR_BASE)
box(x0, 6.9, w_main, 1.1, "Stage 2\u20133: Equation Search & Model Selection\nlinear vs. power-law fit, complexity-penalized score", COLOR_BASE)
box(x0, 5.2, w_main, 1.1, "Stage 4: Residual Sufficiency Classification\npermutation MI + Shapiro test, Bonferroni-corrected", COLOR_BASE)

varrow(cx, 10.3, 9.7)
varrow(cx, 8.6, 8.0)
varrow(cx, 6.9, 6.3)

# Three-way branch from Stage 4
y4_bottom = 5.2
diag_arrow((cx - 2.6, y4_bottom), (1.6, 3.9))
diag_arrow((cx, y4_bottom), (cx, 3.9))
diag_arrow((cx + 2.6, y4_bottom), (7.8, 3.9))

ax.text(1.6, 3.65, "sufficient", ha="center", fontsize=8.5, style="italic", color=COLOR_GRAY)
ax.text(cx, 3.65, "model\ninsufficient", ha="center", fontsize=8.5, style="italic", color=COLOR_GRAY, linespacing=1.3)
ax.text(7.8, 3.65, "representation\ninsufficient", ha="center", fontsize=8.5, style="italic", color=COLOR_EXT, linespacing=1.3)

box(6.2, 1.7, 3.2, 1.2, "Hypothesis Card\nunit \u00b7 direction \u00b7 interaction\nscale \u00b7 smoothness", COLOR_EXT, fontsize=8.5)
diag_arrow((7.8, 3.5), (7.8, 2.9))

# Side panel: stability diagnostic, attached to Stage 2-3
box(9.3, 6.9, 2.5, 1.5, "Selection Stability\nDiagnostic\n(bootstrap resample\nof top-$k$ candidates)", COLOR_ACCENT, fontsize=7.8)
diag_arrow((x0 + w_main, 7.45), (9.3, 7.45), color=COLOR_ACCENT, style="<|-|>")

ax.text(6.05, 0.35,
        "Fig. 1. CVM four-stage workflow. The selection-stability diagnostic (green) audits\n"
        "the Stage 2\u20133 model choice that all downstream inference is conditioned on.",
        ha="center", fontsize=8.5, color="black")

plt.tight_layout()
plt.savefig("figures/fig1_pipeline_flowchart.png")
plt.savefig("figures/fig1_pipeline_flowchart.pdf")
print("Fig 1 saved")
