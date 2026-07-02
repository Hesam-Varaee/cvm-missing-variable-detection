"""Shared setup: fits used across multiple figures, computed once and cached."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from cvm_core import CVM
from equation_search import fit_simplest_equation

plt_rcparams = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}

COLOR_BASE = "#4C72B0"
COLOR_EXT = "#C44E52"
COLOR_ACCENT = "#55A868"
COLOR_GRAY = "#8C8C8C"

def load_wor(path="deep_beam_raw.xlsx"):
    df = pd.read_excel(path, sheet_name="all")
    return df[(df["rho_v"] == 0) & (df["rho_h"] == 0)].copy().reset_index(drop=True)

def load_wwr(path="deep_beam_raw.xlsx"):
    df = pd.read_excel(path, sheet_name="all")
    return df[(df["rho_v"] > 0) | (df["rho_h"] > 0)].copy().reset_index(drop=True)

VARS = ["h", "d", "b", "a", "fck", "rho", "fy"]
UNITS = {"h": "mm", "d": "mm", "b": "mm", "a": "mm", "fck": "MPa", "rho": "dimensionless", "fy": "MPa"}

def fit_wor():
    wor = load_wor()
    X = {v: wor[v].values.astype(float) for v in VARS}
    V = wor["V"].values.astype(float)
    cvm = CVM(variable_names=VARS, variable_units=UNITS, max_power=1, max_terms=3)
    cvm.fit(X, V, top_k=15)
    best_model, _ = fit_simplest_equation(cvm.candidates_, V, lam=0.01)
    best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
    return wor, cvm, best_model, best_cv, V

def fit_wwr():
    wwr = load_wwr()
    X = {v: wwr[v].values.astype(float) for v in VARS}
    V = wwr["V"].values.astype(float)
    cvm = CVM(variable_names=VARS, variable_units=UNITS, max_power=1, max_terms=3)
    cvm.fit(X, V, top_k=15)
    best_model, _ = fit_simplest_equation(cvm.candidates_, V, lam=0.01)
    best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
    return wwr, cvm, best_model, best_cv, V

def aci_318_shear_estimate(wor):
    Vc_N = 0.17 * np.sqrt(wor["fck"]) * wor["b"] * wor["d"]
    return Vc_N / 1000.0
