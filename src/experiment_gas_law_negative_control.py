"""
NEGATIVE CONTROL: Ideal Gas Law with n Included
===================================================
Same gas law problem, but n IS included in the observed variables.
Specificity check: the framework should NOT claim a missing variable
exists when the variable set is actually complete.
"""

import numpy as np
from cvm_core import CVM
from equation_search import fit_simplest_equation
from residual_diagnostics import test_residual_sufficiency

R_GAS_CONSTANT = 8.314


def run_gas_law_negative_control(seed=0, N=2000, verbose=True):
    rng = np.random.default_rng(seed)

    T = rng.uniform(100, 500, N)
    V = rng.uniform(0.01, 0.05, N)
    n = rng.uniform(0.5, 2.0, N)   # now INCLUDED

    P = (n * R_GAS_CONSTANT * T) / V
    P = P * rng.lognormal(0, 0.005, N)

    X_observed = {'T': T, 'V': V, 'n': n}
    units = {'T': 'K', 'V': 'm^3', 'n': 'mol'}

    cvm = CVM(variable_names=['T', 'V', 'n'], variable_units=units,
              max_power=2, max_terms=3)
    cvm.fit(X_observed, P, top_k=15)

    if verbose:
        print("--- Top Creator Variables (full set, n included) ---")
        for c in cvm.top(5):
            print(f"  {c}")

    best_model, pareto = fit_simplest_equation(cvm.candidates_, P, lam=0.01)

    if verbose:
        print(f"\n--- Best model ---\n  {best_model}")

    creator_var_values = {c.name: c.values for c in cvm.top(5)}
    report = test_residual_sufficiency(
        residuals=best_model.residuals,
        observed_vars=X_observed,
        creator_vars=creator_var_values,
        exclude_from_creator_vars=[best_model.creator_var_name],
    )

    if verbose:
        print(f"\n--- Sufficiency report ---\n  {report}")
        passed = report.classification != "representation_insufficient"
        print(f"  {'PASS' if passed else 'FAIL'}: got '{report.classification}'")

    return report


if __name__ == "__main__":
    print("=" * 60)
    print("NEGATIVE CONTROL: gas law, n included")
    print("=" * 60)
    run_gas_law_negative_control(seed=0, N=2000, verbose=True)

    print("\n\n" + "#" * 60)
    print("ROBUSTNESS: gas law negative control across 10 seeds")
    print("#" * 60)
    false_alarms = 0
    for seed in range(10):
        report = run_gas_law_negative_control(seed=seed, N=2000, verbose=False)
        is_false_alarm = report.classification == "representation_insufficient"
        false_alarms += is_false_alarm
        print(f"  seed={seed}: classification='{report.classification}' "
              f"{'[FALSE ALARM]' if is_false_alarm else '[OK]'}")
    print(f"\n  False alarm rate: {false_alarms}/10")
