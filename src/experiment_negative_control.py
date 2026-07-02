"""
NEGATIVE CONTROL: Sufficient Variable Set
============================================
Same beam deflection problem, but I IS included as an observed variable.
The pipeline should now classify residuals as 'sufficient' (or at worst
explain them via I directly), NOT as representation_insufficient -- since
nothing is actually missing. This is the critical specificity check: does
the framework correctly recognize when it does NOT need a new hypothesis,
avoiding false alarms.
"""

import numpy as np
from cvm_core import CVM
from equation_search import fit_simplest_equation
from residual_diagnostics import test_residual_sufficiency


def run_negative_control(seed=0, N=2000, verbose=True):
    rng = np.random.default_rng(seed)

    F = rng.uniform(100, 1000, N)
    L = rng.uniform(1, 5, N)
    E = rng.uniform(50e9, 250e9, N)
    I = rng.uniform(1e-7, 1e-5, N)   # NOW INCLUDED as an observed variable

    deflection = (F * L**3) / (3 * E * I)
    deflection = deflection * rng.lognormal(0, 0.01, N)

    X_observed = {'F': F, 'L': L, 'E': E, 'I': I}   # full variable set
    units = {'F': 'N', 'L': 'm', 'E': 'Pa', 'I': 'm^4'}

    cvm = CVM(variable_names=['F', 'L', 'E', 'I'], variable_units=units,
              max_power=2, max_terms=4)
    cvm.fit(X_observed, deflection, top_k=15)

    if verbose:
        print("--- Top Creator Variables (full variable set, I included) ---")
        for c in cvm.top(5):
            print(f"  {c}")

    best_model, pareto = fit_simplest_equation(cvm.candidates_, deflection, lam=0.01)

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
        expected = "sufficient or model_insufficient (NOT representation_insufficient)"
        passed = report.classification != "representation_insufficient"
        print(f"\n  Expected: {expected}")
        print(f"  {'PASS' if passed else 'FAIL'}: got '{report.classification}'")

    return report


if __name__ == "__main__":
    print("=" * 60)
    print("NEGATIVE CONTROL: I included -- should NOT flag missing variable")
    print("=" * 60)
    run_negative_control(seed=0, N=2000, verbose=True)

    print("\n\n" + "#" * 60)
    print("ROBUSTNESS: negative control across 10 seeds")
    print("#" * 60)
    false_alarms = 0
    for seed in range(10):
        report = run_negative_control(seed=seed, N=2000, verbose=False)
        is_false_alarm = report.classification == "representation_insufficient"
        false_alarms += is_false_alarm
        print(f"  seed={seed}: classification='{report.classification}' "
              f"{'[FALSE ALARM]' if is_false_alarm else '[OK]'}")
    print(f"\n  False alarm rate: {false_alarms}/10")
