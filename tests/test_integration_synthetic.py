"""
Integration tests: run the full four-stage pipeline end-to-end and check
that it reproduces the qualitative results reported in the paper. These
are deliberately lighter-weight than the full experiment scripts
(smaller N, fewer seeds) so they run quickly as part of a normal test
suite, while still catching regressions that would break the paper's
reported findings.
"""
import numpy as np
from cvm_core import CVM
from equation_search import fit_simplest_equation
from residual_diagnostics import test_residual_sufficiency
from property_inference import build_hypothesis_card, _parse_unit_powers


def _run_beam_pipeline(seed, N=800, withhold_I=True):
    rng = np.random.default_rng(seed)
    F = rng.uniform(100, 1000, N)
    L = rng.uniform(1, 5, N)
    E = rng.uniform(50e9, 250e9, N)
    I = rng.uniform(1e-7, 1e-5, N)
    deflection = (F * L ** 3) / (3 * E * I)
    deflection = deflection * rng.lognormal(0, 0.01, N)

    if withhold_I:
        X = {"F": F, "L": L, "E": E}
        units = {"F": "N", "L": "m", "E": "Pa"}
        names = ["F", "L", "E"]
    else:
        X = {"F": F, "L": L, "E": E, "I": I}
        units = {"F": "N", "L": "m", "E": "Pa", "I": "m^4"}
        names = ["F", "L", "E", "I"]

    cvm = CVM(variable_names=names, variable_units=units, max_power=3, max_terms=3)
    cvm.fit(X, deflection, top_k=10)
    best_model, _ = fit_simplest_equation(cvm.candidates_, deflection, lam=0.01)

    creator_var_values = {c.name: c.values for c in cvm.top(5)}
    report = test_residual_sufficiency(
        residuals=best_model.residuals, observed_vars=X,
        creator_vars=creator_var_values,
        exclude_from_creator_vars=[best_model.creator_var_name],
    )
    return cvm, best_model, report


class TestBeamPositiveControlIntegration:
    def test_withheld_I_classified_representation_insufficient(self):
        """Matches the paper's Section 3.1 positive-control result:
        withholding I should classify as representation_insufficient."""
        _, _, report = _run_beam_pipeline(seed=0, withhold_I=True)
        assert report.classification == "representation_insufficient"

    def test_recovered_unit_has_length_power_four(self):
        """The paper reports correct recovery of I's unit ([m]^4) in
        10/10 seeds; check this holds for a fixed seed here."""
        cvm, best_model, report = _run_beam_pipeline(seed=0, withhold_I=True)
        best_cv = next(c for c in cvm.candidates_ if c.name == best_model.creator_var_name)
        # observed_vars must match the residual's length (used for
        # interaction-partner inference inside build_hypothesis_card) --
        # reuse the actual data the pipeline was fit on, not dummy arrays.
        rng = np.random.default_rng(0)
        N = 800
        F = rng.uniform(100, 1000, N)
        L = rng.uniform(1, 5, N)
        E = rng.uniform(50e9, 250e9, N)
        card = build_hypothesis_card(
            residuals=best_model.residuals,
            best_creator_var=(best_cv.values if best_model.form == "linear"
                                else np.log(best_cv.values)),
            best_creator_var_units=best_cv.units,
            y_units="[m]",
            observed_vars={"F": F, "L": L, "E": E},
            fitted_coef=best_model.coef,
            is_log_space=(best_model.form == "power_law"),
        )
        unit_powers = _parse_unit_powers(card.inferred_unit)
        assert abs(unit_powers.get("m", 0)) == 4


class TestBeamNegativeControlIntegration:
    def test_included_I_not_classified_representation_insufficient(self):
        """Matches the paper's Section 3.1 negative-control result:
        including I should NOT produce a representation_insufficient
        false alarm."""
        _, _, report = _run_beam_pipeline(seed=0, withhold_I=False)
        assert report.classification != "representation_insufficient"


class TestGasLawIntegration:
    def test_withheld_n_classified_representation_insufficient(self):
        """Matches the paper's Section 3.2 positive-control result for
        the second, structurally different synthetic system."""
        rng = np.random.default_rng(0)
        N = 800
        R_GAS_CONSTANT = 8.314
        T = rng.uniform(100, 500, N)
        V = rng.uniform(0.01, 0.05, N)
        n = rng.uniform(0.5, 2.0, N)
        P = (n * R_GAS_CONSTANT * T) / V
        P = P * rng.lognormal(0, 0.005, N)

        X = {"T": T, "V": V}
        units = {"T": "K", "V": "m^3"}
        cvm = CVM(variable_names=["T", "V"], variable_units=units, max_power=3, max_terms=2)
        cvm.fit(X, P, top_k=10)
        best_model, _ = fit_simplest_equation(cvm.candidates_, P, lam=0.01)

        creator_var_values = {c.name: c.values for c in cvm.top(5)}
        report = test_residual_sufficiency(
            residuals=best_model.residuals, observed_vars=X,
            creator_vars=creator_var_values,
            exclude_from_creator_vars=[best_model.creator_var_name],
        )
        assert report.classification == "representation_insufficient"
        # paper reports the gas law resolves via the power-law path
        assert best_model.form == "power_law"
