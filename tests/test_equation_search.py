"""
Unit tests for equation_search.py (Stage 2-3: equation search and
accuracy/complexity model selection).
"""
import numpy as np
from cvm_core import CVM
from equation_search import fit_simplest_equation, FittedModel


class TestLinearVsPowerLawSelection:
    def test_genuine_power_law_prefers_power_law_form(self):
        """y = phi^2 exactly (multiplicative, wide dynamic range) should
        be fit far better in log-space than raw-linear, and the model
        selection should reflect that."""
        rng = np.random.default_rng(0)
        N = 1000
        phi = rng.uniform(1, 100, N)  # wide range -> heteroscedastic if forced linear
        y = phi ** 2 * rng.lognormal(0, 0.02, N)
        cvm = CVM(variable_names=["phi"], max_power=1, max_terms=1)
        cvm.fit({"phi": phi}, y, top_k=3, min_abs_corr=0.0)
        best, _ = fit_simplest_equation(cvm.candidates_, y, lam=0.0)
        assert best.form == "power_law"
        assert best.r2 > 0.99

    def test_genuine_linear_relationship_fits_well(self):
        """y = 3*phi + 5 (additive, not multiplicative) should fit with
        very high R^2 regardless of which form is ultimately selected."""
        rng = np.random.default_rng(1)
        N = 500
        phi = rng.uniform(1, 10, N)
        y = 3 * phi + 5 + rng.normal(0, 0.01, N)
        cvm = CVM(variable_names=["phi"], max_power=1, max_terms=1)
        cvm.fit({"phi": phi}, y, top_k=3, min_abs_corr=0.0)
        best, _ = fit_simplest_equation(cvm.candidates_, y, lam=0.0)
        assert best.r2 > 0.999


class TestFittedModelScore:
    def test_score_applies_complexity_penalty(self):
        model = FittedModel(
            creator_var_name="x1 * x2", coef=1.0, intercept=0.0, r2=0.9,
            complexity=3, predictions=np.array([1.0]), residuals=np.array([0.0]),
            form="linear",
        )
        assert model.score(lam=0.1) == pytest_approx(0.9 - 0.1 * 3)

    def test_higher_complexity_penalized_more_at_equal_r2(self):
        simple = FittedModel(
            creator_var_name="x1", coef=1.0, intercept=0.0, r2=0.8,
            complexity=1, predictions=np.array([1.0]), residuals=np.array([0.0]),
        )
        complex_ = FittedModel(
            creator_var_name="x1 * x2 * x3", coef=1.0, intercept=0.0, r2=0.8,
            complexity=6, predictions=np.array([1.0]), residuals=np.array([0.0]),
        )
        assert simple.score(lam=0.02) > complex_.score(lam=0.02)


def pytest_approx(value, rel=1e-6):
    """Minimal local helper so this file has no external pytest.approx
    import surprises across pytest versions."""
    import pytest as _pytest
    return _pytest.approx(value, rel=rel)
