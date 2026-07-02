"""
Unit tests for residual_diagnostics.py (Stage 4: residual sufficiency
classification). Mirrors the three cases demonstrated in the module's
own __main__ block, formalized as regression tests.
"""
import numpy as np
from residual_diagnostics import test_residual_sufficiency, _mutual_info_binned


class TestSufficientCase:
    def test_pure_white_noise_classified_sufficient_at_expected_rate(self):
        """A single fixed seed is the wrong test here: Shapiro-Wilk at
        alpha=0.05 will, by design, reject genuinely normal data on
        roughly 5% of seeds (this is the nominal false-positive rate the
        paper itself discusses, Section 2.4). We therefore check the
        classification rate across several seeds rather than asserting
        on one draw that could land in that ~5% tail."""
        n_seeds = 20
        n_sufficient = 0
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            white = rng.normal(0, 1, 600)
            observed = {"x": rng.normal(0, 1, 600)}
            report = test_residual_sufficiency(white, observed_vars=observed)
            if report.classification == "sufficient":
                n_sufficient += 1
        # expect close to (1 - alpha) = 95%; allow generous slack for
        # a small number of seeds
        assert n_sufficient / n_seeds >= 0.80


class TestRepresentationInsufficientCase:
    def test_structure_from_unobserved_variable_flagged(self):
        """Residual structured by a HIDDEN variable, uncorrelated with
        the only observed variable, should be flagged as evidence of a
        missing variable (representation_insufficient)."""
        rng = np.random.default_rng(1)
        N = 600
        hidden = rng.uniform(0, 10, N)
        observed = rng.uniform(0, 10, N)  # independent of hidden
        structured_resid = 0.5 * np.sin(hidden) + rng.normal(0, 0.05, N)
        report = test_residual_sufficiency(structured_resid, observed_vars={"observed": observed})
        assert report.classification == "representation_insufficient"
        assert report.structured is True


class TestModelInsufficientCase:
    def test_structure_from_observed_variable_flagged(self):
        """Residual structured by an OBSERVED variable should be flagged
        as model_insufficient (functional-form problem), not
        representation_insufficient."""
        rng = np.random.default_rng(2)
        N = 600
        obs = rng.uniform(0, 10, N)
        structured_resid = 0.5 * np.sin(obs) + rng.normal(0, 0.05, N)
        report = test_residual_sufficiency(structured_resid, observed_vars={"obs": obs})
        assert report.classification == "model_insufficient"


class TestMutualInformationEstimator:
    def test_independent_variables_give_low_mi(self):
        rng = np.random.default_rng(3)
        a = rng.normal(0, 1, 2000)
        b = rng.normal(0, 1, 2000)
        mi = _mutual_info_binned(a, b, bins=10)
        assert mi < 0.05  # near-zero for genuinely independent variables

    def test_deterministic_relationship_gives_high_mi(self):
        rng = np.random.default_rng(4)
        a = rng.uniform(0, 10, 2000)
        b = a ** 2  # fully deterministic
        mi = _mutual_info_binned(a, b, bins=10)
        assert mi > 1.0  # should approach log(bins) for a fully deterministic relationship

    def test_mi_is_scale_robust_via_quantile_binning(self):
        """Regression test for the documented failure mode: equal-width
        binning inflates MI under a large scale mismatch between two
        INDEPENDENT variables. Quantile (rank) binning should not."""
        rng = np.random.default_rng(5)
        a = rng.normal(0, 1, 2000)          # small scale
        b = rng.normal(0, 1e6, 2000)         # six orders of magnitude larger, still independent
        mi = _mutual_info_binned(a, b, bins=10)
        assert mi < 0.05


class TestBonferroniCorrection:
    def test_more_tested_variables_do_not_inflate_false_positive_classification(self):
        """With many independent observed variables tested simultaneously,
        white noise residuals should still be classified sufficient
        (Bonferroni correction should prevent spurious model_insufficient
        classifications from accumulating false positives)."""
        rng = np.random.default_rng(6)
        N = 500
        white = rng.normal(0, 1, N)
        observed = {f"x{i}": rng.normal(0, 1, N) for i in range(8)}
        report = test_residual_sufficiency(white, observed_vars=observed)
        assert report.classification == "sufficient"
