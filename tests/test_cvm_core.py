"""
Unit tests for cvm_core.py (Stage 1: Creator Variable generation, and
the selection-stability diagnostic).
"""
import numpy as np
import pytest
from cvm_core import CVM, CreatorVariable, StabilityReport


def _simple_product_data(seed=0, N=500):
    """y = x1 * x2^2, exactly, with two decoy variables that don't matter."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(1, 10, N)
    x2 = rng.uniform(1, 10, N)
    decoy = rng.uniform(1, 10, N)
    y = x1 * x2 ** 2
    return {"x1": x1, "x2": x2, "decoy": decoy}, y


class TestCandidateGeneration:
    def test_finds_exact_relationship(self):
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=3, max_terms=2)
        cvm.fit(X, y, top_k=5)
        assert len(cvm.candidates_) > 0
        top = cvm.top(1)[0]
        # the true relationship x1*x2^2 should score essentially perfectly
        assert abs(top.correlation) > 0.999

    def test_candidates_are_creator_variable_instances(self):
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=2, max_terms=2)
        cvm.fit(X, y, top_k=5)
        for c in cvm.candidates_:
            assert isinstance(c, CreatorVariable)
            assert c.values.shape == y.shape

    def test_min_abs_corr_threshold_is_respected(self):
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=2, max_terms=2)
        cvm.fit(X, y, top_k=20, min_abs_corr=0.9)
        for c in cvm.candidates_:
            assert abs(c.correlation) >= 0.9

    def test_top_k_is_respected(self):
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=3, max_terms=2)
        cvm.fit(X, y, top_k=3, min_abs_corr=0.0)
        assert len(cvm.candidates_) <= 3


class TestReciprocalCanonicalization:
    def test_reciprocal_pair_keeps_positive_correlation_form(self):
        """
        For a relationship with a reciprocal pair of identical |corr|
        (e.g. y = x1*x2 vs 1/y-equivalent candidate x1^-1 * x2^-1),
        the deduplication step must deterministically keep the
        POSITIVELY correlated member, per cvm_core.py's documented
        tie-break rule.
        """
        rng = np.random.default_rng(1)
        N = 500
        x1 = rng.uniform(1, 10, N)
        x2 = rng.uniform(1, 10, N)
        y = x1 * x2
        X = {"x1": x1, "x2": x2}
        cvm = CVM(variable_names=["x1", "x2"], max_power=2, max_terms=2)
        cvm.fit(X, y, top_k=10, min_abs_corr=0.0)
        top = cvm.top(1)[0]
        assert top.correlation > 0, (
            "Deduplication should retain the positively-correlated "
            "member of a reciprocal pair (cvm_core.py tie-break rule)"
        )


class TestDeduplication:
    def test_near_identical_candidates_collapse(self):
        """Two algebraically-equivalent formulas should not both survive
        as separate top-k entries once |corr with each other| > 0.9999."""
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=3, max_terms=2)
        cvm.fit(X, y, top_k=15, min_abs_corr=0.0)
        values_list = [c.values for c in cvm.candidates_]
        for i in range(len(values_list)):
            for j in range(i + 1, len(values_list)):
                corr = abs(np.corrcoef(values_list[i], values_list[j])[0, 1])
                assert corr <= 0.9999, (
                    f"Candidates {cvm.candidates_[i].name!r} and "
                    f"{cvm.candidates_[j].name!r} were not deduplicated "
                    f"(corr={corr:.6f})"
                )


class TestSelectionStabilityDiagnostic:
    def test_returns_stability_report(self):
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=3, max_terms=2)
        cvm.fit(X, y, top_k=5)
        report = cvm.assess_stability(y, n_boot=100, seed=0)
        assert isinstance(report, StabilityReport)
        assert 0.0 <= report.top_candidate_win_rate <= 1.0
        assert report.n_boot == 100

    def test_clear_winner_is_flagged_stable(self):
        """An exact, dominant relationship (x1*x2^2) with clearly weaker
        decoy-based alternatives should be flagged stable under
        resampling."""
        X, y = _simple_product_data(seed=2, N=800)
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=3, max_terms=2)
        cvm.fit(X, y, top_k=8)
        report = cvm.assess_stability(y, n_boot=200, seed=0)
        assert report.top_candidate_win_rate > 0.9

    def test_raises_if_called_before_fit(self):
        cvm = CVM(variable_names=["x1", "x2"], max_power=2, max_terms=2)
        with pytest.raises(ValueError):
            cvm.assess_stability(np.array([1.0, 2.0, 3.0]))

    def test_win_rates_sum_reasonably(self):
        """Win rates across all candidates should sum to ~1.0 (each
        bootstrap resample produces exactly one winner)."""
        X, y = _simple_product_data()
        cvm = CVM(variable_names=["x1", "x2", "decoy"], max_power=2, max_terms=2)
        cvm.fit(X, y, top_k=5)
        report = cvm.assess_stability(y, n_boot=150, seed=0)
        total = sum(report.win_rates.values())
        assert abs(total - 1.0) < 0.05  # allows for the rare skipped-resample edge case
