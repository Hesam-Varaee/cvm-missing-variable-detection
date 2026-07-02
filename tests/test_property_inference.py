"""
Unit tests for property_inference.py (Hypothesis Card generation).
"""
import numpy as np
from property_inference import (
    _parse_unit_powers, _decompose_to_base_units, infer_direction,
    infer_interactions, infer_smoothness, build_hypothesis_card, HypothesisCard,
)


class TestUnitParsing:
    def test_parses_simple_unit(self):
        assert _parse_unit_powers("[N]") == {"N": 1}

    def test_parses_power(self):
        assert _parse_unit_powers("[m]^3") == {"m": 3}

    def test_parses_compound_unit(self):
        result = _parse_unit_powers("[N] * [m]^3")
        assert result == {"N": 1, "m": 3}

    def test_parses_negative_power(self):
        result = _parse_unit_powers("[m]^-1 * [s]^-2")
        assert result == {"m": -1, "s": -2}


class TestSIBaseDecomposition:
    def test_pascal_decomposes_to_base_units(self):
        # Pa = kg * m^-1 * s^-2
        result = _decompose_to_base_units({"Pa": 1})
        assert result == {"kg": 1, "m": -1, "s": -2}

    def test_newton_decomposes_to_base_units(self):
        # N = kg * m * s^-2
        result = _decompose_to_base_units({"N": 1})
        assert result == {"kg": 1, "m": 1, "s": -2}

    def test_base_unit_passes_through(self):
        assert _decompose_to_base_units({"m": 4}) == {"m": 4}

    def test_unknown_unit_passes_through_unchanged(self):
        # domain-specific units not in the SI table (e.g. 'mm', 'MPa' as
        # used literally in the real-data experiments) should not crash
        # and should be treated as already-irreducible
        assert _decompose_to_base_units({"mm": 2}) == {"mm": 2}

    def test_mixing_derived_and_base_units_tracks_length_correctly(self):
        """This is the specific failure mode the decomposition exists to
        prevent: E in Pa (secretly containing m^-1) combined with L in m
        must not silently drop the hidden length power."""
        # E (Pa) * L (m)^3 -> should NOT simplify to just m^3
        combined = {"Pa": 1, "m": 3}
        result = _decompose_to_base_units(combined)
        # Pa contributes m^-1, explicit m contributes m^3 -> net m^2
        assert result["m"] == -1 + 3


class TestDirectionInference:
    def test_returns_cannot_determine_when_uncorrelated(self):
        rng = np.random.default_rng(0)
        N = 1000
        phi = rng.uniform(1, 10, N)
        residual = rng.normal(0, 1, N)  # independent of phi by construction
        direction, confidence = infer_direction(residual, phi)
        assert "cannot be determined" in direction
        assert confidence == "not applicable"

    def test_detects_positive_effect(self):
        rng = np.random.default_rng(1)
        N = 1000
        phi = rng.uniform(1, 10, N)
        residual = 2.0 * phi + rng.normal(0, 0.1, N)
        direction, confidence = infer_direction(residual, phi)
        assert "increases" in direction or "positive" in direction

    def test_detects_negative_effect(self):
        rng = np.random.default_rng(2)
        N = 1000
        phi = rng.uniform(1, 10, N)
        residual = -2.0 * phi + rng.normal(0, 0.1, N)
        direction, confidence = infer_direction(residual, phi)
        assert "decreases" in direction or "negative" in direction or "suppressive" in direction

    def test_invariant_to_reciprocal_form_via_fitted_coef_anchor(self):
        """Direction inference must give the SAME answer whether CVM
        happened to select phi or its reciprocal 1/phi, as long as the
        fitted_coef sign correctly reflects which form was used."""
        rng = np.random.default_rng(3)
        N = 1000
        phi = rng.uniform(1, 10, N)
        residual = 2.0 * phi + rng.normal(0, 0.1, N)

        direction_phi, _ = infer_direction(residual, phi, fitted_coef=1.0)
        direction_recip, _ = infer_direction(residual, 1.0 / phi, fitted_coef=-1.0)
        assert direction_phi == direction_recip


class TestInteractionInference:
    def test_finds_correct_multiplicative_partner(self):
        rng = np.random.default_rng(4)
        N = 1000
        partner = rng.uniform(1, 10, N)
        decoy = rng.uniform(1, 10, N)
        residual = partner * rng.normal(0, 0.3, N)  # variance scales with partner
        best, confidence, scores = infer_interactions(residual, {"partner": partner, "decoy": decoy})
        assert best == "partner"

    def test_returns_none_for_no_observed_vars(self):
        residual = np.random.normal(0, 1, 100)
        best, confidence, scores = infer_interactions(residual, {})
        assert best is None
        assert confidence == "low"


class TestSmoothnessInference:
    def test_runs_without_error_and_returns_valid_category(self):
        residual = np.random.default_rng(5).normal(0, 1, 500)
        smoothness, confidence = infer_smoothness(residual)
        assert smoothness in (
            "low-frequency (bulk property)", "mixed", "high-frequency (local/surface effect)"
        )


class TestBuildHypothesisCard:
    def test_returns_complete_hypothesis_card(self):
        rng = np.random.default_rng(6)
        N = 500
        F = rng.uniform(100, 1000, N)
        L = rng.uniform(1, 5, N)
        best_cv = F * L ** 3
        residual = rng.normal(0, np.std(best_cv) * 0.01, N)

        card = build_hypothesis_card(
            residuals=residual,
            best_creator_var=best_cv,
            best_creator_var_units="[N] * [m]^3",
            y_units="[m]",
            observed_vars={"F": F, "L": L},
        )
        assert isinstance(card, HypothesisCard)
        assert card.inferred_unit  # non-empty string
        assert card.direction
        assert card.smoothness
        assert isinstance(card.scale_estimate, float)

    def test_log_space_flag_annotates_scale_confidence(self):
        rng = np.random.default_rng(7)
        N = 500
        F = rng.uniform(100, 1000, N)
        best_cv = np.log(F)
        residual = rng.normal(0, 0.1, N)
        card = build_hypothesis_card(
            residuals=residual, best_creator_var=best_cv,
            best_creator_var_units="[N]", y_units="[m]",
            observed_vars={"F": F}, is_log_space=True,
        )
        assert "Var(log z*)" in card.scale_confidence
