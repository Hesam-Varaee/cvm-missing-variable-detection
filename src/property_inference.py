"""
Missing Variable Property Inference Engine
=============================================
When residuals are classified as 'representation_insufficient', this module
infers the PROPERTIES the missing variable z* must have, without ever
identifying z* itself.

Five inferred properties:
  1. Dimensional unit  (from dimensional balancing against best Creator Variable)
  2. Directionality    (sign of correlation -> monotone increase/decrease)
  3. Interaction structure (which observed variables z* couples with)
  4. Scale / magnitude  (bound on Var(z*) from Var(residual))
  5. Smoothness         (low vs high frequency structure -> bulk vs local effect)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import re


@dataclass
class HypothesisCard:
    """The final output: a structured hypothesis about the missing variable."""
    inferred_unit: str
    unit_confidence: str
    direction: str
    direction_confidence: str
    interaction_partner: Optional[str]
    interaction_confidence: str
    scale_estimate: float
    scale_confidence: str
    smoothness: str
    smoothness_confidence: str
    raw_scores: Dict = field(default_factory=dict)

    def display(self):
        lines = [
            "=" * 60,
            "HYPOTHESIS CARD: properties of the missing variable z*",
            "=" * 60,
            f"  Unit            : {self.inferred_unit:<20} [{self.unit_confidence}]",
            f"  Direction       : {self.direction:<20} [{self.direction_confidence}]",
            f"  Interacts with  : {str(self.interaction_partner):<20} [{self.interaction_confidence}]",
            f"  Scale (Var est.): {self.scale_estimate:<20.4g} [{self.scale_confidence}]",
            f"  Smoothness      : {self.smoothness:<20} [{self.smoothness_confidence}]",
            "=" * 60,
        ]
        return "\n".join(lines)


def _parse_unit_powers(unit_string):
    """Parse a unit string like '[N] * [m]^3' into {'N': 1, 'm': 3}."""
    powers = {}
    tokens = re.findall(r"\[(\w+)\](?:\^(-?\d+))?", unit_string)
    for base, exp in tokens:
        e = int(exp) if exp else 1
        powers[base] = powers.get(base, 0) + e
    return powers


# SI base-unit decomposition for common derived units used in this domain.
# Dimensional inference (subtracting unit-power dicts) is only valid if
# every unit is expressed in a common, irreducible basis. Without this,
# e.g. 'Pa' and 'm' are treated as independent symbols, but Pa = kg/(m*s^2)
# secretly contains an 'm' that must be accounted for -- silently omitting
# it produces wrong power-of-length results in any formula that mixes a
# derived unit (Pa, N, J, W, ...) with a base unit (m) directly, which is
# exactly the beam deflection case (E in Pa, L in m).
_SI_BASE_DECOMPOSITION = {
    'N': {'kg': 1, 'm': 1, 's': -2},           # Newton = kg*m/s^2
    'Pa': {'kg': 1, 'm': -1, 's': -2},          # Pascal = N/m^2 = kg/(m*s^2)
    'J': {'kg': 1, 'm': 2, 's': -2},            # Joule = N*m
    'W': {'kg': 1, 'm': 2, 's': -3},            # Watt = J/s
    # Base / already-irreducible units decompose to themselves
    'm': {'m': 1}, 'kg': {'kg': 1}, 's': {'s': 1},
}


def _decompose_to_base_units(powers: Dict[str, int]) -> Dict[str, int]:
    """
    Expand a unit-power dict that may contain derived units (N, Pa, J, W)
    into pure SI base units (kg, m, s). Units not found in the
    decomposition table are passed through unchanged (treated as already
    irreducible / domain-specific base units).
    """
    base_powers = {}
    for unit, power in powers.items():
        decomposition = _SI_BASE_DECOMPOSITION.get(unit, {unit: 1})
        for base, base_power in decomposition.items():
            base_powers[base] = base_powers.get(base, 0) + power * base_power
    return {k: v for k, v in base_powers.items() if v != 0}


def infer_unit(residuals, best_creator_var, y_unit_powers, cv_unit_powers,
                fitted_coef=None, is_log_space=False):
    """
    Property 1: Dimensional unit inference.

    The dimensional balancing equation U_y = U_phi^sign * U_z assumes phi*
    enters the fitted model with a POSITIVE exponent. But CVM may have
    selected the reciprocal form 1/phi* as its best candidate (algebraically
    equivalent, identical |correlation|), in which case the unit powers of
    cv_unit_powers are already negated relative to the "natural" form -- and
    the formula U_z = U_y - U_phi (in log/power space) would then produce
    the WRONG sign for every unit power.

    Fix: anchor on the SIGN OF THE FITTED COEFFICIENT, which tells us
    whether the selected creator variable (as stored, including whichever
    reciprocal form was kept) has a positive or negative power-law exponent
    in the actual fitted equation. If fitted_coef > 0, use cv_unit_powers
    as given. If fitted_coef < 0, the creator variable is acting with an
    effective NEGATIVE exponent in the equation, so we negate cv_unit_powers
    before subtracting -- restoring the correct dimensional relationship
    regardless of which reciprocal form CVM happened to select.
    """
    effective_cv_powers = dict(cv_unit_powers)
    if fitted_coef is not None and fitted_coef < 0:
        effective_cv_powers = {k: -v for k, v in cv_unit_powers.items()}

    # Decompose into SI base units (kg, m, s) before subtracting -- mixing
    # derived units (N, Pa) with base units (m) directly, without expanding
    # them to a common basis, silently drops length-power contributions
    # hidden inside the derived units (Pa = kg/(m*s^2) secretly has m^-1).
    y_base = _decompose_to_base_units(y_unit_powers)
    cv_base = _decompose_to_base_units(effective_cv_powers)

    z_powers = {}
    all_bases = set(y_base) | set(cv_base)
    for base in all_bases:
        z_powers[base] = y_base.get(base, 0) - cv_base.get(base, 0)
    z_powers = {k: v for k, v in z_powers.items() if v != 0}

    unit_str = " * ".join(f"[{b}]^{p}" if p != 1 else f"[{b}]"
                            for b, p in z_powers.items() if p != 0)
    if not unit_str:
        unit_str = "[dimensionless]"

    corr = np.corrcoef(residuals, best_creator_var)[0, 1]

    if is_log_space:
        if abs(corr) > 0.3:
            confidence = "medium (assumes unit power-law exponent)"
        else:
            confidence = "low (assumes unit power-law exponent)"
    else:
        if abs(corr) > 0.5:
            confidence = "high"
        elif abs(corr) > 0.2:
            confidence = "medium"
        else:
            confidence = "low"

    return unit_str, confidence, z_powers


def infer_direction(residuals, best_creator_var, fitted_coef=None, is_log_space=False):
    """
    Property 2: Directionality / monotonicity.

    The critical subtlety: CVM may select EITHER phi* or its reciprocal
    1/phi* as the best Creator Variable (both score identically), and which
    one it picks changes the SIGN of everything downstream unless we
    anchor to something that doesn't flip arbitrarily with that choice.

    The anchor is the fitted model itself: y (or log y) ~ fitted_coef *
    best_creator_var + intercept. If fitted_coef > 0, phi* (as selected)
    has a positive effect on y; the residual is what's LEFT OVER after
    that. A residual that is negatively correlated with the FITTED
    PREDICTION (not with best_creator_var directly, which can have either
    sign depending on which reciprocal form was kept) indicates the
    missing variable opposes the explained part of y -- a suppressive
    /negative effect. This formulation is invariant to which reciprocal
    form CVM happened to select, because it always asks the right
    question: does the leftover residual move WITH or AGAINST the
    explained trend.
    """
    if fitted_coef is not None:
        signed_var = best_creator_var * np.sign(fitted_coef)
    else:
        signed_var = best_creator_var

    slope = np.polyfit(signed_var, residuals, 1)[0]
    r = np.corrcoef(signed_var, residuals)[0, 1]

    # If the residual is essentially uncorrelated with the creator variable
    # (which happens precisely when the missing variable is statistically
    # independent of the observed variables, e.g. a true confound rather
    # than an interacting factor), this slope test has NO signal to work
    # with -- reporting a sign here would just be amplifying noise. We are
    # explicit about this rather than forcing a guess: in this regime,
    # directionality must be inferred from a different signal (residual
    # sign asymmetry, domain priors, or simply acknowledged as unknown).
    if abs(r) < 0.05:
        direction = "cannot be determined from this signal (residual independent of phi*)"
        confidence = "not applicable"
        return direction, confidence

    if slope > 0:
        direction = "increases y (positive effect)"
    else:
        direction = "decreases y (negative / suppressive effect)"

    if abs(r) > 0.5:
        confidence = "high"
    elif abs(r) > 0.2:
        confidence = "medium"
    else:
        confidence = "low"

    return direction, confidence


def infer_interactions(residuals, observed_vars: Dict[str, np.ndarray]):
    """
    Property 3: Interaction structure.

    For each observed variable x_i, test corr(residual, x_i) directly AND
    corr(residual^2, x_i) -- the latter catches multiplicative/heteroscedastic
    coupling where the mean effect is weak but the VARIANCE of the residual
    scales with x_i (a strong sign that z* multiplies x_i in the true model).
    """
    scores = {}
    for name, vals in observed_vars.items():
        linear_corr = abs(np.corrcoef(residuals, vals)[0, 1])
        variance_corr = abs(np.corrcoef(residuals**2, vals)[0, 1])
        scores[name] = max(linear_corr, variance_corr)

    if not scores:
        return None, "low", scores

    best_partner = max(scores, key=scores.get)
    best_score = scores[best_partner]

    if best_score > 0.4:
        confidence = "high"
    elif best_score > 0.15:
        confidence = "medium"
    else:
        confidence = "low"

    return best_partner, confidence, scores


def infer_scale(residuals, coupling_values=None, is_log_space=False):
    """
    Property 4: Scale / magnitude bound.

    Raw-scale case: if residual ~ z* * h(X) for coupling h, then
    Var(residual) ~= Var(z*) * E[h(X)^2], so we back out Var(z*) by
    dividing by the mean squared coupling.

    Log-space case: the fitted model is log y = a*log(phi) + b, so the
    residual IS already log(z*) (up to an additive constant and noise) --
    there is no separate multiplicative coupling factor to divide out.
    Var(residual) directly estimates Var(log z*) and should be reported
    as-is, not divided by anything.
    """
    var_resid = np.var(residuals)

    if is_log_space:
        return var_resid, "medium"

    if coupling_values is not None and np.var(coupling_values) > 1e-12:
        mean_h2 = np.mean(coupling_values**2)
        var_z_estimate = var_resid / mean_h2 if mean_h2 > 1e-12 else var_resid
        confidence = "medium"
    else:
        var_z_estimate = var_resid
        confidence = "low"

    return var_z_estimate, confidence


def infer_smoothness(residuals):
    """
    Property 5: Smoothness / variability structure.

    Use the ratio of high-frequency to low-frequency power in the residual
    spectrum (after sorting is irrelevant here since data are iid samples,
    not time series -- so instead we use a local-variance test: compare
    variance of residuals to variance of a locally-smoothed version of the
    SAME residuals when sorted by their own magnitude rank as a proxy for
    "structure granularity"). This is a simplified diagnostic appropriate
    for cross-sectional (non-temporal) data.
    """
    sorted_resid = np.sort(residuals)
    n = len(sorted_resid)
    window = max(3, n // 20)
    smoothed = np.convolve(sorted_resid, np.ones(window) / window, mode='valid')
    high_freq_power = np.var(sorted_resid[:len(smoothed)] - smoothed)
    low_freq_power = np.var(smoothed)
    ratio = high_freq_power / (low_freq_power + 1e-12)

    if ratio < 0.1:
        smoothness = "low-frequency (bulk property)"
        confidence = "high"
    elif ratio < 0.5:
        smoothness = "mixed"
        confidence = "medium"
    else:
        smoothness = "high-frequency (local/surface effect)"
        confidence = "medium"

    return smoothness, confidence


def build_hypothesis_card(residuals: np.ndarray,
                            best_creator_var: np.ndarray,
                            best_creator_var_units: str,
                            y_units: str,
                            observed_vars: Dict[str, np.ndarray],
                            fitted_coef: float = None,
                            is_log_space: bool = False) -> HypothesisCard:
    """
    Master function: runs all five property inferences and assembles
    the final Hypothesis Card.

    fitted_coef: the coefficient from the upstream model fit
    (y ~ fitted_coef * best_creator_var + intercept, or in log space,
    log y ~ fitted_coef * log(best_creator_var) + intercept). This anchors
    the sign convention for unit and direction inference, making both
    invariant to whether CVM selected phi* or its reciprocal 1/phi* as the
    best candidate (they are algebraically equivalent but have opposite
    unit-power and slope signs unless anchored to the fit itself).

    is_log_space: True when the upstream model was fit as a power law
    (log y ~ a*log(phi) + b). In that case 'residuals' and
    'best_creator_var' are BOTH already in log space (the caller is
    responsible for passing log(phi) as best_creator_var to match).
    Scale inference in log-space estimates Var(log z*), i.e. the squared
    coefficient of variation of z*, NOT Var(z*) directly -- this is
    reported explicitly to avoid misleading comparisons to a raw-scale
    variance.
    """
    y_powers = _parse_unit_powers(y_units)
    cv_powers = _parse_unit_powers(best_creator_var_units)

    unit_str, unit_conf, _ = infer_unit(residuals, best_creator_var, y_powers, cv_powers,
                                          fitted_coef=fitted_coef, is_log_space=is_log_space)
    direction, dir_conf = infer_direction(residuals, best_creator_var,
                                            fitted_coef=fitted_coef, is_log_space=is_log_space)
    partner, partner_conf, all_scores = infer_interactions(residuals, observed_vars)
    scale, scale_conf = infer_scale(residuals, coupling_values=best_creator_var, is_log_space=is_log_space)
    if is_log_space:
        scale_conf = scale_conf + " (this is Var(log z*), i.e. squared CV, not Var(z*))"
    smoothness, smooth_conf = infer_smoothness(residuals)

    return HypothesisCard(
        inferred_unit=unit_str,
        unit_confidence=unit_conf,
        direction=direction,
        direction_confidence=dir_conf,
        interaction_partner=partner,
        interaction_confidence=partner_conf,
        scale_estimate=scale,
        scale_confidence=scale_conf,
        smoothness=smoothness,
        smoothness_confidence=smooth_conf,
        raw_scores={"interaction_scores": all_scores},
    )


if __name__ == "__main__":
    np.random.seed(0)
    N = 1000
    F = np.random.uniform(100, 1000, N)
    L = np.random.uniform(1, 5, N)
    E = np.random.uniform(50e9, 250e9, N)
    I = np.random.uniform(1e-7, 1e-5, N)
    true_deflection = (F * L**3) / (3 * E * I)

    best_cv = F * L**3
    pred = 8.78e-6 * best_cv  # rough linear fit, illustrative
    residuals = true_deflection - pred * (np.std(true_deflection) / np.std(pred))

    card = build_hypothesis_card(
        residuals=residuals,
        best_creator_var=best_cv,
        best_creator_var_units="[N] * [m]^3",
        y_units="[m]",
        observed_vars={'F': F, 'L': L, 'E': E},
    )
    print(card.display())
