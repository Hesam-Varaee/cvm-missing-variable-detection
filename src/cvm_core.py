"""
Creator Variable Machine (CVM) - Core Module
==============================================
Generates "Creator Variables" - explicit symbolic algebraic combinations
of observed input variables that maximize correlation with the response.

This is a white-box machine: every output is an explicit formula,
never a black-box transformation.
"""

import numpy as np
import itertools
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple


@dataclass
class CreatorVariable:
    """A single Creator Variable: a symbolic formula + its evaluated values."""
    name: str                  # symbolic representation, e.g. "x0 * x1^3"
    formula: Callable           # function(X_dict) -> array
    values: np.ndarray          # evaluated on the dataset
    correlation: float           # |corr| with target
    units: str = ""              # symbolic unit expression, e.g. "F*L^3"
    complexity: int = 1           # number of operations used

    def __repr__(self):
        return f"CreatorVariable('{self.name}', corr={self.correlation:.4f}, units=[{self.units}])"


@dataclass
class StabilityReport:
    """
    Result of bootstrap-resampling the top-k candidate ranking (Stage 1
    selection stability diagnostic).

    win_rates: {candidate_name: fraction of bootstrap resamples in which
        this candidate had the highest |score| among the top-k}
    top_candidate: the candidate selected on the full (un-resampled) data
    top_candidate_win_rate: its win_rate under resampling
    runner_up: name of the second-most-frequent winner
    runner_up_win_rate: its win rate
    margin: top_candidate_win_rate - runner_up_win_rate
    stable: True if the full-data top candidate also clearly dominates
        under resampling (win_rate exceeds `stable_threshold` AND margin
        over the runner-up exceeds `margin_threshold`)
    n_boot: number of bootstrap resamples used
    """
    win_rates: Dict[str, float]
    top_candidate: str
    top_candidate_win_rate: float
    runner_up: str
    runner_up_win_rate: float
    margin: float
    stable: bool
    n_boot: int

    def __repr__(self):
        status = "STABLE" if self.stable else "UNSTABLE (near-tie)"
        return (f"StabilityReport({status}: top='{self.top_candidate}' "
                f"won {self.top_candidate_win_rate:.1%} of {self.n_boot} resamples, "
                f"runner-up='{self.runner_up}' won {self.runner_up_win_rate:.1%}, "
                f"margin={self.margin:.1%})")

    def display(self):
        lines = [
            "=" * 60,
            "STAGE 1 SELECTION STABILITY REPORT",
            "=" * 60,
            f"  Status: {'STABLE selection' if self.stable else 'UNSTABLE selection (near-tie among top candidates)'}",
            f"  Full-data top candidate : {self.top_candidate}",
            f"  Its bootstrap win rate  : {self.top_candidate_win_rate:.1%}  (out of {self.n_boot} resamples)",
            f"  Runner-up               : {self.runner_up}  (win rate {self.runner_up_win_rate:.1%})",
            f"  Margin (top - runner-up): {self.margin:.1%}",
            "-" * 60,
            "  All candidate win rates:",
        ]
        for name, rate in sorted(self.win_rates.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"    {rate:6.1%}  {name}")
        lines.append("=" * 60)
        if not self.stable:
            lines.append(
                "  WARNING: the Stage 1 'best' Creator Variable is not a clear\n"
                "  winner -- several candidates are close enough in score that\n"
                "  resampling the data changes which one is selected. Any\n"
                "  downstream fit, direction/interaction inference, or\n"
                "  confirmatory test that is anchored to a single winning\n"
                "  candidate should be treated as conditional on this\n"
                "  (somewhat arbitrary) selection, not as evidence about that\n"
                "  specific formula being uniquely correct. Consider reporting\n"
                "  results across the top-N tied candidates rather than only\n"
                "  the nominal winner."
            )
        return "\n".join(lines)


class CVM:
    """
    Creator Variable Machine.

    Searches the space of algebraic combinations of input variables
    (products, ratios, powers) to find variables that best correlate
    with the response y. All outputs are explicit symbolic formulas.
    """

    def __init__(self, variable_names, variable_units=None, max_power=3, max_terms=2):
        """
        Parameters
        ----------
        variable_names : list of str
            Names of observed variables, e.g. ['F', 'L', 'E']
        variable_units : dict, optional
            Maps variable name -> unit string, e.g. {'F': 'N', 'L': 'm', 'E': 'Pa'}
        max_power : int
            Maximum exponent to try for any single variable
        max_terms : int
            Maximum number of distinct variables combined in one Creator Variable
        """
        self.variable_names = variable_names
        self.variable_units = variable_units or {v: v for v in variable_names}
        self.max_power = max_power
        self.max_terms = max_terms
        self.candidates_: List[CreatorVariable] = []

    def _powers_grid(self):
        """Generate all power combinations for max_terms variables."""
        powers = list(range(-self.max_power, self.max_power + 1))
        powers = [p for p in powers if p != 0]  # exclude power 0 (= constant)
        return powers

    def _build_formula(self, var_subset, exponents):
        """Build a callable formula: product of var^exp for each var in subset."""
        def formula(X):
            result = np.ones(len(next(iter(X.values()))))
            for v, e in zip(var_subset, exponents):
                result = result * np.power(X[v].astype(float), e)
            return result
        return formula

    def _build_name_and_units(self, var_subset, exponents):
        """Build symbolic name and unit string."""
        parts = []
        unit_parts = []
        for v, e in zip(var_subset, exponents):
            if e == 1:
                parts.append(f"{v}")
                unit_parts.append(f"[{self.variable_units[v]}]")
            else:
                parts.append(f"{v}^{e}")
                unit_parts.append(f"[{self.variable_units[v]}]^{e}")
        name = " * ".join(parts)
        units = " * ".join(unit_parts)
        return name, units

    def fit(self, X: Dict[str, np.ndarray], y: np.ndarray, top_k=10, min_abs_corr=0.05,
            use_log_space=True):
        """
        Search algebraic combinations and rank by |correlation| with y.

        Parameters
        ----------
        X : dict of {var_name: array}
        y : array, the response
        top_k : int
            Number of top candidates to retain
        min_abs_corr : float
            Discard candidates below this correlation threshold
        use_log_space : bool
            If True, ALSO score each candidate by correlation in log-space
            (log|phi| vs log|y|). Many physical laws are power laws (e.g.
            F*L^3/E/I), where the multiplicative structure means log-space
            correlation is far more sensitive than raw Pearson correlation,
            especially when one factor has high relative variance. We take
            the max of the two correlation scores so power-law and additive
            relationships are both detected.
        """
        self.candidates_ = []
        powers = self._powers_grid()

        y_loggable = np.all(y > 0)
        log_y = np.log(np.abs(y) + 1e-300) if use_log_space else None

        for n_terms in range(1, self.max_terms + 1):
            for var_subset in itertools.combinations(self.variable_names, n_terms):
                for exponents in itertools.product(powers, repeat=n_terms):
                    formula = self._build_formula(var_subset, exponents)
                    try:
                        values = formula(X)
                    except Exception:
                        continue
                    if not np.all(np.isfinite(values)) or np.std(values) < 1e-12:
                        continue

                    corr = np.corrcoef(values, y)[0, 1]
                    if not np.isfinite(corr):
                        corr = 0.0

                    best_corr = corr
                    if use_log_space and np.all(values > 0):
                        log_vals = np.log(values)
                        log_corr = np.corrcoef(log_vals, log_y)[0, 1]
                        if np.isfinite(log_corr) and abs(log_corr) > abs(best_corr):
                            best_corr = log_corr

                    if abs(best_corr) < min_abs_corr:
                        continue
                    name, units = self._build_name_and_units(var_subset, exponents)
                    cv = CreatorVariable(
                        name=name,
                        formula=formula,
                        values=values,
                        correlation=best_corr,
                        units=units,
                        complexity=n_terms + sum(abs(e) for e in exponents),
                    )
                    self.candidates_.append(cv)

        # Sort by |correlation| descending. Tie-break secondary key prefers
        # POSITIVE correlation over negative -- this matters specifically
        # for reciprocal pairs (e.g. F*L^3*E^-1 vs F^-1*L^-3*E), which have
        # IDENTICAL |correlation| but opposite sign. Without a deterministic
        # tie-break, which one survives dedup depends on itertools iteration
        # order, not the data -- and the two forms give OPPOSITE answers for
        # direction and unit-power sign downstream despite being the same
        # physical statement. Always keeping the positively-correlated form
        # makes the choice deterministic and the result reproducible.
        self.candidates_.sort(key=lambda c: (-abs(c.correlation), -c.correlation))
        deduped = []
        seen_vals = []
        for c in self.candidates_:
            is_dup = False
            for sv in seen_vals:
                abs_corr_with_seen = abs(np.corrcoef(c.values, sv)[0, 1])
                if abs_corr_with_seen > 0.9999:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(c)
                seen_vals.append(c.values)
        self.candidates_ = deduped[:top_k]
        return self

    def top(self, n=1):
        return self.candidates_[:n]

    def assess_stability(self, y: np.ndarray, n_boot: int = 500, seed: int = 0,
                          stable_threshold: float = 0.5, margin_threshold: float = 0.15,
                          use_log_space: bool = True) -> StabilityReport:
        """
        Stage 1 selection stability diagnostic.

        WHY THIS EXISTS: fit() picks a single 'best' Creator Variable by
        |correlation| on the full dataset and every downstream stage (2-3
        equation fitting, 4 residual diagnostics, and any confound-controlled
        confirmatory test built on top of it) treats that choice as fixed.
        This is fine when the winner is a clear winner. But empirically
        (WWR real-data experiment), several candidates can be near-tied in
        score -- e.g. corr=0.908 vs 0.905 vs 0.898 vs 0.898 -- close enough
        that which one wins flips depending on which ~80% of rows you
        happen to have. Because downstream coefficients (e.g. a
        confound-controlled effect for a withheld variable) are PARTIAL
        w.r.t. whichever base candidate was selected, a flipped selection
        can flip the sign of a downstream coefficient even when the
        underlying effect being tested is itself robust -- this was
        observed directly and traced to exactly this cause, not to
        instability of the effect itself. This method makes that failure
        mode visible and diagnosable BEFORE it contaminates downstream
        inference, rather than leaving the analyst to discover it by
        chance via an unrelated robustness check.

        Method: case-resampling bootstrap. Re-scores the ALREADY-FOUND
        top-k candidates (self.candidates_) -- not a full re-search of the
        combinatorial space, which would be prohibitively expensive -- on
        n_boot resamples (with replacement) of the rows, using the same
        scoring rule as fit() (max of raw and log-space |correlation|).
        Tracks which candidate has the top score in each resample.

        Must be called AFTER fit(), since it reuses self.candidates_.

        Parameters
        ----------
        y : array
            The same response array passed to fit().
        n_boot : int
            Number of bootstrap resamples.
        stable_threshold : float
            Minimum win rate (fraction of resamples) the full-data top
            candidate must achieve to be called 'stable'.
        margin_threshold : float
            Minimum win-rate gap over the runner-up required to be called
            'stable' -- guards against e.g. a 3-way near-tie where each
            candidate wins ~33% (none individually below stable_threshold's
            complement, but none is a real winner either).
        """
        if not self.candidates_:
            raise ValueError("assess_stability() must be called after fit()")

        n = len(y)
        y_loggable = np.all(y > 0)
        rng = np.random.default_rng(seed)

        cand_names = [c.name for c in self.candidates_]
        cand_values = [c.values for c in self.candidates_]
        win_counts = {name: 0 for name in cand_names}

        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            y_b = y[idx]
            log_y_b = np.log(np.abs(y_b) + 1e-300) if use_log_space else None

            best_idx = None
            best_score = -1.0
            for i, vals in enumerate(cand_values):
                v_b = vals[idx]
                if np.std(v_b) < 1e-12:
                    continue
                corr = np.corrcoef(v_b, y_b)[0, 1]
                score = abs(corr) if np.isfinite(corr) else 0.0
                if use_log_space and np.all(v_b > 0):
                    log_corr = np.corrcoef(np.log(v_b), log_y_b)[0, 1]
                    if np.isfinite(log_corr):
                        score = max(score, abs(log_corr))
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx is not None:
                win_counts[cand_names[best_idx]] += 1

        win_rates = {name: count / n_boot for name, count in win_counts.items()}
        top_candidate = cand_names[0]  # winner on the full, un-resampled data
        top_rate = win_rates[top_candidate]

        ranked = sorted(win_rates.items(), key=lambda kv: -kv[1])
        runner_up, runner_up_rate = next(
            ((name, rate) for name, rate in ranked if name != top_candidate),
            (None, 0.0)
        )
        margin = top_rate - runner_up_rate
        stable = (top_rate >= stable_threshold) and (margin >= margin_threshold)

        return StabilityReport(
            win_rates=win_rates,
            top_candidate=top_candidate,
            top_candidate_win_rate=top_rate,
            runner_up=runner_up,
            runner_up_win_rate=runner_up_rate,
            margin=margin,
            stable=stable,
            n_boot=n_boot,
        )


if __name__ == "__main__":
    # Quick smoke test
    np.random.seed(0)
    N = 500
    F = np.random.uniform(100, 1000, N)
    L = np.random.uniform(1, 5, N)
    E = np.random.uniform(50e9, 250e9, N)
    I = np.random.uniform(1e-7, 1e-5, N)

    deflection = (F * L**3) / (3 * E * I)

    X = {'F': F, 'L': L, 'E': E}  # I withheld
    units = {'F': 'N', 'L': 'm', 'E': 'Pa'}

    cvm = CVM(variable_names=['F', 'L', 'E'], variable_units=units, max_power=3, max_terms=2)
    cvm.fit(X, deflection, top_k=5)

    print("Top Creator Variables (I withheld):")
    for c in cvm.top(5):
        print(f"  {c}")

    print()
    stability = cvm.assess_stability(deflection, n_boot=300, seed=0)
    print(stability.display())
