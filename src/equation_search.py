"""
Stage 2-3: Symbolic Equation Search and Joint Evaluation
==========================================================
Given top Creator Variables, fit the simplest interpretable equation
g(phi_1, ..., phi_k) -> y, then evaluate accuracy vs complexity jointly.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from dataclasses import dataclass


@dataclass
class FittedModel:
    creator_var_name: str
    coef: float
    intercept: float
    r2: float
    complexity: int
    predictions: np.ndarray
    residuals: np.ndarray
    form: str = "linear"

    def score(self, lam=0.02):
        """Joint accuracy/complexity score. Higher is better."""
        return self.r2 - lam * self.complexity

    def __repr__(self):
        return (f"FittedModel(form={self.form}, y ~ {self.coef:.4g} * [{self.creator_var_name}] "
                f"+ {self.intercept:.4g}, R2={self.r2:.4f}, complexity={self.complexity})")


def fit_simplest_equation(creator_variables, y, lam=0.02, try_log_space=True):
    """
    Stage 2 + Stage 3 combined: for each candidate Creator Variable,
    fit a simple model to y, score on accuracy/complexity tradeoff,
    return the best model and the full Pareto set.

    If try_log_space is True, each candidate is fit BOTH as a raw linear
    model (y ~ a*phi + b) and, when both phi and y are strictly positive,
    as a log-log power-law model (log y ~ a*log(phi) + b, equivalent to
    y ~ exp(b) * phi^a).

    IMPORTANT: the two forms are compared using R^2 computed on the SCALE
    THEY WERE FIT ON (raw R^2 for linear, log-space R^2 for power-law),
    not by converting the power-law fit back to raw scale. This matters
    because when an omitted variable has high multiplicative range (e.g.
    I varying over 2 orders of magnitude in the beam example), raw-scale
    residual variance is dominated by that omitted variable regardless of
    how well the included variables are captured -- comparing on raw scale
    would unfairly penalize a power-law candidate that is, on the scale
    where the physics is actually linear (log-log), an excellent fit.
    A model that explains log y well IS the more useful representation
    even if its raw R^2 looks unimpressive; that's the whole point of
    recognizing power-law structure.
    """
    models = []
    y_positive = np.all(y > 0)
    log_y = np.log(y) if y_positive else None

    for cv in creator_variables:
        X = cv.values.reshape(-1, 1)

        reg_lin = LinearRegression().fit(X, y)
        preds_lin = reg_lin.predict(X)
        r2_lin = reg_lin.score(X, y)

        best_preds, best_r2, best_coef, best_intercept, form = (
            preds_lin, r2_lin, reg_lin.coef_[0], reg_lin.intercept_, "linear"
        )

        if try_log_space and y_positive and np.all(cv.values > 0):
            log_x = np.log(cv.values).reshape(-1, 1)
            reg_log = LinearRegression().fit(log_x, log_y)
            r2_log = reg_log.score(log_x, log_y)  # compared on log scale directly

            # Tiebreak rule: prefer log-space whenever it is not clearly
            # WORSE than linear (allow a small tolerance), rather than only
            # when it is strictly better. Rationale: a linear fit to a
            # genuinely multiplicative relationship (y = a*phi, no real
            # additive constant) can achieve similar or even marginally
            # higher raw R^2 than the log-space fit purely because raw R^2
            # rewards explaining the LARGEST-magnitude points well, while
            # leaving residuals that are strongly HETEROSCEDASTIC (their
            # variance scales with 1/phi or similar) -- which is a clear
            # structural defect that close R^2 alone won't reveal. We
            # detect this directly: if the linear residual's squared
            # magnitude correlates strongly with phi (heteroscedasticity),
            # and the log-space fit doesn't have this problem, prefer
            # log-space even at a slightly lower raw R^2.
            log_preds_raw = np.exp(reg_log.predict(log_x))
            lin_resid_for_hetero_check = y - preds_lin
            hetero_lin = abs(np.corrcoef(lin_resid_for_hetero_check**2, cv.values)[0, 1])

            log_resid_check = log_y - reg_log.predict(log_x)
            hetero_log = abs(np.corrcoef(log_resid_check**2, cv.values)[0, 1])

            prefer_log = (r2_log > best_r2) or (
                r2_log > best_r2 - 0.05 and hetero_log < hetero_lin - 0.1
            )

            if prefer_log:
                best_preds, best_r2 = log_preds_raw, r2_log
                best_coef, best_intercept, form = reg_log.coef_[0], reg_log.intercept_, "power_law"

        # Residuals are computed on the scale the model was actually scored
        # on: log-space for power_law (where the relationship is linear and
        # leftover structure is meaningful), raw scale for linear models.
        if form == "power_law":
            resid = log_y - (best_coef * np.log(cv.values) + best_intercept)
        else:
            resid = y - best_preds
        model = FittedModel(
            creator_var_name=cv.name,
            coef=best_coef,
            intercept=best_intercept,
            r2=best_r2,
            complexity=cv.complexity,
            predictions=best_preds,
            residuals=resid,
            form=form,
        )
        models.append(model)

    models.sort(key=lambda m: -m.score(lam))
    return models[0], models


if __name__ == "__main__":
    from cvm_core import CVM

    np.random.seed(0)
    N = 500
    F = np.random.uniform(100, 1000, N)
    L = np.random.uniform(1, 5, N)
    E = np.random.uniform(50e9, 250e9, N)
    I = np.random.uniform(1e-7, 1e-5, N)
    deflection = (F * L**3) / (3 * E * I)

    X = {'F': F, 'L': L, 'E': E}
    units = {'F': 'N', 'L': 'm', 'E': 'Pa'}

    cvm = CVM(variable_names=['F', 'L', 'E'], variable_units=units, max_power=3, max_terms=2)
    cvm.fit(X, deflection, top_k=10)

    best, pareto = fit_simplest_equation(cvm.candidates_, deflection)
    print("Best model (I withheld):")
    print(f"  {best}")
    print(f"  Residual std: {np.std(best.residuals):.4g}")
    print(f"  Residual / signal ratio: {np.std(best.residuals)/np.std(deflection):.2%}")
