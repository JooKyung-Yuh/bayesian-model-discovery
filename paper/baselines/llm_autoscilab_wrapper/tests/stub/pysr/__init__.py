"""STUB ONLY - a stand-in for pysr.PySRRegressor so the released EquationLearner runs on the development host.

PySR/Julia exist only in the bai venv up_asl. This fake implements the four calls the released
learner makes (constructor kwargs, fit(X_df, y), predict(X_df), sympy()) with an ordinary least
squares fit y ~ c0 + sum_i c_i x_i over the DataFrame's columns. It is never on sys.path in a real
run (the runner puts tests/stub first only with --stub, and preflight refuses a run where `pysr`
resolves to this file).
"""
import numpy as np

__version__ = "0.0-stub"
STUB = True


class PySRRegressor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._cols = None
        self._coef = None

    def fit(self, X, y, **_):
        self._cols = [str(c) for c in X.columns]
        A = np.column_stack([np.ones(len(X))] + [np.asarray(X[c], dtype=float) for c in X.columns])
        self._coef, *_ = np.linalg.lstsq(A, np.asarray(y, dtype=float), rcond=None)
        return self

    def predict(self, X):
        A = np.column_stack([np.ones(len(X))] + [np.asarray(X[c], dtype=float) for c in self._cols])
        return A @ self._coef

    def sympy(self):
        import sympy as sp
        expr = sp.Float(float(self._coef[0]))
        for c, k in zip(self._cols, self._coef[1:]):
            expr = expr + sp.Float(float(k)) * sp.Symbol(c)
        return expr
