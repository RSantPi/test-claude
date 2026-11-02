"""Compare learners with the ideal model on generated data.

GPR and KRR use scikit-learn, installed with `pip install smdata[eval]`. Any
object with fit(X, y) and predict(X) works as a learner.
"""

import numpy as np
from scipy.linalg import solve_triangular

from .complexity import solve_gp
from .dataset import make_dataset


class IdealGP:
    """The true model: the GP that knows the kernel, noise and mean."""

    def __init__(self, kernel, noise, mean=None):
        self.kernel, self.noise, self.mean = kernel, noise, mean

    def _mean(self, X):
        return np.zeros(X.shape[0]) if self.mean is None else np.asarray(self.mean(X))

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.factor, self.weights = solve_gp(self.kernel(self.X), y - self._mean(self.X), self.noise)
        return self

    def predict(self, X, return_std=False):
        K = self.kernel(X, self.X)
        prediction = self._mean(X) + K @ self.weights
        if not return_std:
            return prediction
        v = solve_triangular(self.factor[0], K.T, lower=True, check_finite=False)
        return prediction, np.sqrt(np.clip(1.0 - np.sum(v**2, axis=0), 0, None))


def make_gpr(n_restarts=4, seed=0):
    """GPR with an RBF kernel, tuned by marginal likelihood.

    Length scales are limited to what can matter on [-1, 1]^D. Wider limits let
    the optimiser settle on "the signal is all noise".
    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    class TunedGPR:
        def fit(self, X, y):
            kernel = (
                ConstantKernel(1.0, (1e-2, 1e2))
                * RBF(0.3 * np.ones(X.shape[1]), (0.02, 20.0))
                + WhiteKernel(0.01, (1e-8, 1.0))
            )
            self.model = GaussianProcessRegressor(
                kernel=kernel, normalize_y=True, n_restarts_optimizer=n_restarts, random_state=seed
            ).fit(X, y)
            return self

        def predict(self, X, return_std=False):
            if not return_std:
                return self.model.predict(X)
            mean, std = self.model.predict(X, return_std=True)
            # scikit-learn's std includes the learned noise. Remove it, so this
            # is uncertainty about the signal, as in IdealGP.
            noise = self.model.kernel_.k2.noise_level * self.model._y_train_std**2
            return mean, np.sqrt(np.clip(std**2 - noise, 0, None))

    return TunedGPR()


def make_krr(n_folds=5):
    """KRR with an RBF kernel, tuned by cross-validation."""
    from sklearn.kernel_ridge import KernelRidge
    from sklearn.model_selection import GridSearchCV

    grid = {
        "alpha": np.logspace(-6, 1, 8),
        "gamma": 1 / (2 * np.logspace(-1.5, 1, 11) ** 2),
    }

    class TunedKRR:
        def fit(self, X, y):
            self.offset = y.mean()
            self.model = GridSearchCV(
                KernelRidge(kernel="rbf"), grid, cv=min(n_folds, X.shape[0]),
                scoring="neg_mean_squared_error",
            ).fit(X, y - self.offset)
            return self

        def predict(self, X):
            return self.model.predict(X) + self.offset

    return TunedKRR()


def signal_error(prediction, data):
    """Mean squared error against the true signal. The signal variance is 1."""
    return float(np.mean((prediction - data.f_test) ** 2))


def coverage(mean, std, data, z=1.96):
    """Share of true noisy test outputs inside the 95% predictive interval."""
    width = z * np.sqrt(std**2 + data.noise**2)
    return float(np.mean(np.abs(data.y_test - mean) <= width))


def learning_curve(kernel, n_values, noise, learners, mean=None, n_repeats=5, n_test=500, seed=0):
    """Test each learner on fresh datasets at each training size.

    learners maps names to functions that return a new learner.
    Each row gives averages over repeats. Excess is a learner's error minus the
    ideal model's, on the same data, which measures its mismatch with the data.
    """
    seeds = np.random.SeedSequence(seed)
    rows = []
    for n in n_values:
        ideal_errors, complexities = [], []
        errors = {name: [] for name in learners}
        covered = {name: [] for name in learners}
        for child in seeds.spawn(n_repeats):
            data = make_dataset(
                kernel, n, n_test=n_test, noise=noise, mean=mean,
                seed=np.random.default_rng(child),
            )
            ideal = IdealGP(kernel, noise, mean).fit(data.X_train, data.y_train)
            ideal_errors.append(signal_error(ideal.predict(data.X_test), data))
            complexities.append(data.complexity)
            for name, factory in learners.items():
                model = factory().fit(data.X_train, data.y_train)
                try:
                    result, std = model.predict(data.X_test, return_std=True)
                    covered[name].append(coverage(result, std, data))
                except TypeError:
                    result = model.predict(data.X_test)
                errors[name].append(signal_error(result, data))
        ideal_error = float(np.mean(ideal_errors))
        row = {
            "n": n,
            "complexity": float(np.mean(complexities)),
            "ideal_error": ideal_error,
            "ideal_stderr": float(np.std(ideal_errors, ddof=1) / np.sqrt(n_repeats)) if n_repeats > 1 else float("nan"),
        }
        for name in learners:
            row[f"{name}_error"] = float(np.mean(errors[name]))
            row[f"{name}_excess"] = row[f"{name}_error"] - ideal_error
            if covered[name]:
                row[f"{name}_coverage"] = float(np.mean(covered[name]))
        rows.append(row)
    return rows
