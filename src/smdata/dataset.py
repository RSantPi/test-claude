"""Draw datasets from a Gaussian process with a spectral mixture kernel."""

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .complexity import complexity, effective_rank, solve_gp
from .kernel import SpectralMixture

_JITTER = 1e-8


@dataclass(frozen=True)
class Dataset:
    """Training data, test data and the settings that produced them.

    f_train and f_test hold the true signal, without noise.
    rank belongs to the kernel, and complexity to this particular dataset.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    f_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    f_test: np.ndarray
    kernel: SpectralMixture
    noise: float
    mean: Optional[Callable]
    rank: float
    complexity: Optional[float]


def _draw_exact(kernel, X, rng):
    K = kernel(X)
    K[np.diag_indices_from(K)] += _JITTER
    return np.linalg.cholesky(K) @ rng.standard_normal(X.shape[0])


def _draw_features(kernel, X, rng, n_features):
    frequencies = kernel.sample_frequencies(n_features, rng)
    phases = rng.uniform(0, 2 * np.pi, size=n_features)
    amplitude = np.sqrt(2.0 / n_features)
    out = np.zeros(X.shape[0])
    for start in range(0, X.shape[0], 2048):
        chunk = X[start : start + 2048]
        out[start : start + 2048] = np.sum(
            np.cos(2 * np.pi * chunk @ frequencies.T + phases), axis=1
        )
    return amplitude * out


def make_dataset(
    kernel,
    n_train,
    n_test=500,
    noise=0.1,
    mean=None,
    observed=None,
    observed_noise=0.0,
    method="exact",
    n_features=4096,
    n_eval=1000,
    measure_complexity=True,
    seed=None,
):
    """Draw one dataset, with inputs uniform on [-1, 1]^D.

    kernel:         a SpectralMixture.
    noise:          standard deviation of the noise. The signal has variance 1.
    mean:           optional mean function, such as smdata.ramp([1, 0]).
    observed:       optional (X, y) pair. The signal is forced through these
                    points, with noise observed_noise.
    method:         "exact" samples the true Gaussian process, at cubic cost.
                    "features" uses random waves, which is approximate but
                    scales to many more points.
    measure_complexity: computing complexity is cubic in n_train, whichever
                    method is used. Set this to False for large datasets, and
                    Dataset.complexity will be None.
    """
    if method not in ("exact", "features"):
        raise ValueError('method must be "exact" or "features"')
    rng = np.random.default_rng(seed)
    dim = kernel.dim
    X_train = rng.uniform(-1, 1, size=(n_train, dim))
    X_test = rng.uniform(-1, 1, size=(n_test, dim))

    if observed is None:
        X_obs, y_obs = np.empty((0, dim)), np.empty(0)
    else:
        X_obs = np.atleast_2d(np.asarray(observed[0], dtype=float))
        y_obs = np.asarray(observed[1], dtype=float)
        if X_obs.shape != (y_obs.size, dim):
            raise ValueError("observed must be (X, y) with X of shape (m, D)")
    X_all = np.vstack([X_obs, X_train, X_test])
    if np.any(np.abs(X_all) > 1):
        raise ValueError("inputs must lie inside [-1, 1]^D")

    if method == "exact":
        prior = _draw_exact(kernel, X_all, rng)
    else:
        prior = _draw_features(kernel, X_all, rng, n_features)
    baseline = np.zeros(X_all.shape[0]) if mean is None else np.asarray(mean(X_all), dtype=float)
    signal = prior + baseline

    m = X_obs.shape[0]
    if m:
        # Matheron's rule: correct the prior draw so it passes through the points.
        target = y_obs - signal[:m] - observed_noise * rng.standard_normal(m)
        factor, weights = solve_gp(kernel(X_obs), target, observed_noise)
        signal = signal + kernel(X_all, X_obs) @ weights

    f_train = signal[m : m + n_train]
    f_test = signal[m + n_train :]
    return Dataset(
        X_train=X_train,
        y_train=f_train + noise * rng.standard_normal(n_train),
        f_train=f_train,
        X_test=X_test,
        y_test=f_test + noise * rng.standard_normal(n_test),
        f_test=f_test,
        kernel=kernel,
        noise=noise,
        mean=mean,
        rank=effective_rank(kernel),
        complexity=complexity(
            kernel, X_train, noise, n_eval=n_eval, seed=rng,
            X_observed=X_obs, observed_noise=observed_noise,
        ) if measure_complexity else None,
    )
