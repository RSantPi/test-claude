"""Complexity measures.

There are two levels, described in docs/complexity.md:

effective_rank: how much a kernel can express on [-1, 1]^D.
complexity:     how much of the signal an ideal model can't recover from a
                given dataset. It depends on the kernel, the inputs and noise.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import dawsn, wofz

_SMALL_A = 1e-7
_MIN_NOISE_VARIANCE = 1e-10


def _axis_integral(delta, u):
    """Double integral of cos(2 pi delta t) exp(-2 pi^2 u t^2) over [-1, 1]^2.

    Here t = x - x'. It uses the Faddeeva function rather than a raw erf of a
    complex argument, which overflows for narrow peaks far from zero.
    """
    delta, u = np.broadcast_arrays(np.abs(delta), u)
    a = 2 * np.pi**2 * u
    b = 2 * np.pi * delta
    out = np.empty(a.shape)

    tiny = a < _SMALL_A
    with np.errstate(divide="ignore", invalid="ignore"):
        wave = 2 * (1 - np.cos(2 * b)) / b**2
    out[tiny] = np.where(b < 1e-8, 4.0, wave)[tiny]

    a, b = a[~tiny], b[~tiny]
    root = np.sqrt(a)
    z = b / (2 * root)
    edge = np.exp(-(z**2)) - np.exp(-4 * a + 2j * b) * wofz(z + 2j * root)
    Z = np.sqrt(np.pi) / (2 * root) * edge + 1j * dawsn(z) / root
    out[~tiny] = 4 * Z.real - (1 - np.exp(-4 * a) * np.cos(2 * b) - b * Z.imag) / a
    return out


def effective_rank(kernel):
    """The exact effective rank of a kernel on [-1, 1]^D with uniform inputs.

    r = tr(T)^2 / tr(T^2), where T is the kernel's integral operator. It runs
    from 1 for a constant function upwards, and a pure wave gives 2.
    """
    v = kernel.sigmas**2
    u = v[:, None, :] + v[None, :, :]
    minus = kernel.means[:, None, :] - kernel.means[None, :, :]
    plus = kernel.means[:, None, :] + kernel.means[None, :, :]
    # Dividing each axis by its area, 4, keeps the product stable at high D.
    terms = np.prod(_axis_integral(minus, u) / 4, axis=-1)
    terms = terms + np.prod(_axis_integral(plus, u) / 4, axis=-1)
    pairs = np.outer(kernel.weights, kernel.weights) / 2
    return 1.0 / np.sum(pairs * terms)


def matrix_rank(K):
    """Effective rank of a covariance matrix.

    With n random inputs, 1 / r_n is close to 1 / r + 1 / n.
    """
    K = np.asarray(K)
    return np.trace(K) ** 2 / np.sum(K**2)


def effective_dof(K, noise):
    """Degrees of freedom: the sum of mu / (mu + noise^2) over eigenvalues mu."""
    variance = max(noise**2, _MIN_NOISE_VARIANCE)
    mu = np.clip(np.linalg.eigvalsh(K), 0, None)
    return float(np.sum(mu / (mu + variance)))


def complexity(kernel, X, noise, n_eval=1000, seed=None, X_observed=None, observed_noise=0.0):
    """The share of the signal an ideal model can't recover.

    This is the posterior variance of the true GP, averaged over new uniform
    inputs in [-1, 1]^D. It runs from 0, fully determined, to 1, nothing
    learned. It depends on where the inputs are, but not on the outputs.

    X_observed adds fixed points that the signal was conditioned on.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=float)
    variances = np.full(X.shape[0], max(noise**2, _MIN_NOISE_VARIANCE))
    if X_observed is not None and len(X_observed):
        X = np.vstack([np.asarray(X_observed, dtype=float), X])
        extra = max(observed_noise**2, _MIN_NOISE_VARIANCE)
        variances = np.concatenate([np.full(len(X_observed), extra), variances])
    A = kernel(X)
    A[np.diag_indices_from(A)] += variances
    L = cho_factor(A, lower=True)[0]
    X_eval = rng.uniform(-1, 1, size=(n_eval, kernel.dim))
    v = solve_triangular(L, kernel(X, X_eval), lower=True, check_finite=False)
    return float(np.clip(np.mean(1.0 - np.sum(v**2, axis=0)), 0.0, 1.0))


def solve_gp(K, y, noise):
    """Cholesky factor of K + noise^2 I and the weights for y."""
    A = np.array(K, dtype=float)
    A[np.diag_indices_from(A)] += max(noise**2, _MIN_NOISE_VARIANCE)
    factor = cho_factor(A, lower=True)
    return factor, cho_solve(factor, y)
