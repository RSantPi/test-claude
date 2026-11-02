"""The spectral mixture kernel on the domain [-1, 1]^D."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectralMixture:
    """A stationary kernel whose spectrum is a mixture of Gaussians.

    k(t) = sum_q w_q cos(2 pi mu_q . t) prod_p exp(-2 pi^2 t_p^2 sigma_qp^2)

    Weights are rescaled to sum to 1, so k(0) = 1 and every signal has unit
    variance. Noise levels therefore mean the same thing for every kernel.

    weights: shape (Q,). Positive.
    means:   shape (Q, D). Peak frequencies, in cycles per unit of input.
    sigmas:  shape (Q, D). Peak widths, the square roots of the variances.
    """

    weights: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray

    def __post_init__(self):
        weights = np.asarray(self.weights, dtype=float)
        means = np.atleast_2d(np.asarray(self.means, dtype=float))
        sigmas = np.atleast_2d(np.asarray(self.sigmas, dtype=float))
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if means.shape != sigmas.shape or means.shape[0] != weights.size:
            raise ValueError("means and sigmas must both have shape (Q, D)")
        if np.any(weights <= 0) or np.any(sigmas <= 0):
            raise ValueError("weights and sigmas must be positive")
        object.__setattr__(self, "weights", weights / weights.sum())
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "sigmas", sigmas)

    @property
    def dim(self):
        return self.means.shape[1]

    @property
    def n_components(self):
        return self.weights.size

    @classmethod
    def random(
        cls,
        dim,
        n_components=3,
        max_frequency=2.0,
        sigma_range=(0.1, 1.0),
        seed=None,
    ):
        """Draw a kernel: random directions, frequencies, widths and weights."""
        rng = np.random.default_rng(seed)
        directions = rng.normal(size=(n_components, dim))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        means = directions * rng.uniform(0, max_frequency, size=(n_components, 1))
        low, high = np.log(sigma_range)
        sigmas = np.exp(rng.uniform(low, high, size=(n_components, dim)))
        return cls(rng.dirichlet(np.ones(n_components)), means, sigmas)

    def __call__(self, X, Y=None):
        """The covariance matrix between rows of X and rows of Y."""
        X = np.asarray(X, dtype=float)
        Y = X if Y is None else np.asarray(Y, dtype=float)
        K = np.zeros((X.shape[0], Y.shape[0]))
        for weight, mean, sigma in zip(self.weights, self.means, self.sigmas):
            phase = np.zeros_like(K)
            exponent = np.zeros_like(K)
            for p in range(self.dim):
                delta = X[:, p, None] - Y[None, :, p]
                phase += mean[p] * delta
                exponent += sigma[p] ** 2 * delta**2
            K += weight * np.cos(2 * np.pi * phase) * np.exp(-2 * np.pi**2 * exponent)
        return K

    def sample_frequencies(self, n, rng):
        """Draw n frequencies from the spectrum."""
        which = rng.choice(self.n_components, size=n, p=self.weights)
        signs = rng.choice([-1.0, 1.0], size=(n, 1))
        return signs * self.means[which] + rng.normal(size=(n, self.dim)) * self.sigmas[which]
