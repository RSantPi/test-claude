"""Mean functions. Each one maps an array X of shape (n, D) to shape (n,)."""

import numpy as np


def constant(value):
    return lambda X: np.full(np.asarray(X).shape[0], float(value))


def ramp(slopes):
    slopes = np.asarray(slopes, dtype=float)
    return lambda X: np.asarray(X) @ slopes


def bowl(height):
    return lambda X: height * np.sum(np.asarray(X) ** 2, axis=1)


def bump(height, width, center=0.0):
    def mean(X):
        distance2 = np.sum((np.asarray(X) - center) ** 2, axis=1)
        return height * np.exp(-distance2 / (2 * width**2))

    return mean


def step(height, direction):
    direction = np.asarray(direction, dtype=float)
    return lambda X: height * np.where(np.asarray(X) @ direction >= 0, 1.0, -1.0)
