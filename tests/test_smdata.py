import numpy as np
import pytest

from smdata import (
    SpectralMixture,
    bump,
    complexity,
    effective_dof,
    effective_rank,
    make_dataset,
    matrix_rank,
    ramp,
    step,
)
from smdata.evaluate import IdealGP, coverage, learning_curve, signal_error


def midpoint_grid(dim, m):
    axis = -1 + (np.arange(m) + 0.5) * 2 / m
    return np.stack([g.ravel() for g in np.meshgrid(*[axis] * dim, indexing="ij")], axis=1)


def kernel_1d(means, sigmas, weights=None):
    means, sigmas = np.array(means, float)[:, None], np.array(sigmas, float)[:, None]
    return SpectralMixture(np.ones(len(means)) if weights is None else weights, means, sigmas)


# Kernel


def test_weights_are_normalised_so_signal_variance_is_one():
    k = SpectralMixture([2, 6], [[0.0], [1.0]], [[0.2], [0.3]])
    assert k.weights.sum() == pytest.approx(1)
    assert k(np.zeros((1, 1)))[0, 0] == pytest.approx(1)


def test_kernel_matches_formula_in_1d():
    k = kernel_1d([1.5], [0.3])
    t = 0.4
    expected = np.exp(-2 * np.pi**2 * t**2 * 0.09) * np.cos(2 * np.pi * 1.5 * t)
    assert k(np.array([[t]]), np.array([[0.0]]))[0, 0] == pytest.approx(expected)


def test_kernel_matrix_is_symmetric_and_positive():
    k = SpectralMixture.random(3, seed=1)
    X = np.random.default_rng(0).uniform(-1, 1, (80, 3))
    K = k(X)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-9


def test_rejects_bad_parameters():
    with pytest.raises(ValueError):
        SpectralMixture([1.0], [[0.0]], [[-1.0]])
    with pytest.raises(ValueError):
        SpectralMixture([1.0], [[0.0, 0.0]], [[1.0]])


# Effective rank


@pytest.mark.parametrize(
    "means, sigmas, weights",
    [
        ([0], [0.05], None),
        ([0], [0.3], None),
        ([0], [2.0], None),
        ([3], [0.02], None),
        ([2, 6], [0.2, 0.5], [0.7, 0.3]),
    ],
)
def test_rank_matches_matrix_in_1d(means, sigmas, weights):
    k = kernel_1d(means, sigmas, weights)
    assert effective_rank(k) == pytest.approx(matrix_rank(k(midpoint_grid(1, 1500))), rel=1e-3)


@pytest.mark.parametrize(
    "means, sigmas",
    [
        ([[0, 0]], [[0.3, 0.1]]),
        ([[1.5, -0.7], [0.3, 2.0]], [[0.3, 0.2], [0.15, 0.5]]),
        ([[1, 1]], [[0.05, 0.05]]),
    ],
)
def test_rank_matches_matrix_in_2d(means, sigmas):
    k = SpectralMixture(np.ones(len(means)), means, sigmas)
    assert effective_rank(k) == pytest.approx(matrix_rank(k(midpoint_grid(2, 64))), rel=2e-3)


def test_rank_limits():
    assert effective_rank(kernel_1d([0], [1e-6])) == pytest.approx(1)
    assert effective_rank(kernel_1d([4], [1e-6])) == pytest.approx(2, rel=1e-2)
    assert effective_rank(kernel_1d([40], [0.01])) == pytest.approx(2.0053, rel=1e-3)


def test_rank_is_stable_for_narrow_distant_peaks():
    rank = effective_rank(kernel_1d([200, 350], [0.001, 0.002]))
    assert np.isfinite(rank) and 2 < rank < 5


def test_rank_approaches_the_infinite_domain_formula():
    k = kernel_1d([0], [8.0])
    assert effective_rank(k) == pytest.approx(4 * np.sqrt(np.pi) * 8.0, rel=0.02)


def test_inactive_axis_adds_no_complexity():
    one = SpectralMixture([1.0], [[0.0]], [[0.8]])
    two = SpectralMixture([1.0], [[0.0, 0.0]], [[0.8, 1e-6]])
    assert effective_rank(two) == pytest.approx(effective_rank(one), rel=1e-6)


def test_rank_of_independent_axes_multiplies():
    x = SpectralMixture([1.0], [[0.0]], [[0.5]])
    y = SpectralMixture([1.0], [[0.0]], [[0.9]])
    both = SpectralMixture([1.0], [[0.0, 0.0]], [[0.5, 0.9]])
    assert effective_rank(both) == pytest.approx(effective_rank(x) * effective_rank(y))


def test_finite_sample_rank_relation():
    k = SpectralMixture.random(2, seed=4)
    X = np.random.default_rng(0).uniform(-1, 1, (1200, 2))
    expected = 1 / (1 / effective_rank(k) + 1 / 1200)
    assert matrix_rank(k(X)) == pytest.approx(expected, rel=0.03)


# Complexity


def test_complexity_is_between_zero_and_one_and_moves_the_right_way():
    k = SpectralMixture.random(2, seed=2)
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (600, 2))
    few = complexity(k, X[:30], 0.1, seed=1)
    many = complexity(k, X, 0.1, seed=1)
    noisy = complexity(k, X, 1.0, seed=1)
    assert 0 <= many < few < 1
    assert many < noisy < 1


def test_complexity_near_one_with_overwhelming_noise():
    k = SpectralMixture.random(2, seed=2)
    X = np.random.default_rng(0).uniform(-1, 1, (50, 2))
    assert complexity(k, X, 100.0, seed=0) > 0.99


def test_complexity_is_the_error_of_the_ideal_model():
    k = SpectralMixture.random(2, n_components=2, seed=3)
    errors, values = [], []
    for seed in range(60):
        data = make_dataset(k, 120, n_test=200, noise=0.2, seed=seed)
        model = IdealGP(k, 0.2).fit(data.X_train, data.y_train)
        errors.append(signal_error(model.predict(data.X_test), data))
        values.append(data.complexity)
    assert np.mean(errors) == pytest.approx(np.mean(values), rel=0.08)


def test_effective_dof_limits():
    K = SpectralMixture.random(1, seed=0)(np.linspace(-1, 1, 50)[:, None])
    assert effective_dof(K, 1e-5) > 10
    assert effective_dof(K, 1e4) < 0.01


# Generator


def test_same_seed_gives_same_dataset():
    k = SpectralMixture.random(2, seed=0)
    a, b = make_dataset(k, 50, seed=9), make_dataset(k, 50, seed=9)
    assert np.array_equal(a.y_train, b.y_train)
    assert make_dataset(k, 50, seed=10).y_train[0] != a.y_train[0]


def test_inputs_lie_in_the_domain_and_shapes_match():
    k = SpectralMixture.random(3, seed=0)
    d = make_dataset(k, 40, n_test=25, seed=1)
    assert d.X_train.shape == (40, 3) and d.X_test.shape == (25, 3)
    assert np.abs(np.vstack([d.X_train, d.X_test])).max() <= 1


def test_noise_has_the_requested_size():
    k = SpectralMixture.random(1, seed=0)
    d = make_dataset(k, 1500, noise=0.4, seed=2)
    assert np.std(d.y_train - d.f_train) == pytest.approx(0.4, rel=0.08)


@pytest.mark.parametrize("method", ["exact", "features"])
def test_samples_have_the_kernel_covariance(method):
    k = SpectralMixture([0.6, 0.4], [[0.8], [0.0]], [[0.2], [0.4]])
    X = np.array([[-0.5], [0.0], [0.3], [0.9]])
    from smdata.dataset import _draw_exact, _draw_features

    rng = np.random.default_rng(1)
    if method == "exact":
        draws = np.array([_draw_exact(k, X, rng) for _ in range(6000)])
    else:
        draws = np.array([_draw_features(k, X, rng, 2048) for _ in range(6000)])
    assert np.abs(draws.T @ draws / 6000 - k(X)).max() < 0.07


def test_mean_shifts_the_signal():
    k = SpectralMixture.random(1, seed=0)
    mean = ramp([2.0])
    plain = make_dataset(k, 100, seed=3)
    shifted = make_dataset(k, 100, mean=mean, seed=3)
    assert np.allclose(shifted.f_train - plain.f_train, mean(plain.X_train))


def test_mean_shapes():
    X = np.array([[0.5, 0.0], [-0.5, 0.0]])
    assert step(2.0, [1, 0])(X).tolist() == [2.0, -2.0]
    assert bump(1.0, 0.5)(np.zeros((1, 2)))[0] == pytest.approx(1)


@pytest.mark.parametrize("method", ["exact", "features"])
def test_signal_passes_through_observed_points(method):
    # A smooth kernel with many inputs, so some land almost on each point.
    k = SpectralMixture([1.0], [[0.0]], [[0.2]])
    X_obs = np.array([[-0.4], [0.5]])
    y_obs = np.array([2.5, -2.5])
    d = make_dataset(k, 3000, n_test=5, noise=0.0, observed=(X_obs, y_obs), method=method, seed=0)
    for x, y in zip(X_obs, y_obs):
        nearest = np.argmin(np.abs(d.X_train[:, 0] - x[0]))
        assert d.f_train[nearest] == pytest.approx(y, abs=0.02)


def test_conditioning_really_changes_the_signal():
    k = SpectralMixture([1.0], [[0.0]], [[0.2]])
    plain = make_dataset(k, 200, seed=1)
    pinned = make_dataset(k, 200, observed=(np.array([[0.0]]), np.array([3.0])), seed=1)
    assert np.array_equal(plain.X_train, pinned.X_train)
    assert np.abs(plain.f_train - pinned.f_train).max() > 0.5


def test_observed_points_lower_complexity():
    k = SpectralMixture.random(2, seed=5)
    X_obs = np.random.default_rng(1).uniform(-1, 1, (40, 2))
    y_obs = np.zeros(40)
    plain = make_dataset(k, 30, seed=0)
    pinned = make_dataset(k, 30, observed=(X_obs, y_obs), seed=0)
    assert pinned.complexity < plain.complexity


def test_large_dataset_without_measuring_complexity():
    k = SpectralMixture.random(4, seed=0)
    d = make_dataset(k, 50000, n_test=10, method="features", measure_complexity=False, seed=0)
    assert d.complexity is None and d.rank > 0
    assert np.var(d.f_train) == pytest.approx(1, abs=0.35)


def test_rejects_points_outside_the_domain():
    k = SpectralMixture.random(1, seed=0)
    with pytest.raises(ValueError):
        make_dataset(k, 5, observed=(np.array([[1.5]]), np.array([0.0])))


# Evaluator


def test_ideal_gp_intervals_are_calibrated():
    k = SpectralMixture.random(2, seed=6)
    d = make_dataset(k, 200, n_test=2000, noise=0.2, seed=0)
    mean, std = IdealGP(k, 0.2).fit(d.X_train, d.y_train).predict(d.X_test, return_std=True)
    assert coverage(mean, std, d) == pytest.approx(0.95, abs=0.03)


def test_tuned_gpr_recovers_an_rbf_signal_and_is_calibrated():
    pytest.importorskip("sklearn")
    from smdata.evaluate import make_gpr

    k = SpectralMixture([1.0], [[0.0, 0.0, 0.0]], [[0.5, 0.5, 0.5]])
    d = make_dataset(k, 500, n_test=800, noise=0.1, seed=0)
    mean, std = make_gpr(n_restarts=1).fit(d.X_train, d.y_train).predict(d.X_test, return_std=True)
    ideal = IdealGP(k, 0.1).fit(d.X_train, d.y_train).predict(d.X_test)
    assert signal_error(mean, d) < 1.5 * signal_error(ideal, d)
    assert coverage(mean, std, d) == pytest.approx(0.95, abs=0.03)


def test_learning_curve_with_sklearn():
    pytest.importorskip("sklearn")
    from smdata.evaluate import make_gpr, make_krr

    k = SpectralMixture([1.0], [[0.0]], [[0.5]])
    rows = learning_curve(
        k, [20, 80], 0.1, {"GPR": lambda: make_gpr(n_restarts=0), "KRR": make_krr},
        n_repeats=2, n_test=100,
    )
    assert [r["n"] for r in rows] == [20, 80]
    assert rows[1]["ideal_error"] < rows[0]["ideal_error"]
    assert rows[1]["GPR_excess"] > -0.05 and "GPR_coverage" in rows[1]
    assert "KRR_coverage" not in rows[1]
