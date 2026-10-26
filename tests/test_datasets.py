"""
Tests for the chaotic benchmark systems in pyreco.datasets (Lorenz, Mackey-Glass).
"""

import numpy as np
import pytest

from pyreco import datasets as ds


# ------------------------------------------------------------------- Lorenz


def test_lorenz_satisfies_its_equations():
    h = 0.01
    x = ds._generate_lorenz(2000, h=h)
    assert x.shape == (2000, 3)
    # central finite differences of the trajectory against the right-hand side
    dxdt = (x[2:] - x[:-2]) / (2 * h)
    state = x[1:-1]
    rhs = np.stack(
        [
            10.0 * (state[:, 1] - state[:, 0]),
            state[:, 0] * (28.0 - state[:, 2]) - state[:, 1],
            state[:, 0] * state[:, 1] - 8.0 / 3.0 * state[:, 2],
        ],
        axis=1,
    )
    assert np.abs(dxdt - rhs).max() < 0.05 * np.abs(rhs).max()


def test_lorenz_lives_on_the_attractor():
    x = ds._generate_lorenz(5000)
    settled = x[500:]  # after the start-up transient
    assert np.isfinite(x).all()
    assert np.abs(settled[:, 0]).max() < 25 and np.abs(settled[:, 1]).max() < 30
    assert 0 < settled[:, 2].min() and settled[:, 2].max() < 55
    assert settled[:, 0].std() > 5  # it switches between the two lobes


def test_lorenz_starts_at_the_initial_condition_and_is_deterministic():
    a = ds._generate_lorenz(200, x0=[2.0, 1.0, 0.5])
    assert np.allclose(a[0], [2.0, 1.0, 0.5])
    assert np.array_equal(a, ds._generate_lorenz(200, x0=[2.0, 1.0, 0.5]))


# ------------------------------------------------------------ Mackey-Glass


def test_mackey_glass_range_and_shape():
    x = ds._generate_mackey_glass(2000)
    assert x.shape == (2000, 1) and np.isfinite(x).all()
    assert 0.3 < x[300:].min() and x[300:].max() < 1.5


def test_mackey_glass_is_converged_in_the_integration_step():
    coarse = ds._generate_mackey_glass(500, h_int=0.1)
    fine = ds._generate_mackey_glass(500, h_int=0.02)
    assert np.abs(coarse - fine).max() < 5e-3


def test_mackey_glass_satisfies_the_delay_equation():
    # sampled with h=1 and an exact delay of 17 samples:
    # x'(t) = a x(t-tau)/(1+x(t-tau)^n) - b x(t)
    x = ds._generate_mackey_glass(800).ravel()
    t = np.arange(40, 780)
    dxdt = (x[t + 1] - x[t - 1]) / 2.0
    rhs = 0.2 * x[t - 17] / (1 + x[t - 17] ** 10) - 0.1 * x[t]
    assert np.abs(dxdt - rhs).max() < 0.02


def test_mackey_glass_seed():
    same = [ds._generate_mackey_glass(300, seed=1) for _ in range(2)]
    assert np.array_equal(same[0], same[1])
    assert not np.array_equal(same[0], ds._generate_mackey_glass(300, seed=2))
    assert np.array_equal(ds._generate_mackey_glass(300), ds._generate_mackey_glass(300))


@pytest.mark.parametrize("kwargs", [dict(h_int=0.3), dict(h=0.25, h_int=0.1), dict(tau=17.55)])
def test_mackey_glass_requires_commensurable_steps(kwargs):
    with pytest.raises(ValueError, match="integer"):
        ds._generate_mackey_glass(100, **kwargs)


# --------------------------------------------------------------------- load


@pytest.mark.parametrize("name, n_features", [("lorenz", 3), ("mackey_glass", 1)])
def test_load_windows_are_aligned_and_chronological(name, n_features):
    n_in, n_out, n = 20, 3, 1000
    x_train, y_train, x_test, y_test = ds.load(name, n_samples=n, n_in=n_in, n_out=n_out)
    raw = (ds._generate_lorenz if name == "lorenz" else ds._generate_mackey_glass)(n)
    n_train = int(n * 0.7)

    assert x_train.shape == (n_train - n_in - n_out + 1, n_in, n_features)
    assert y_train.shape == (x_train.shape[0], n_out, n_features)
    assert x_test.shape == (n - n_train - n_in - n_out + 1, n_in, n_features)
    for i in (0, 7, x_train.shape[0] - 1):
        assert np.allclose(x_train[i], raw[i : i + n_in])
        assert np.allclose(y_train[i], raw[i + n_in : i + n_in + n_out])
    # the test windows come from the part of the series after the training segment
    assert np.allclose(x_test[0], raw[n_train : n_train + n_in])
    assert np.allclose(y_test[-1], raw[-n_out:])


def test_load_return_arity():
    assert len(ds.load("lorenz", n_samples=800, n_in=10)) == 4
    assert len(ds.load("lorenz", n_samples=800, n_in=10, standardize=True)) == 5
    assert len(ds.load("lorenz", n_samples=800, n_in=10, val_fraction=0.2)) == 7


def test_standardization_uses_training_statistics_only():
    n, n_in = 1000, 10
    raw = ds._generate_lorenz(n)
    n_train = int(n * 0.7)
    *_, scaler = ds.load("lorenz", n_samples=n, n_in=n_in, standardize=True)
    assert np.allclose(scaler.mean_, raw[:n_train].mean(axis=0))

    # with a validation split the scaler sees only the final training segment
    x_train, y_train, x_val, y_val, x_test, y_test, scaler = ds.load(
        "lorenz", n_samples=n, n_in=n_in, val_fraction=0.2
    )
    n_final = int(n_train * 0.8)
    assert np.allclose(scaler.mean_, raw[:n_final].mean(axis=0))
    # validation windows are the segment between the final training data and the test data
    assert np.allclose(x_val[0], scaler.transform(raw[n_final : n_final + n_in]))
    assert np.allclose(x_test[0], scaler.transform(raw[n_train : n_train + n_in]))


def test_dataset_names_are_case_and_separator_insensitive():
    ref = ds.load("lorenz", n_samples=600, n_in=10)[0]
    for name in ["lorentz69", "Lorenz63", "LORENZ"]:
        assert np.array_equal(ds.load(name, n_samples=600, n_in=10)[0], ref)
    ref = ds.load("mackey_glass", n_samples=600, n_in=10)[0]
    for name in ["mg", "MackeyGlass", "Mackey-Glass", "mackey glass"]:
        assert np.array_equal(ds.load(name, n_samples=600, n_in=10)[0], ref)


def test_seed_only_affects_mackey_glass():
    a = ds.load("lorenz", n_samples=600, n_in=10, seed=1)[0]
    b = ds.load("lorenz", n_samples=600, n_in=10, seed=2)[0]
    assert np.array_equal(a, b)
    c = ds.load("mg", n_samples=600, n_in=10, seed=1)[0]
    assert np.array_equal(c, ds.load("mg", n_samples=600, n_in=10, seed=1)[0])
    assert not np.array_equal(c, ds.load("mg", n_samples=600, n_in=10, seed=2)[0])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(dataset_name="nope"), "Unknown dataset"),
        (dict(dataset_name="lorenz", train_fraction=1.0), "train_fraction"),
        (dict(dataset_name="lorenz", train_fraction=0.0), "train_fraction"),
        (dict(dataset_name="lorenz", n_samples=50, n_in=100), "timesteps"),
        (dict(dataset_name="lorenz", val_fraction=1.5), "val_fraction"),
    ],
)
def test_load_validates_its_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ds.load(**{**dict(n_samples=1000, n_in=20), **kwargs})


def test_load_does_not_print(capsys):
    ds.load("lorenz", n_samples=500, n_in=10)
    assert capsys.readouterr().out == ""
