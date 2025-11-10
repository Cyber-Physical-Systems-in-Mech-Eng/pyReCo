"""
Test that datasets.load() correctly implements sliding window logic
by comparing against manual application of sliding window.
"""
import sys
# Add local PyReCo repository to path (with new implementation)
sys.path.insert(0, '/Users/hengz/pyReCo/src')

from pyreco import datasets
from pyreco.datasets import _generate_lorenz, _generate_mackey_glass, _sliding_window
import numpy as np

print("=" * 80)
print("TESTING SLIDING WINDOW LOGIC IN DATASETS.LOAD()")
print("=" * 80)

# Test parameters
n_samples = 1000
train_frac = 0.7
n_in = 100
n_out = 1
seed = 42

print("\n1. Testing Lorenz sliding window logic:")
print("-" * 80)

# Generate raw data using the same function datasets.load() uses
raw_lorenz = _generate_lorenz(n_timesteps=n_samples, sigma=10.0, rho=28.0, beta=8.0/3.0, h=0.01, x0=[1.0, 1.0, 1.0])
print(f"Raw data shape: {raw_lorenz.shape}")

# Manually split (same as datasets.load())
n_train = int(n_samples * train_frac)
train_data = raw_lorenz[:n_train]
test_data = raw_lorenz[n_train:]
print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# Manually apply sliding window
x_train_manual, y_train_manual = _sliding_window(train_data, n_in=n_in, n_out=n_out)
x_test_manual, y_test_manual = _sliding_window(test_data, n_in=n_in, n_out=n_out)
print(f"Manual x_train shape: {x_train_manual.shape}")
print(f"Manual y_train shape: {y_train_manual.shape}")
print(f"Manual x_test shape: {x_test_manual.shape}")
print(f"Manual y_test shape: {y_test_manual.shape}")

# Use datasets.load() (Mode 1: no standardization)
x_train_auto, y_train_auto, x_test_auto, y_test_auto = datasets.load(
    'lorenz',
    n_samples=n_samples,
    train_fraction=train_frac,
    n_in=n_in,
    n_out=n_out,
    seed=seed  # Not used for Lorenz
)
print(f"Auto x_train shape: {x_train_auto.shape}")
print(f"Auto y_train shape: {y_train_auto.shape}")
print(f"Auto x_test shape: {x_test_auto.shape}")
print(f"Auto y_test shape: {y_test_auto.shape}")

# Compare
lorenz_match = (
    np.allclose(x_train_manual, x_train_auto, rtol=1e-10, atol=1e-10) and
    np.allclose(y_train_manual, y_train_auto, rtol=1e-10, atol=1e-10) and
    np.allclose(x_test_manual, x_test_auto, rtol=1e-10, atol=1e-10) and
    np.allclose(y_test_manual, y_test_auto, rtol=1e-10, atol=1e-10)
)

print(f"\n✓ Shapes match: {x_train_manual.shape == x_train_auto.shape}")
print(f"✓ Data match: {lorenz_match}")

if not lorenz_match:
    print(f"Max diff x_train: {np.abs(x_train_manual - x_train_auto).max()}")
    print(f"Max diff y_train: {np.abs(y_train_manual - y_train_auto).max()}")

print("\n2. Testing Mackey-Glass sliding window logic:")
print("-" * 80)

# Generate raw data
raw_mg = _generate_mackey_glass(n_timesteps=n_samples, tau=17, a=0.2, b=0.1, n=10, x0=1.2, h=1.0, seed=seed)
print(f"Raw data shape: {raw_mg.shape}")

# Manually split
train_data_mg = raw_mg[:n_train]
test_data_mg = raw_mg[n_train:]

# Manually apply sliding window
x_train_manual_mg, y_train_manual_mg = _sliding_window(train_data_mg, n_in=n_in, n_out=n_out)
x_test_manual_mg, y_test_manual_mg = _sliding_window(test_data_mg, n_in=n_in, n_out=n_out)
print(f"Manual x_train shape: {x_train_manual_mg.shape}")
print(f"Manual y_train shape: {y_train_manual_mg.shape}")

# Use datasets.load()
x_train_auto_mg, y_train_auto_mg, x_test_auto_mg, y_test_auto_mg = datasets.load(
    'mackey_glass',
    n_samples=n_samples,
    train_fraction=train_frac,
    n_in=n_in,
    n_out=n_out,
    seed=seed
)
print(f"Auto x_train shape: {x_train_auto_mg.shape}")
print(f"Auto y_train shape: {y_train_auto_mg.shape}")

# Compare
mg_match = (
    np.allclose(x_train_manual_mg, x_train_auto_mg, rtol=1e-10, atol=1e-10) and
    np.allclose(y_train_manual_mg, y_train_auto_mg, rtol=1e-10, atol=1e-10) and
    np.allclose(x_test_manual_mg, x_test_auto_mg, rtol=1e-10, atol=1e-10) and
    np.allclose(y_test_manual_mg, y_test_auto_mg, rtol=1e-10, atol=1e-10)
)

print(f"\n✓ Shapes match: {x_train_manual_mg.shape == x_train_auto_mg.shape}")
print(f"✓ Data match: {mg_match}")

if not mg_match:
    print(f"Max diff x_train: {np.abs(x_train_manual_mg - x_train_auto_mg).max()}")
    print(f"Max diff y_train: {np.abs(y_train_manual_mg - y_train_auto_mg).max()}")

print("\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)
print(f"Lorenz: {'✅ PASS' if lorenz_match else '❌ FAIL'}")
print(f"Mackey-Glass: {'✅ PASS' if mg_match else '❌ FAIL'}")

if lorenz_match and mg_match:
    print("\n🎉 SUCCESS! Sliding window logic is correct!")
    exit(0)
else:
    print("\n❌ FAILURE! Sliding window logic has issues.")
    exit(1)
