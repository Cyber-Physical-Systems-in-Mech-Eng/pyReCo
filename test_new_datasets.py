"""
Quick test of new datasets.py without reservoirpy dependency.
"""
import sys
sys.path.insert(0, 'src')

from pyreco import datasets
import numpy as np

print("="*80)
print("TESTING NEW DATASETS.PY (NO RESERVOIRPY)")
print("="*80)

# Test Lorenz
print("\n1. Testing Lorenz dataset:")
try:
    x_train, y_train, x_test, y_test = datasets.load(
        'lorenz',
        n_samples=1000,
        train_fraction=0.7,
        n_in=100,
        n_out=1,
        seed=None
    )
    print(f"   ✓ Lorenz loaded successfully")
    print(f"   x_train shape: {x_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   x_test shape:  {x_test.shape}")
    print(f"   y_test shape:  {y_test.shape}")
    print(f"   Sample value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

    # Check for NaN/Inf
    assert not np.isnan(x_train).any(), "Found NaN in x_train"
    assert not np.isinf(x_train).any(), "Found Inf in x_train"
    print(f"   ✓ No NaN/Inf values")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test Mackey-Glass
print("\n2. Testing Mackey-Glass dataset:")
try:
    x_train, y_train, x_test, y_test = datasets.load(
        'mackey_glass',
        n_samples=1000,
        train_fraction=0.7,
        n_in=100,
        n_out=1,
        seed=42
    )
    print(f"   ✓ Mackey-Glass loaded successfully")
    print(f"   x_train shape: {x_train.shape}")
    print(f"   y_train shape: {y_train.shape}")
    print(f"   x_test shape:  {x_test.shape}")
    print(f"   y_test shape:  {y_test.shape}")
    print(f"   Sample value range: [{x_train.min():.2f}, {x_train.max():.2f}]")

    # Check for NaN/Inf
    assert not np.isnan(x_train).any(), "Found NaN in x_train"
    assert not np.isinf(x_train).any(), "Found Inf in x_train"
    print(f"   ✓ No NaN/Inf values")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test reproducibility with seed
print("\n3. Testing reproducibility (Mackey-Glass with seed=42):")
try:
    x1, y1, _, _ = datasets.load('mackey_glass', n_samples=500, seed=42)
    x2, y2, _, _ = datasets.load('mackey_glass', n_samples=500, seed=42)

    matches = np.allclose(x1, x2, rtol=1e-10, atol=1e-10)
    print(f"   Reproducibility test: {'✓ PASS' if matches else '✗ FAIL'}")
    if not matches:
        diff = np.abs(x1 - x2)
        print(f"   Max difference: {diff.max()}, Mean: {diff.mean()}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Test expected shapes
print("\n4. Testing shape calculations:")
n_samples = 5000
train_frac = 0.7
n_in = 100
n_out = 1

n_train_timesteps = int(n_samples * train_frac)  # 3500
n_test_timesteps = n_samples - n_train_timesteps  # 1500

expected_train_samples = n_train_timesteps - n_in - n_out + 1  # 3400
expected_test_samples = n_test_timesteps - n_in - n_out + 1    # 1400

x_train, y_train, x_test, y_test = datasets.load(
    'lorenz',
    n_samples=n_samples,
    train_fraction=train_frac,
    n_in=n_in,
    n_out=n_out
)

actual_train = x_train.shape[0]
actual_test = x_test.shape[0]

print(f"   Expected train samples: {expected_train_samples}, Actual: {actual_train}")
print(f"   Expected test samples:  {expected_test_samples}, Actual: {actual_test}")
print(f"   Shape calculation: {'✓ CORRECT' if actual_train == expected_train_samples and actual_test == expected_test_samples else '✗ WRONG'}")

print("\n" + "="*80)
print("ALL TESTS COMPLETED")
print("="*80)
