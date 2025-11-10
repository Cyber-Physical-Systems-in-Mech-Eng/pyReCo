"""
Test script for all three modes of datasets.load()
Tests: Mode 1 (no standardize), Mode 2 (standardize only), Mode 3 (val_fraction)
"""
import sys
sys.path.insert(0, 'src')

from pyreco import datasets
import numpy as np

print("=" * 80)
print("TESTING ALL THREE MODES OF datasets.load()")
print("=" * 80)

# Mode 1: No standardization, no validation (backward compatible)
print("\n1. Mode 1: standardize=False, val_fraction=None (default)")
try:
    result = datasets.load('lorenz', n_samples=1000, train_fraction=0.7)
    
    if len(result) == 4:
        x_train, y_train, x_test, y_test = result
        print(f"   ✓ Returns 4 items (backward compatible)")
        print(f"   x_train shape: {x_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   x_test shape:  {x_test.shape}")
        print(f"   y_test shape:  {y_test.shape}")
        print(f"   Data range: [{x_train.min():.2f}, {x_train.max():.2f}] (not standardized)")
    else:
        print(f"   ✗ FAILED: Expected 4 items, got {len(result)}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

# Mode 2: With standardization, no validation
print("\n2. Mode 2: standardize=True, val_fraction=None")
try:
    result = datasets.load('lorenz', n_samples=1000, train_fraction=0.7, standardize=True)
    
    if len(result) == 5:
        x_train, y_train, x_test, y_test, scaler = result
        print(f"   ✓ Returns 5 items (with scaler)")
        print(f"   x_train shape: {x_train.shape}")
        print(f"   y_train shape: {y_train.shape}")
        print(f"   x_test shape:  {x_test.shape}")
        print(f"   y_test shape:  {y_test.shape}")
        print(f"   scaler type:   {type(scaler).__name__}")
        print(f"   x_train mean:  {x_train.mean():.6f} (should be ~0)")
        print(f"   x_train std:   {x_train.std():.6f} (should be ~1)")
    else:
        print(f"   ✗ FAILED: Expected 5 items, got {len(result)}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Mode 3: With validation set (automatically enables standardization)
print("\n3. Mode 3: val_fraction=0.15 (standardize auto-enabled)")
try:
    result = datasets.load('lorenz', n_samples=1000, train_fraction=0.7, val_fraction=0.15)
    
    if len(result) == 7:
        x_train, y_train, x_val, y_val, x_test, y_test, scaler = result
        print(f"   ✓ Returns 7 items (train/val/test + scaler)")
        print(f"   x_train shape: {x_train.shape}")
        print(f"   x_val shape:   {x_val.shape}")
        print(f"   x_test shape:  {x_test.shape}")
        print(f"   scaler type:   {type(scaler).__name__}")
        print(f"   x_train mean:  {x_train.mean():.6f} (should be ~0)")
        print(f"   x_train std:   {x_train.std():.6f} (should be ~1)")
    else:
        print(f"   ✗ FAILED: Expected 7 items, got {len(result)}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test with Mackey-Glass
print("\n4. Testing Mode 2 with Mackey-Glass")
try:
    x_tr, y_tr, x_te, y_te, scaler = datasets.load(
        'mackey_glass', n_samples=1000, train_fraction=0.7, 
        standardize=True, seed=42
    )
    print(f"   ✓ Mackey-Glass with standardization")
    print(f"   x_train shape: {x_tr.shape}")
    print(f"   Mean: {x_tr.mean():.6f}, Std: {x_tr.std():.6f}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")

print("\n" + "=" * 80)
print("ALL MODE TESTS COMPLETED")
print("=" * 80)
