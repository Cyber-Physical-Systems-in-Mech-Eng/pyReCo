import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pyreco.custom_models import RC  # or wherever your RC model is


def test_model_visualize_invalid_file_type():
    """
    Test that model_visualize() raises RuntimeError for unsupported file types.
    This is where your actual validation logic is.
    """
    model = RC()

    # Mock the required attributes that model_visualize needs
    model.reservoir_layer = MagicMock()
    model.input_layer = MagicMock()
    model.readout_layer = MagicMock()
    # Setup dummy weights
    n_reservoir = 10
    model.reservoir_layer.weights = np.random.randn(n_reservoir, n_reservoir) * 0.3
    model.input_layer.weights = np.random.randn(1, n_reservoir)
    model.readout_layer.weights = np.random.randn(n_reservoir, 1)
    model.input_layer.n_states = 1
    model.readout_layer.n_states = 1

    # Test 1: Invalid file type 'bmp' should raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        model.model_visualize(save=True, file_type="bmp")

    # Check the exact error message from your code
    expected_error = "Error: Unsupported file type. Use png, jpg, jpeg, svg, or pdf."
    assert expected_error in str(exc_info.value)

    # Test 2: Another invalid file type
    with pytest.raises(RuntimeError) as exc_info:
        model.model_visualize(save=True, file_type="tiff")

    assert "Unsupported file type" in str(exc_info.value)

    # Test 3: Valid file types should NOT raise error
    valid_types = ["png", "jpg", "jpeg", "pdf", "svg", "PNG", "PDF"]

    for file_type in valid_types:
        with patch('matplotlib.pyplot.show'), \
             patch('matplotlib.pyplot.savefig'):
            try:
                model.model_visualize(save=True, file_type=file_type)
                # If we get here, no error was raised (good!)
            except RuntimeError:
                pytest.fail
                (f"Valid file type '{file_type}' should not raise RuntimeError")


def test_model_visualize_edge_cases():
    """
    Test edge cases for file type validation in model_visualize.
    """
    model = RC()

    # Test: When save=False, file_type validation shouldn't happen
    # (invalid file type shouldn't raise error if not saving)
    try:
        model.model_visualize(save=False, file_type="bmp")
        # Should not raise error
    except RuntimeError:
        pytest.fail("Should not validate file type when save=False")

    # Test: Whitespace handling
    with pytest.raises(RuntimeError):
        model.model_visualize(save=True, file_type=" bmp ")

    # Test: Empty string file type
    with pytest.raises(RuntimeError):
        model.model_visualize(save=True, file_type="")

    # Test: None file type should use default "jpeg"
    with patch('matplotlib.pyplot.show'), \
         patch('matplotlib.pyplot.savefig'):
        model.model_visualize(save=True, file_type=None)
