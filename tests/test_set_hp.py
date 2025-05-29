import unittest
import numpy as np

# Mock or import your actual classes
from pyreco.custom_models import RC
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer

class TestSetHp(unittest.TestCase):
    def setUp(self):
        # Minimal RC model setup
        self.model = RC()
        self.model.input_layer = InputLayer(input_shape=(10, 1))
        self.model.reservoir_layer = RandomReservoirLayer(nodes=10, density=0.1, spec_rad=0.9)
        self.model.readout_layer = ReadoutLayer(output_shape=(10, 1))

    def test_set_valid_hp(self):
        # Setting a simple HP
        self.model.reservoir_layer.set_hp('leakage_rate', 0.25)
        self.assertEqual(self.model.reservoir_layer.leakage_rate, 0.25)

    def test_set_special_hp_spec_rad(self):
        # Setting spec_rad triggers rescale logic
        old_weights = self.model.reservoir_layer.weights.copy()
        self.model.reservoir_layer.set_hp('spec_rad', 0.8)
        self.assertAlmostEqual(self.model.reservoir_layer.spec_rad, 0.8)
        self.assertFalse(np.allclose(old_weights, self.model.reservoir_layer.weights))  # Should be different

    def test_set_structural_hp_triggers_reinit(self):
        # nodes/density should reinitialize (or raise, depending on your logic)
        try:
            self.model.reservoir_layer.set_hp('nodes', 20)
            self.assertEqual(self.model.reservoir_layer.nodes, 20)
            # Optionally check the new weight shape, etc.
            self.assertEqual(self.model.reservoir_layer.weights.shape[0], 20)
        except RuntimeError:
            # If your logic raises instead of reinit, that's fine too
            pass

    def test_set_invalid_hp_raises(self):
        # Setting an unknown HP should raise ValueError
        with self.assertRaises(ValueError):
            self.model.reservoir_layer.set_hp('not_a_real_hp', 123)

    def test_layer_without_hps(self):
        # Example: input_layer has no HPs to set (unless you define some)
        with self.assertRaises(ValueError):
            self.model.input_layer.set_hp('foo', 123)

if __name__ == "__main__":
    unittest.main()
