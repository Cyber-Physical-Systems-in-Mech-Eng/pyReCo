# reservoir_validator.py
import warnings
import numpy as np
from typing import Any, Optional, Union, List, Tuple

from pyreco.squence_registory import sequence_registry


class SequenceParameterValidator:
    """Validator for sequence_to_sequence function parameters"""

    @staticmethod
    def validate_name(name):
        """Validate sequence name parameter"""
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")

        if name not in sequence_registry:
            raise ValueError(
                f"Invalid sequence name '{name}'. "
                f"Available: {list(sequence_registry.keys())}"
            )



    @staticmethod
    def validate_n_batch(n_batch):
        """Validate n_batch parameter"""
        if not isinstance(n_batch, int):
            raise TypeError(f"n_batch must be an integer, got {type(n_batch).__name__}")

        if n_batch < 1:
            raise ValueError(f"n_batch must be ≥ 1, got {n_batch}")

        if n_batch < 2:
            warnings.warn(
                f"Small n_batch ({n_batch}). May not have enough samples for train/test split."
            )
        elif n_batch > 10000:
            warnings.warn(f"Large n_batch ({n_batch}). May use significant memory.")

    @staticmethod
    def validate_n_states(n_states):
        """Validate n_states parameter"""
        if not isinstance(n_states, int):
            raise TypeError(
                f"n_states must be an integer, got {type(n_states).__name__}"
            )

        if n_states < 1:
            raise ValueError(f"n_states must be ≥ 1, got {n_states}")

        if n_states == 1:
            warnings.warn("n_states=1 means univariate sequences.")
        elif n_states > 100:
            warnings.warn(
                f"Large n_states ({n_states}). May create high-dimensional data."
            )

    @staticmethod
    def validate_n_time(n_time):
        """Validate n_time parameter"""
        if not isinstance(n_time, int):
            raise TypeError(f"n_time must be an integer, got {type(n_time).__name__}")

        if n_time < 1:
            raise ValueError(f"n_time must be ≥ 1, got {n_time}")

        if n_time == 1:
            warnings.warn("n_time=1 means single time step sequences.")
        elif n_time > 1000:
            warnings.warn(f"Large n_time ({n_time}). May create long sequences.")

    @staticmethod
    def validate_n_time_in(n_time_in):
        """Validate n_time_in parameter"""
        if n_time_in is not None:
            if not isinstance(n_time_in, int):
                raise TypeError(
                    f"n_time_in must be an integer, got {type(n_time_in).__name__}"
                )

            if n_time_in < 1:
                raise ValueError(f"n_time_in must be ≥ 1, got {n_time_in}")

    @staticmethod
    def validate_n_time_out(n_time_out):
        """Validate n_time_out parameter"""
        if n_time_out is not None:
            if not isinstance(n_time_out, int):
                raise TypeError(
                    f"n_time_out must be an integer, got {type(n_time_out).__name__}"
                )

            if n_time_out < 1:
                raise ValueError(f"n_time_out must be ≥ 1, got {n_time_out}")

    @staticmethod
    def validate_noise_level(noise_level):
        """Validate noise_level parameter"""
        if noise_level is not None:
            if not isinstance(noise_level, (int, float)):
                raise TypeError(
                    f"noise_level must be numeric, got {type(noise_level).__name__}"
                )

            if noise_level < 0:
                raise ValueError(f"noise_level must be ≥ 0, got {noise_level}")

            if noise_level > 1.0:
                warnings.warn(f"High noise_level ({noise_level}). May obscure signal.")

    @staticmethod
    def validate_frequencies(frequencies):
        """Validate frequencies parameter"""
        if frequencies is not None:
            if isinstance(frequencies, (int, float)):
                if frequencies <= 0:
                    raise ValueError(f"Frequency must be > 0, got {frequencies}")
            elif isinstance(frequencies, (list, tuple)):
                for i, freq in enumerate(frequencies):
                    if not isinstance(freq, (int, float)):
                        raise TypeError(
                            f"All frequencies must be numeric, got {type(freq).__name__} at index {i}"
                        )
                    if freq <= 0:
                        raise ValueError(
                            f"Frequency must be > 0, got {freq} at index {i}"
                        )
            else:
                raise TypeError(
                    f"frequencies must be numeric or list/tuple, got {type(frequencies).__name__}"
                )

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all sequence generation parameters"""
        # Required parameter
        if "name" not in kwargs:
            raise ValueError("name is a required parameter")

        cls.validate_name(kwargs["name"])

        # Validate main parameters
        cls.validate_n_batch(kwargs.get("n_batch", 50))
        cls.validate_n_states(kwargs.get("n_states", 2))
        cls.validate_n_time(kwargs.get("n_time", 3))

        # Validate optional parameters if provided
        if "n_time_in" in kwargs:
            cls.validate_n_time_in(kwargs["n_time_in"])

        if "n_time_out" in kwargs:
            cls.validate_n_time_out(kwargs["n_time_out"])

        if "noise_level" in kwargs:
            cls.validate_noise_level(kwargs["noise_level"])

        if "frequencies" in kwargs:
            cls.validate_frequencies(kwargs["frequencies"])

        # Check for parameter consistency
        name = kwargs.get("name", "")
        n_batch = kwargs.get("n_batch", 50)

        if n_batch < 2 and name:
            warnings.warn(
                f"n_batch={n_batch} may be too small for proper train/test split."
            )


class ReservoirParameterValidator:
    """Validator class for Reservoir Computer parameters"""

    @staticmethod
    def validate_num_nodes(num_nodes):
        if not isinstance(num_nodes, int) or num_nodes <= 0:
            raise ValueError(f"num_nodes must be a positive integer, got {num_nodes}")

    @staticmethod
    def validate_density(density, num_nodes):
        if not isinstance(density, (int, float)) or density <= 0 or density > 1:
            raise ValueError(
                f"density must be a float between 0 and 1 (exclusive), got {density}"
            )
        if density < 0.01:
            warnings.warn(
                f"Very low density ({density}). Reservoir may have isolated nodes."
            )
        if density < 1.0 / num_nodes:
            warnings.warn(
                f"Warning: density ({density}) is very low for {num_nodes} nodes. Some nodes may have no connections."
            )

    @staticmethod
    def validate_activation(activation):
        valid_activations = [
            "tanh",
            "sigmoid",
            "relu",
            "linear",
            "softmax",
            "elu",
            "selu",
        ]
        if activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got '{activation}'"
            )

    @staticmethod
    def validate_leakage_rate(leakage_rate):
        if (
            not isinstance(leakage_rate, (int, float))
            or leakage_rate < 0
            or leakage_rate > 1
        ):
            raise ValueError(
                f"leakage_rate must be a float between 0 and 1 (inclusive), got {leakage_rate}"
            )
        if leakage_rate == 0:
            warnings.warn("leakage_rate=0 means no state memory (purely feedforward)")
        elif leakage_rate == 1:
            warnings.warn("leakage_rate=1 means full memory (no forgetting)")
        elif leakage_rate < 0.01:
            warnings.warn(
                f"Very low leakage_rate ({leakage_rate}). Reservoir states may change very slowly."
            )

    @staticmethod
    def validate_spec_rad(spec_rad):
        if not isinstance(spec_rad, (int, float)) or spec_rad <= 0:
            raise ValueError(f"spec_rad must be a positive float, got {spec_rad}")
        if spec_rad < 0.1:
            warnings.warn(
                "Very low spectral radius. Reservoir may not have enough dynamics."
            )
        elif spec_rad > 1.5:
            warnings.warn("Spectral radius > 1. Reservoir may be at edge of stability.")

    @staticmethod
    def validate_fraction(fraction, name):
        if not isinstance(fraction, (int, float)) or fraction <= 0 or fraction > 1:
            raise ValueError(
                f"{name} must be a float between 0 and 1 (inclusive), got {fraction}"
            )

    @staticmethod
    def validate_positive_int(value, name, allow_none=False):
        if allow_none and value is None:
            return
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value}")

    @staticmethod
    def validate_init_sampling(init_res_sampling):
        valid_sampling_methods = [
            "random_normal",
            "uniform",
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "he_uniform",
        ]
        if init_res_sampling not in valid_sampling_methods:
            raise ValueError(
                f"init_res_sampling must be one of {valid_sampling_methods}, got '{init_res_sampling}'"
            )

    @staticmethod
    def validate_metrics(metrics):
        if metrics is not None:
            if isinstance(metrics, str):
                pass  # Accept any string, framework will validate
            elif isinstance(metrics, list):
                for metric in metrics:
                    if not isinstance(metric, str):
                        raise ValueError(
                            f"All metrics must be strings, got {type(metric)}"
                        )
            else:
                raise ValueError(f"metrics must be str or list, got {type(metrics)}")

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all parameters at once"""
        cls.validate_num_nodes(kwargs.get("num_nodes", 100))
        cls.validate_density(kwargs.get("density", 0.8), kwargs.get("num_nodes", 100))
        cls.validate_activation(kwargs.get("activation", "tanh"))
        cls.validate_leakage_rate(kwargs.get("leakage_rate", 0.5))
        cls.validate_spec_rad(kwargs.get("spec_rad", 0.9))
        cls.validate_fraction(kwargs.get("fraction_input", 1.0), "fraction_input")
        cls.validate_fraction(kwargs.get("fraction_output", 1.0), "fraction_output")
        cls.validate_positive_int(kwargs.get("n_time_in"), "n_time_in", allow_none=True)
        cls.validate_positive_int(
            kwargs.get("n_time_out"), "n_time_out", allow_none=True
        )
        cls.validate_positive_int(
            kwargs.get("n_states_in"), "n_states_in", allow_none=True
        )
        cls.validate_positive_int(
            kwargs.get("n_states_out"), "n_states_out", allow_none=True
        )
        cls.validate_init_sampling(kwargs.get("init_res_sampling", "random_normal"))
        cls.validate_metrics(kwargs.get("metrics", "mean_squared_error"))


class InputParameterValidator:
    """Validator class for Input Layer parameters"""

    @staticmethod
    def validate_input_shape(input_shape):
        """Validate the input_shape parameter"""
        if not isinstance(input_shape, tuple):
            raise TypeError(
                f"input_shape must be a tuple, got {type(input_shape).__name__}"
            )

        if len(input_shape) != 2:
            raise ValueError(
                f"input_shape must have 2 dimensions (n_timesteps, n_states), got {len(input_shape)} dimensions"
            )

        n_time, n_states = input_shape

        # Validate n_time
        if not isinstance(n_time, int) or n_time <= 0:
            raise ValueError(
                f"n_time (first element of input_shape) must be a positive integer, got {n_time}"
            )

        # Validate n_states
        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError(
                f"n_states (second element of input_shape) must be a positive integer, got {n_states}"
            )

        # Warn for unusual values
        if n_time == 1:
            warnings.warn(
                "n_time=1 means single time step. This may not capture temporal patterns."
            )

        if n_states == 1:
            warnings.warn(
                "n_states=1 means univariate input. Consider multiple features for richer representations."
            )

    @staticmethod
    def validate_fraction_nonzero(fraction_nonzero):
        """Validate fraction_nonzero_entries parameter (if provided separately)"""
        if fraction_nonzero is not None:
            if not isinstance(fraction_nonzero, (int, float)):
                raise TypeError(
                    f"fraction_nonzero_entries must be a float, got {type(fraction_nonzero).__name__}"
                )

            if fraction_nonzero <= 0 or fraction_nonzero > 1:
                raise ValueError(
                    f"fraction_nonzero_entries must be between 0 and 1 (inclusive), got {fraction_nonzero}"
                )

            if fraction_nonzero < 0.1:
                warnings.warn(
                    f"Very low fraction_nonzero_entries ({fraction_nonzero}). Input layer may be too sparse."
                )

    @staticmethod
    def validate_name(name):
        """Validate the name parameter"""
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")

        if not name.strip():
            raise ValueError("name cannot be empty or whitespace")

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all input layer parameters at once"""
        # Check if input_shape is provided
        if "input_shape" not in kwargs:
            raise ValueError("input_shape is a required parameter")

        cls.validate_input_shape(kwargs["input_shape"])

        # Validate optional parameters if provided
        if "fraction_nonzero_entries" in kwargs:
            cls.validate_fraction_nonzero(kwargs["fraction_nonzero_entries"])

        if "name" in kwargs:
            cls.validate_name(kwargs["name"])


class RandomReservoirValidator:
    """Validator for RandomReservoirLayer parameters"""

    @staticmethod
    def validate_nodes(nodes):
        """Validate number of nodes/reservoir size"""
        if not isinstance(nodes, int) or nodes <= 0:
            raise ValueError(f"nodes must be a positive integer, got {nodes}")

        if nodes < 10:
            warnings.warn(
                f"Very small reservoir size ({nodes}). May not have enough capacity."
            )
        elif nodes > 10000:
            warnings.warn(
                f"Very large reservoir size ({nodes}). May be computationally expensive."
            )

    @staticmethod
    def validate_density(density):
        """Validate connection density"""
        if not isinstance(density, (int, float)):
            raise TypeError(f"density must be a float, got {type(density).__name__}")

        if density <= 0 or density > 1:
            raise ValueError(f"density must be > 0 and ≤ 1, got {density}")

        if density < 0.01:
            warnings.warn(
                f"Very low density ({density}). Reservoir may have isolated nodes."
            )
        elif density == 1.0:
            warnings.warn("density=1.0 means fully connected reservoir (no sparsity).")

    @staticmethod
    def validate_activation(activation):
        """Validate activation function"""
        valid_activations = [
            "tanh",
            "sigmoid",
            "relu",
            "linear",
            "softmax",
            "elu",
            "selu",
            "softplus",
            "softsign",
            "hard_sigmoid",
        ]

        if activation not in valid_activations:
            raise ValueError(
                f"activation must be one of {valid_activations}, got '{activation}'"
            )

    @staticmethod
    def validate_leakage_rate(leakage_rate):
        """Validate leakage rate (also called alpha)"""
        if not isinstance(leakage_rate, (int, float)):
            raise TypeError(
                f"leakage_rate must be a float, got {type(leakage_rate).__name__}"
            )

        if leakage_rate < 0 or leakage_rate > 1:
            raise ValueError(
                f"leakage_rate must be between 0 and 1 (inclusive), got {leakage_rate}"
            )

        if leakage_rate == 0:
            warnings.warn("leakage_rate=0 means no memory (purely feedforward).")
        elif leakage_rate == 1:
            warnings.warn("leakage_rate=1 means full memory (no forgetting).")
        elif leakage_rate < 0.01:
            warnings.warn(
                f"Very low leakage_rate ({leakage_rate}). Slow state updates."
            )
        elif leakage_rate > 0.99:
            warnings.warn(
                f"Very high leakage_rate ({leakage_rate}). Minimal forgetting."
            )

    @staticmethod
    def validate_fraction_input(fraction_input):
        """Validate input fraction"""
        if not isinstance(fraction_input, (int, float)):
            raise TypeError(
                f"fraction_input must be a float, got {type(fraction_input).__name__}"
            )

        if fraction_input <= 0 or fraction_input > 1:
            raise ValueError(
                f"fraction_input must be > 0 and ≤ 1, got {fraction_input}"
            )

        if fraction_input < 0.1:
            warnings.warn(
                f"Very low fraction_input ({fraction_input}). Weak input connection."
            )

    @staticmethod
    def validate_spec_rad(spec_rad):
        """Validate spectral radius"""
        if not isinstance(spec_rad, (int, float)):
            raise TypeError(f"spec_rad must be a float, got {type(spec_rad).__name__}")

        if spec_rad <= 0:
            raise ValueError(f"spec_rad must be > 0, got {spec_rad}")

        if spec_rad < 0.1:
            warnings.warn(f"Very low spectral radius ({spec_rad}). May lack dynamics.")
        elif spec_rad > 1.5:
            warnings.warn(f"High spectral radius ({spec_rad}). May be unstable.")

    @staticmethod
    def validate_init_sampling(init_res_sampling):
        """Validate initialization sampling method"""
        valid_sampling = [
            "random_normal",
            "uniform",
            "glorot_normal",
            "glorot_uniform",
            "he_normal",
            "he_uniform",
            "lecun_normal",
            "lecun_uniform",
        ]

        if init_res_sampling not in valid_sampling:
            raise ValueError(
                f"init_res_sampling must be one of {valid_sampling}, got '{init_res_sampling}'"
            )

    @staticmethod
    def validate_seed(seed):
        """Validate random seed"""
        if seed is not None:
            if not isinstance(seed, int):
                raise TypeError(
                    f"seed must be an integer or None, got {type(seed).__name__}"
                )

            if seed < 0 or seed > 2**32 - 1:
                warnings.warn(
                    f"Seed value {seed} may be outside typical random seed range."
                )

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all parameters at once"""
        # Required parameter
        if "nodes" not in kwargs:
            raise ValueError("nodes is a required parameter")

        cls.validate_nodes(kwargs["nodes"])

        # Optional parameters with defaults
        cls.validate_density(kwargs.get("density", 0.1))
        cls.validate_activation(kwargs.get("activation", "tanh"))
        cls.validate_leakage_rate(kwargs.get("leakage_rate", 0.5))
        cls.validate_fraction_input(kwargs.get("fraction_input", 0.8))
        cls.validate_spec_rad(kwargs.get("spec_rad", 0.9))
        cls.validate_init_sampling(kwargs.get("init_res_sampling", "random_normal"))
        cls.validate_seed(kwargs.get("seed", None))


class OutputParameterValidator:
    """Validator for OutputLayer parameters"""

    @staticmethod
    def validate_output_shape(output_shape):
        """Validate the output_shape parameter"""
        if not isinstance(output_shape, tuple):
            raise TypeError(
                f"output_shape must be a tuple, got {type(output_shape).__name__}"
            )

        if len(output_shape) != 2:
            raise ValueError(
                f"output_shape must have 2 dimensions (n_timesteps, n_states), got {len(output_shape)} dimensions"
            )

        n_time, n_states = output_shape

        # Validate n_time
        if not isinstance(n_time, int) or n_time <= 0:
            raise ValueError(
                f"n_time (first element of output_shape) must be a positive integer, got {n_time}"
            )

        # Validate n_states
        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError(
                f"n_states (second element of output_shape) must be a positive integer, got {n_states}"
            )

        # Warnings for edge cases
        if n_time == 1:
            warnings.warn(
                "n_time=1 means single time step output (no sequence prediction)."
            )

        if n_states == 1:
            warnings.warn("n_states=1 means univariate output.")

        if n_time > 1000:
            warnings.warn(
                f"Large output sequence length ({n_time}). May be computationally expensive."
            )

        if n_states > 100:
            warnings.warn(
                f"Large output dimension ({n_states}). May require many output weights."
            )

    @staticmethod
    def validate_fraction_out(fraction_out):
        """Validate fraction_out parameter"""
        if not isinstance(fraction_out, (int, float)):
            raise TypeError(
                f"fraction_out must be a float, got {type(fraction_out).__name__}"
            )

        if fraction_out <= 0 or fraction_out > 1:
            raise ValueError(f"fraction_out must be > 0 and ≤ 1, got {fraction_out}")

        if fraction_out < 0.1:
            warnings.warn(
                f"Very low fraction_out ({fraction_out}). Sparse output connections."
            )
        elif fraction_out == 1.0:
            # This is usually fine, just informational
            pass

    @staticmethod
    def validate_shape_consistency(input_shape, output_shape):
        """Validate that input and output shapes are compatible"""
        if input_shape is None or output_shape is None:
            return

        # Check if time dimensions match (for sequence-to-sequence)
        if input_shape[0] != output_shape[0]:
            warnings.warn(
                f"Input time steps ({input_shape[0]}) != output time steps ({output_shape[0]}). "
                f"May require temporal alignment."
            )

        # Note: n_states can be different (that's the point of transformation)

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all output layer parameters"""
        # Check required parameter
        if "output_shape" not in kwargs:
            raise ValueError("output_shape is a required parameter")

        cls.validate_output_shape(kwargs["output_shape"])

        # Validate optional parameter
        if "fraction_out" in kwargs:
            cls.validate_fraction_out(kwargs["fraction_out"])


class FeedbackParameterValidator:
    """Validator for FeedbackLayer parameters"""

    @staticmethod
    def validate_feedback_shape(feedback_shape):
        """Validate the feedback_shape parameter"""
        if not isinstance(feedback_shape, tuple):
            raise TypeError(
                f"feedback_shape must be a tuple, got {type(feedback_shape).__name__}"
            )

        if len(feedback_shape) != 2:
            raise ValueError(
                f"feedback_shape must have 2 dimensions (n_timesteps, n_states), got {len(feedback_shape)} dimensions"
            )

        n_time, n_states = feedback_shape

        # Validate n_time
        if not isinstance(n_time, int) or n_time <= 0:
            raise ValueError(
                f"n_time (first element of feedback_shape) must be a positive integer, got {n_time}"
            )

        # Validate n_states
        if not isinstance(n_states, int) or n_states <= 0:
            raise ValueError(
                f"n_states (second element of feedback_shape) must be a positive integer, got {n_states}"
            )

        # Warnings for edge cases
        if n_time == 1:
            warnings.warn(
                "n_time=1 means single time step feedback. May not capture temporal dependencies."
            )

        if n_states == 1:
            warnings.warn("n_states=1 means univariate feedback.")

        if n_time > 100:
            warnings.warn(
                f"Long feedback sequence ({n_time}). May create long temporal dependencies."
            )

        if n_states > 50:
            warnings.warn(
                f"High-dimensional feedback ({n_states}). May increase complexity."
            )

    @staticmethod
    def validate_fraction_nonzero(fraction_nonzero):
        """Validate fraction_nonzero_entries parameter"""
        if fraction_nonzero is not None:
            if not isinstance(fraction_nonzero, (int, float)):
                raise TypeError(
                    f"fraction_nonzero_entries must be a float, got {type(fraction_nonzero).__name__}"
                )

            if fraction_nonzero <= 0 or fraction_nonzero > 1:
                raise ValueError(
                    f"fraction_nonzero_entries must be > 0 and ≤ 1, got {fraction_nonzero}"
                )

            if fraction_nonzero < 0.1:
                warnings.warn(
                    f"Very low fraction_nonzero_entries ({fraction_nonzero}). Sparse feedback connections."
                )
            elif fraction_nonzero == 1.0:
                warnings.warn(
                    "fraction_nonzero_entries=1.0 means fully connected feedback (no sparsity)."
                )

    @staticmethod
    def validate_feedback_delay(delay_steps):
        """Validate feedback delay parameter"""
        if delay_steps is not None:
            if not isinstance(delay_steps, int) or delay_steps < 0:
                raise ValueError(
                    f"delay_steps must be a non-negative integer, got {delay_steps}"
                )

            if delay_steps > 10:
                warnings.warn(
                    f"Large feedback delay ({delay_steps} steps). May create long-term dependencies."
                )

    @staticmethod
    def validate_feedback_strength(strength):
        """Validate feedback strength parameter"""
        if strength is not None:
            if not isinstance(strength, (int, float)):
                raise TypeError(
                    f"feedback_strength must be a float, got {type(strength).__name__}"
                )

            if strength < 0:
                raise ValueError(f"feedback_strength must be ≥ 0, got {strength}")

            if strength == 0:
                warnings.warn("feedback_strength=0 means no feedback effect.")
            elif strength > 1.0:
                warnings.warn(
                    f"High feedback strength ({strength}). May cause instability."
                )

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all feedback layer parameters"""
        # Check required parameter
        if "feedback_shape" not in kwargs:
            raise ValueError("feedback_shape is a required parameter")

        cls.validate_feedback_shape(kwargs["feedback_shape"])

        # Validate optional parameters if provided
        if "fraction_nonzero_entries" in kwargs:
            cls.validate_fraction_nonzero(kwargs["fraction_nonzero_entries"])

        if "delay_steps" in kwargs:
            cls.validate_feedback_delay(kwargs["delay_steps"])

        if "feedback_strength" in kwargs:
            cls.validate_feedback_strength(kwargs["feedback_strength"])


class RidgeParameterValidator:
    """Validator for RidgeSK optimizer parameters"""

    @staticmethod
    def validate_name(name):
        """Validate the name parameter"""
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")

        if not name.strip():
            warnings.warn("Empty name provided. Using default naming.")

    @staticmethod
    def validate_alpha(alpha):
        """Validate alpha (regularization strength) parameter"""
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be a float, got {type(alpha).__name__}")

        if alpha < 0:
            raise ValueError(f"alpha must be ≥ 0, got {alpha}")

        if alpha == 0:
            warnings.warn(
                "alpha=0 means no regularization (equivalent to ordinary least squares)."
            )
        elif alpha < 1e-10:
            warnings.warn(f"Very small alpha ({alpha}). Minimal regularization.")
        elif alpha > 1e10:
            warnings.warn(
                f"Very large alpha ({alpha}). Strong regularization may oversmooth."
            )

        # Check for common pitfalls
        if isinstance(alpha, int) and alpha > 1000:
            warnings.warn(
                f"Large integer alpha ({alpha}). Consider using float for fine control."
            )

    @staticmethod
    def validate_fit_intercept(fit_intercept):
        """Validate fit_intercept parameter"""
        if not isinstance(fit_intercept, bool):
            raise TypeError(
                f"fit_intercept must be a boolean, got {type(fit_intercept).__name__}"
            )

    @staticmethod
    def validate_solver(solver):
        """Validate solver parameter"""
        valid_solvers = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
        if solver not in valid_solvers:
            raise ValueError(f"solver must be one of {valid_solvers}, got '{solver}'")

    @staticmethod
    def validate_tolerance(tol):
        """Validate tolerance parameter"""
        if not isinstance(tol, (int, float)):
            raise TypeError(f"tol must be a float, got {type(tol).__name__}")

        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}")

        if tol < 1e-12:
            warnings.warn(f"Very strict tolerance ({tol}). May converge slowly.")
        elif tol > 0.1:
            warnings.warn(
                f"Loose tolerance ({tol}). May converge quickly but less accurately."
            )

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all ridge optimizer parameters"""
        # Validate name if provided
        if "name" in kwargs:
            cls.validate_name(kwargs["name"])

        # Validate alpha (always check, has default)
        cls.validate_alpha(kwargs.get("alpha", 1.0))

        # Validate optional parameters if provided
        if "fit_intercept" in kwargs:
            cls.validate_fit_intercept(kwargs["fit_intercept"])

        if "solver" in kwargs:
            cls.validate_solver(kwargs["solver"])

        if "tol" in kwargs:
            cls.validate_tolerance(kwargs["tol"])


class CompileParameterValidator:
    """Validator for compile method parameters"""

    @staticmethod
    def validate_optimizer(optimizer):
        """Validate optimizer parameter - can be string or optimizer object"""
        # Case 1: String optimizer name
        if isinstance(optimizer, str):
            valid_optimizers = [
                "ridge",
                "sgd",
                "adam",
                "rmsprop",
                "adagrad",
                "lbfgs",
                "svd",
            ]
            if optimizer.lower() not in valid_optimizers:
                raise ValueError(
                    f"optimizer must be one of {valid_optimizers}, got '{optimizer}'"
                )

            if optimizer.lower() == "ridge":
                warnings.warn(
                    "Using Ridge optimizer. Ensure you have sklearn installed."
                )

        # Case 2: Optimizer object (like RidgeSK)
        elif hasattr(optimizer, "fit") or hasattr(optimizer, "__class__"):
            # Check if it looks like a valid optimizer
            optimizer_class_name = optimizer.__class__.__name__

            # Check for common optimizer methods
            if not (hasattr(optimizer, "fit") or hasattr(optimizer, "update")):
                warnings.warn(
                    f"Optimizer object {optimizer_class_name} may not have standard optimizer methods."
                )

            print(f"Using optimizer object: {optimizer_class_name}")

        # Case 3: Invalid type
        else:
            raise TypeError(
                f"optimizer must be a string or optimizer object, got {type(optimizer).__name__}"
            )

    @staticmethod
    def validate_metrics(metrics):
        """Validate metrics parameter"""
        if metrics is None:
            return

        # Define valid metrics
        valid_metrics = [
            "mse",
            "mean_squared_error",
            "mae",
            "mean_absolute_error",
            "r2",
            "r_squared",
            "accuracy",
            "mape",
            "mean_absolute_percentage_error",
            "msle",
            "mean_squared_logarithmic_error",
            "cosine",
            "cosine_similarity",
            "binary_crossentropy",
            "categorical_crossentropy",
        ]

        if isinstance(metrics, str):
            # Check if metric is valid
            if metrics not in valid_metrics:
                # Try to suggest correct spelling
                suggestions = []
                for valid in valid_metrics:
                    if "mean_squared" in metrics.lower() and "mean_squared" in valid:
                        suggestions.append(valid)
                    elif metrics.lower() in valid.lower():
                        suggestions.append(valid)

                if suggestions:
                    error_msg = (
                        f"Unknown metric '{metrics}'. Did you mean: {suggestions}?"
                    )
                else:
                    error_msg = f"Unknown metric '{metrics}'. Available metrics: {valid_metrics}"

                raise ValueError(error_msg)

        elif isinstance(metrics, list):
            if len(metrics) == 0:
                warnings.warn(
                    "Empty metrics list provided. No metrics will be tracked."
                )

            for metric in metrics:
                if not isinstance(metric, str):
                    raise TypeError(
                        f"All metrics must be strings, got {type(metric).__name__}"
                    )

                # Check if metric is valid
                if metric not in valid_metrics:
                    # Try to suggest correct spelling for "mean_squafkfred_error"
                    if "squafkfred" in metric.lower():
                        suggestions = [m for m in valid_metrics if "squared" in m]
                        if suggestions:
                            error_msg = f"Unknown metric '{metric}'. Did you mean 'mean_squared_error'?"
                        else:
                            error_msg = f"Unknown metric '{metric}'. Available metrics: {valid_metrics}"
                    else:
                        # Look for similar metrics
                        suggestions = []
                        for valid in valid_metrics:
                            if (
                                metric.lower() in valid.lower()
                                or valid.lower() in metric.lower()
                            ):
                                suggestions.append(valid)

                        if suggestions:
                            error_msg = f"Unknown metric '{metric}'. Similar metrics: {suggestions}"
                        else:
                            error_msg = f"Unknown metric '{metric}'. Available metrics: {valid_metrics}"

                    raise ValueError(error_msg)

        else:
            raise TypeError(
                f"metrics must be str, list, or None, got {type(metrics).__name__}"
            )

    @staticmethod
    def validate_discard_transients(discard_transients):
        """Validate discard_transients parameter"""
        if not isinstance(discard_transients, int):
            raise TypeError(
                f"discard_transients must be an integer, got {type(discard_transients).__name__}"
            )

        if discard_transients < 0:
            raise ValueError(
                f"discard_transients must be ≥ 0, got {discard_transients}"
            )

        if discard_transients > 1000:
            warnings.warn(
                f"Large discard_transients value ({discard_transients}). Many samples will be discarded."
            )

        if discard_transients > 0:
            warnings.warn(
                f"Discarding first {discard_transients} time steps as transients."
            )

    @staticmethod
    def validate_learning_rate(learning_rate):
        """Validate learning_rate parameter if provided"""
        if learning_rate is not None:
            if not isinstance(learning_rate, (int, float)):
                raise TypeError(
                    f"learning_rate must be a float, got {type(learning_rate).__name__}"
                )

            if learning_rate <= 0:
                raise ValueError(f"learning_rate must be > 0, got {learning_rate}")

            if learning_rate < 1e-6:
                warnings.warn(
                    f"Very small learning_rate ({learning_rate}). May learn very slowly."
                )
            elif learning_rate > 1.0:
                warnings.warn(
                    f"Large learning_rate ({learning_rate}). May cause instability."
                )

    @staticmethod
    def validate_loss(loss):
        """Validate loss function parameter if provided"""
        if loss is not None:
            if not isinstance(loss, str):
                raise TypeError(f"loss must be a string, got {type(loss).__name__}")

            valid_losses = [
                "mse",
                "mae",
                "huber",
                "logcosh",
                "binary_crossentropy",
                "categorical_crossentropy",
                "sparse_categorical_crossentropy",
            ]

            if loss not in valid_losses:
                warnings.warn(
                    f"Loss function '{loss}' may not be standard. Ensure compatibility."
                )

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all compile parameters"""
        # Validate required/commonly used parameters
        if "optimizer" in kwargs:
            cls.validate_optimizer(kwargs["optimizer"])

        if "metrics" in kwargs:
            cls.validate_metrics(kwargs["metrics"])

        if "discard_transients" in kwargs:
            cls.validate_discard_transients(kwargs["discard_transients"])

        # Validate optional parameters if provided
        if "learning_rate" in kwargs:
            cls.validate_learning_rate(kwargs["learning_rate"])

        if "loss" in kwargs:
            cls.validate_loss(kwargs["loss"])

        # Check for incompatible parameters
        if (
            kwargs.get("optimizer", "ridge") == "ridge"
            and kwargs.get("learning_rate") is not None
        ):
            warnings.warn("learning_rate parameter is ignored for Ridge optimizer.")

        if (
            kwargs.get("discard_transients", 0) > 0
            and kwargs.get("optimizer", "ridge") != "ridge"
        ):
            warnings.warn("discard_transients is typically used with Ridge optimizer.")


class FitParameterValidator:
    """Validator for fit method parameters"""

    @staticmethod
    def validate_input_data(x):
        """Validate input data X"""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x_train must be a numpy array, got {type(x).__name__}")

        if x.ndim not in [2, 3]:
            raise ValueError(
                f"x_train must be 2D (samples, features) or 3D (samples, time, features), got shape {x.shape}"
            )

        if x.size == 0:
            raise ValueError(f"x_train is empty, shape {x.shape}")

        # Check for NaN or Inf values
        if np.any(np.isnan(x)):
            raise ValueError("x_train contains NaN values")

        if np.any(np.isinf(x)):
            warnings.warn(
                "x_train contains infinite values. This may cause numerical issues."
            )

    @staticmethod
    def validate_target_data(y):
        """Validate target data y"""
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y_train must be a numpy array, got {type(y).__name__}")

        if y.ndim not in [1, 2, 3]:
            raise ValueError(f"y_train must be 1D, 2D, or 3D, got shape {y.shape}")

        if y.size == 0:
            raise ValueError(f"y_train is empty, shape {y.shape}")

        # Check for NaN values
        if np.any(np.isnan(y)):
            raise ValueError("y_train contains NaN values")

        if np.any(np.isinf(y)):
            warnings.warn(
                "y contains infinite values. This may cause numerical issues."
            )

    @staticmethod
    def validate_data_shapes(x, y):
        """Validate that x_train and y_train shapes are compatible"""
        # Check sample dimension matches
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples mismatch: x_train has {x.shape[0]} samples, "
                f"y_train has {y.shape[0]} samples"
            )

        # For time series data, check time dimension if both are 3D
        if x.ndim == 3 and y.ndim == 3:
            if x.shape[1] != y.shape[1]:
                warnings.warn(
                    f"Time dimension mismatch: x_train has {x.shape[1]} time steps, "
                    f"y_train has {y.shape[1]} time steps. Make sure this is intentional."
                )

    @staticmethod
    def validate_n_init(n_init):
        """Validate n_init parameter"""
        if not isinstance(n_init, int):
            raise TypeError(f"n_init must be an integer, got {type(n_init).__name__}")

        if n_init <= 0:
            raise ValueError(f"n_init must be > 0, got {n_init}")

        if n_init > 100:
            warnings.warn(f"Large n_init value ({n_init}). Training may be slow.")

    @staticmethod
    def validate_store_states(store_states):
        """Validate store_states parameter"""
        if not isinstance(store_states, bool):
            raise TypeError(
                f"store_states must be a boolean, got {type(store_states).__name__}"
            )

        if store_states:
            warnings.warn(
                "store_states=True may use significant memory for large datasets."
            )

    @staticmethod
    def validate_validation_data(val_data):
        """Validate validation_data parameter if provided"""
        if val_data is not None:
            if not isinstance(val_data, tuple) or len(val_data) != 2:
                raise ValueError("validation_data must be a tuple (x_val, y_val)")

            x_val, y_val = val_data
            FitParameterValidator.validate_input_data(x_val)
            FitParameterValidator.validate_target_data(y_val)
            FitParameterValidator.validate_data_shapes(x_val, y_val)

    @staticmethod
    def validate_verbose(verbose):
        """Validate verbose parameter"""
        if not isinstance(verbose, (int, bool)):
            raise TypeError(
                f"verbose must be int or bool, got {type(verbose).__name__}"
            )

        if isinstance(verbose, int):
            if verbose < 0 or verbose > 2:
                warnings.warn(f"verbose={verbose} outside typical range 0-2")

    @staticmethod
    def validate_epochs(epochs):
        """Validate epochs parameter"""
        if not isinstance(epochs, int):
            raise TypeError(f"epochs must be an integer, got {type(epochs).__name__}")

        if epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {epochs}")

        if epochs > 10000:
            warnings.warn(
                f"Large number of epochs ({epochs}). Training may be very slow."
            )
        elif epochs == 1:
            warnings.warn("Only 1 epoch. Model may not converge.")

    @staticmethod
    def validate_batch_size(batch_size, n_samples):
        """Validate batch_size parameter"""
        if not isinstance(batch_size, int):
            raise TypeError(
                f"batch_size must be an integer, got {type(batch_size).__name__}"
            )

        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        if batch_size > n_samples:
            warnings.warn(
                f"batch_size ({batch_size}) > number of samples ({n_samples}). Using batch_size={n_samples}"
            )

        if batch_size < 8 and batch_size != 1:
            warnings.warn(
                f"Small batch_size ({batch_size}). May lead to noisy gradients."
            )

    @classmethod
    def validate_all(cls, x, y, **kwargs):
        """Validate all fit parameters"""
        # Validate main parameters
        cls.validate_input_data(x)
        cls.validate_target_data(y)
        cls.validate_data_shapes(x, y)

        # Validate integer parameters
        cls.validate_n_init(kwargs.get("n_init", 1))

        # Validate boolean parameters
        cls.validate_store_states(kwargs.get("store_states", False))

        # Validate optional parameters if provided
        if "validation_data" in kwargs:
            cls.validate_validation_data(kwargs["validation_data"])

        if "verbose" in kwargs:
            cls.validate_verbose(kwargs["verbose"])

        if "epochs" in kwargs:
            cls.validate_epochs(kwargs["epochs"])

        if "batch_size" in kwargs:
            cls.validate_batch_size(kwargs["batch_size"], x.shape[0])

        # Check for data scaling issues
        cls._check_data_characteristics(x, y)

    @staticmethod
    def _check_data_characteristics(x, y):
        """Check data characteristics and issue warnings"""
        # Check for large values
        x_max = np.abs(x).max()
        y_max = np.abs(y).max()

        if x_max > 1000:
            warnings.warn(
                f"Large values in x_train (max abs: {x_max:.2f}). Consider scaling/normalizing."
            )

        if y_max > 1000:
            warnings.warn(
                f"Large values in y_train (max abs: {y_max:.2f}). Consider scaling/normalizing."
            )

        # Check for constant features
        if x.ndim == 2:
            x_std = x.std(axis=0)
            if np.any(x_std < 1e-10):
                warnings.warn(
                    "Some features in x_train have very low variance. May cause numerical issues."
                )

        # Check for imbalanced regression targets
        if y.ndim == 1:
            y_std = y.std()
            if y_std < 1e-10:
                warnings.warn(
                    "y_train has very low variance. Check if targets are constant."
                )


class VisualizeParameterValidator:
    """Validator for model_visualize method parameters"""

    @staticmethod
    def validate_save(save):
        """Validate save parameter"""
        if not isinstance(save, bool):
            raise TypeError(f"save must be a boolean, got {type(save).__name__}")

    @staticmethod
    def validate_file_name(file_name, save):
        """Validate file_name parameter"""
        if file_name is not None:
            if not isinstance(file_name, str):
                raise TypeError(
                    f"file_name must be a string, got {type(file_name).__name__}"
                )

            if save and not file_name.strip():
                raise ValueError("file_name cannot be empty when save=True")

            # Check for invalid characters in filename
            invalid_chars = ["<", ">", ":", '"', "|", "?", "*", "\\", "/"]
            for char in invalid_chars:
                if char in file_name:
                    raise ValueError(f"Invalid character '{char}' in file_name")

            # Check filename length
            if len(file_name) > 255:
                warnings.warn(
                    f"Long file name ({len(file_name)} chars). May cause filesystem issues."
                )

    @staticmethod
    def validate_file_type(file_type, save):
        """Validate file_type parameter"""
        if file_type is not None:
            if not isinstance(file_type, str):
                raise TypeError(
                    f"file_type must be a string, got {type(file_type).__name__}"
                )

            file_type = file_type.lower().strip()
            valid_types = ["png", "jpg", "jpeg", "pdf", "svg", "eps", "tiff", "bmp"]

            if file_type not in valid_types:
                raise ValueError(
                    f"Unsupported file type '{file_type}'. "
                    f"Supported types: {valid_types}"
                )

            if save and file_type in ["jpg", "jpeg"]:
                warnings.warn(
                    f"{file_type.upper()} format may have lower quality than PNG for diagrams."
                )

    @staticmethod
    def validate_node_colors(node_colors):
        """Validate Node_colors parameter"""
        if node_colors is not None:
            if not isinstance(node_colors, dict):
                raise TypeError(
                    f"Node_colors must be a dictionary, got {type(node_colors).__name__}"
                )

            # Validate color values
            for key, color in node_colors.items():
                if not isinstance(key, str):
                    raise TypeError(
                        f"Node_colors keys must be strings, got {type(key).__name__}"
                    )

                if not isinstance(color, str):
                    raise TypeError(
                        f"Node_colors values must be strings, got {type(color).__name__}"
                    )

                # Check if color is valid
                if not VisualizeParameterValidator._is_valid_color(color):
                    warnings.warn(
                        f"Color '{color}' for key '{key}' may not be a standard color name. "
                        f"Ensure it's recognized by matplotlib."
                    )

    @staticmethod
    def validate_edge_weights(edge_weights):
        """Validate Edge_Weights parameter"""
        if edge_weights is not None:
            if not isinstance(edge_weights, (int, float, list, tuple)):
                raise TypeError(
                    f"Edge_Weights must be numeric or list/tuple, got {type(edge_weights).__name__}"
                )

            if isinstance(edge_weights, (int, float)):
                if edge_weights <= 0:
                    warnings.warn(
                        f"Small edge weight ({edge_weights}). Edges may not be visible."
                    )
                elif edge_weights > 10:
                    warnings.warn(
                        f"Large edge weight ({edge_weights}). Edges may dominate visualization."
                    )
            else:  # list or tuple
                for i, weight in enumerate(edge_weights):
                    if not isinstance(weight, (int, float)):
                        raise TypeError(
                            f"All Edge_Weights must be numeric, got {type(weight).__name__} at index {i}"
                        )

                    if weight < 0:
                        warnings.warn(f"Negative edge weight at index {i} ({weight}).")

    @staticmethod
    def validate_resolution(dpi):
        """Validate DPI/resolution parameter"""
        if dpi is not None:
            if not isinstance(dpi, (int, float)):
                raise TypeError(f"dpi must be numeric, got {type(dpi).__name__}")

            if dpi < 72:
                warnings.warn(f"Low DPI ({dpi}). Output may be pixelated.")
            elif dpi > 600:
                warnings.warn(f"High DPI ({dpi}). File size may be large.")

    @staticmethod
    def validate_figsize(figsize):
        """Validate figsize parameter"""
        if figsize is not None:
            if not isinstance(figsize, (tuple, list)):
                raise TypeError(
                    f"figsize must be tuple or list, got {type(figsize).__name__}"
                )

            if len(figsize) != 2:
                raise ValueError(
                    f"figsize must have 2 elements (width, height), got {len(figsize)}"
                )

            width, height = figsize
            if not isinstance(width, (int, float)) or not isinstance(
                height, (int, float)
            ):
                raise TypeError("figsize elements must be numeric")

            if width <= 0 or height <= 0:
                raise ValueError("figsize dimensions must be > 0")

            if width > 50 or height > 50:
                warnings.warn(
                    f"Large figure size ({width}x{height}). May be difficult to view."
                )

    @staticmethod
    def _is_valid_color(color):
        """Check if a color string is valid"""
        # Simple check - can be expanded
        valid_color_patterns = [
            "red",
            "blue",
            "green",
            "yellow",
            "black",
            "white",
            "gray",
            "grey",
            "orange",
            "purple",
            "pink",
            "brown",
            "cyan",
            "magenta",
            "light",
            "dark",
            "sky",
            "sea",
            "forest",
            "gold",
            "silver",
            "navy",
            "maroon",
        ]

        # Check for matplotlib color formats
        import re

        # Hex color: #RGB or #RRGGBB
        hex_pattern = r"^#(?:[0-9a-fA-F]{3}){1,2}$"
        # RGB tuple string: "rgb(255, 255, 255)" or "rgba(255, 255, 255, 1.0)"
        rgb_pattern = r"^rgba?\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*(?:,\s*\d*\.?\d+\s*)?\)$"

        color_lower = color.lower()
        if any(pattern in color_lower for pattern in valid_color_patterns):
            return True
        elif re.match(hex_pattern, color):
            return True
        elif re.match(rgb_pattern, color_lower):
            return True

        return False

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all visualization parameters"""
        # Validate main parameters
        if "save" in kwargs:
            cls.validate_save(kwargs["save"])

        save = kwargs.get("save", False)

        if "file_name" in kwargs:
            cls.validate_file_name(kwargs["file_name"], save)

        if "file_type" in kwargs:
            cls.validate_file_type(kwargs["file_type"], save)

        if "Node_colors" in kwargs:
            cls.validate_node_colors(kwargs["Node_colors"])

        if "Edge_Weights" in kwargs:
            cls.validate_edge_weights(kwargs["Edge_Weights"])

        # Validate optional parameters
        if "dpi" in kwargs:
            cls.validate_resolution(kwargs["dpi"])

        if "figsize" in kwargs:
            cls.validate_figsize(kwargs["figsize"])

        if "title" in kwargs:
            if not isinstance(kwargs["title"], str):
                raise TypeError(
                    f"title must be a string, got {type(kwargs['title']).__name__}"
                )

        # Check for conflicts
        if kwargs.get("save", False):
            if kwargs.get("file_name") is None:
                warnings.warn(
                    "save=True but no file_name provided. Using default name."
                )

            if kwargs.get("file_type") is None:
                warnings.warn(
                    "save=True but no file_type provided. Using default format."
                )


class PredictParameterValidator:
    """Validator for predict method parameters"""

    @staticmethod
    def validate_input_data(x, model_fitted=True):
        """Validate input data X for prediction"""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")

        if x.ndim not in [2, 3]:
            raise ValueError(
                f"x must be 2D (samples, features) or 3D (samples, time, features), "
                f"got shape {x.shape} with {x.ndim} dimensions"
            )

        if x.size == 0:
            raise ValueError(f"x is empty, shape {x.shape}")

        # Check for NaN or Inf values
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values")

        if np.any(np.isinf(x)):
            warnings.warn("x contains infinite values. Predictions may be unreliable.")

        # Check data characteristics
        PredictParameterValidator._check_data_range(x)

        # Check if model is fitted
        if model_fitted:
            warnings.warn("Model not fitted. Predictions may not be accurate.")

    @staticmethod
    def _check_data_range(x):
        """Check if data values are in reasonable range"""
        x_abs_max = np.abs(x).max()
        if x_abs_max > 1e6:
            warnings.warn(
                f"Large values in x (max abs: {x_abs_max:.2e}). "
                f"Consider scaling/normalizing."
            )

        # Check for constant features (may cause issues)
        if x.ndim == 2:
            x_std = x.std(axis=0)
            if np.any(x_std < 1e-10):
                warnings.warn(
                    "Some features have very low variance. "
                    "Predictions may be unreliable."
                )

    @staticmethod
    def validate_batch_size(batch_size, n_samples):
        """Validate batch_size parameter"""
        if batch_size is not None:
            if not isinstance(batch_size, int):
                raise TypeError(
                    f"batch_size must be an integer, got {type(batch_size).__name__}"
                )

            if batch_size <= 0:
                raise ValueError(f"batch_size must be > 0, got {batch_size}")

            if batch_size > n_samples:
                warnings.warn(
                    f"batch_size ({batch_size}) > number of samples ({n_samples}). "
                    f"Using batch_size={n_samples}"
                )

    @staticmethod
    def validate_verbose(verbose):
        """Validate verbose parameter"""
        if verbose is not None:
            if not isinstance(verbose, (int, bool)):
                raise TypeError(
                    f"verbose must be int or bool, got {type(verbose).__name__}"
                )

            if isinstance(verbose, int) and (verbose < 0 or verbose > 2):
                warnings.warn(f"verbose={verbose} outside typical range 0-2")

    @staticmethod
    def validate_return_states(return_states):
        """Validate return_states parameter"""
        if not isinstance(return_states, bool):
            raise TypeError(
                f"return_states must be a boolean, got {type(return_states).__name__}"
            )

    @staticmethod
    def validate_return_confidence(return_confidence):
        """Validate return_confidence parameter"""
        if not isinstance(return_confidence, bool):
            raise TypeError(
                f"return_confidence must be a boolean, got {type(return_confidence).__name__}"
            )

    @staticmethod
    def validate_model_state(model_fitted, model_compiled):
        """Validate model state before prediction"""
        if not model_compiled:
            raise RuntimeError("Model must be compiled before prediction")

        if not model_fitted:
            warnings.warn("Model not fitted. Predictions may not be accurate.")

    @staticmethod
    def validate_shape_compatibility(x_shape, expected_input_shape):
        """Validate that input shape matches model expectations"""
        if expected_input_shape is not None:
            # For 2D data
            if len(x_shape) == 2 and len(expected_input_shape) == 2:
                if x_shape[1] != expected_input_shape[1]:
                    raise ValueError(
                        f"Input feature dimension mismatch: "
                        f"x has {x_shape[1]} features, "
                        f"model expects {expected_input_shape[1]} features"
                    )

            # For 3D data (time series)
            elif len(x_shape) == 3 and len(expected_input_shape) == 2:
                # x: (samples, time, features), expected: (time, features)
                if x_shape[2] != expected_input_shape[1]:
                    raise ValueError(
                        f"Input feature dimension mismatch: "
                        f"x has {x_shape[2]} features, "
                        f"model expects {expected_input_shape[1]} features"
                    )

    @classmethod
    def validate_all(cls, x, **kwargs):
        """Validate all predict parameters"""
        # Validate input data
        model_fitted = kwargs.get("model_fitted", True)
        cls.validate_input_data(x, model_fitted)

        # Validate optional parameters
        if "batch_size" in kwargs:
            cls.validate_batch_size(kwargs["batch_size"], x.shape[0])

        if "verbose" in kwargs:
            cls.validate_verbose(kwargs["verbose"])

        if "return_states" in kwargs:
            cls.validate_return_states(kwargs["return_states"])

        if "return_confidence" in kwargs:
            cls.validate_return_confidence(kwargs["return_confidence"])


class SequenceScalarParameterValidator:
    """Validator for sequence_to_scalar function parameters"""

    @staticmethod
    def validate_name(name):
        """Validate sequence name parameter"""
        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}")

        valid_names = ["sine_prediction", "sine_to_cosine", "sin_to_cos2"]

        if name not in valid_names:
            raise ValueError(
                f"Invalid sequence name: '{name}'. " f"Available options: {valid_names}"
            )

    @staticmethod
    def validate_n_batch(n_batch):
        """Validate n_batch parameter"""
        if not isinstance(n_batch, int):
            raise TypeError(f"n_batch must be an integer, got {type(n_batch).__name__}")

        if n_batch < 1:
            raise ValueError(f"n_batch must be ≥ 1, got {n_batch}")

        if n_batch < 2:
            warnings.warn(
                f"Small n_batch ({n_batch}). May not have enough samples for train/test split."
            )
        elif n_batch > 10000:
            warnings.warn(f"Large n_batch ({n_batch}). May use significant memory.")

    @staticmethod
    def validate_n_states(n_states):
        """Validate n_states parameter"""
        if not isinstance(n_states, int):
            raise TypeError(
                f"n_states must be an integer, got {type(n_states).__name__}"
            )

        if n_states < 1:
            raise ValueError(f"n_states must be ≥ 1, got {n_states}")

        if n_states == 1:
            warnings.warn("n_states=1 means univariate sequences.")
        elif n_states > 100:
            warnings.warn(
                f"Large n_states ({n_states}). May create high-dimensional data."
            )

    @staticmethod
    def validate_n_time_in(n_time_in):
        """Validate n_time_in parameter"""
        if not isinstance(n_time_in, int):
            raise TypeError(
                f"n_time_in must be an integer, got {type(n_time_in).__name__}"
            )

        if n_time_in < 1:
            raise ValueError(f"n_time_in must be ≥ 1, got {n_time_in}")

        if n_time_in == 1:
            warnings.warn("n_time_in=1 means single time step input sequences.")
        elif n_time_in > 1000:
            warnings.warn(
                f"Large n_time_in ({n_time_in}). May create long input sequences."
            )

    @staticmethod
    def validate_n_time_out(n_time_out):
        """Validate n_time_out is always 1 for scalar output"""
        if n_time_out != 1:
            raise ValueError(
                f"n_time_out must be 1 for sequence_to_scalar, got {n_time_out}"
            )

    @staticmethod
    def validate_noise_level(noise_level):
        """Validate noise_level parameter"""
        if noise_level is not None:
            if not isinstance(noise_level, (int, float)):
                raise TypeError(
                    f"noise_level must be numeric, got {type(noise_level).__name__}"
                )

            if noise_level < 0:
                raise ValueError(f"noise_level must be ≥ 0, got {noise_level}")

            if noise_level > 1.0:
                warnings.warn(f"High noise_level ({noise_level}). May obscure signal.")

    @staticmethod
    def validate_frequencies(frequencies):
        """Validate frequencies parameter"""
        if frequencies is not None:
            if isinstance(frequencies, (int, float)):
                if frequencies <= 0:
                    raise ValueError(f"Frequency must be > 0, got {frequencies}")
            elif isinstance(frequencies, (list, tuple)):
                for i, freq in enumerate(frequencies):
                    if not isinstance(freq, (int, float)):
                        raise TypeError(
                            f"All frequencies must be numeric, got {type(freq).__name__} at index {i}"
                        )
                    if freq <= 0:
                        raise ValueError(
                            f"Frequency must be > 0, got {freq} at index {i}"
                        )
            else:
                raise TypeError(
                    f"frequencies must be numeric or list/tuple, got {type(frequencies).__name__}"
                )

    @staticmethod
    def validate_scalar_output_shape(y_shape):
        """Validate that output is scalar (single time step)"""
        if len(y_shape) == 3:  # (samples, time, features)
            if y_shape[1] != 1:
                warnings.warn(
                    f"Output has {y_shape[1]} time steps. Expected 1 for scalar output."
                )
        elif len(y_shape) == 2:  # (samples, features) - already scalar
            if y_shape[1] > 1:
                warnings.warn(f"Output has {y_shape[1]} features. May not be scalar.")

    @classmethod
    def validate_all(cls, **kwargs):
        """Validate all sequence_to_scalar parameters"""
        # Required parameter
        if "name" not in kwargs:
            raise ValueError("name is a required parameter")

        cls.validate_name(kwargs["name"])

        # Validate main parameters
        cls.validate_n_batch(kwargs.get("n_batch", 50))
        cls.validate_n_states(kwargs.get("n_states", 1))
        cls.validate_n_time_in(kwargs.get("n_time_in", 2))

        # n_time_out is always 1 for scalar output
        cls.validate_n_time_out(kwargs.get("n_time_out", 1))

        # Validate optional parameters if provided
        if "noise_level" in kwargs:
            cls.validate_noise_level(kwargs["noise_level"])

        if "frequencies" in kwargs:
            cls.validate_frequencies(kwargs["frequencies"])

        # Check for parameter consistency
        name = kwargs.get("name", "")
        n_batch = kwargs.get("n_batch", 50)

        if n_batch < 2 and name:
            warnings.warn(
                f"n_batch={n_batch} may be too small for proper train/test split."
            )


class EvaluateParameterValidator:
    """Validator for evaluate method parameters"""

    @staticmethod
    def validate_input_data(x, model_fitted=True):
        """Validate input data X for evaluation"""
        if not isinstance(x, np.ndarray):
            raise TypeError(f"x must be a numpy array, got {type(x).__name__}")

        if x.ndim not in [2, 3]:
            raise ValueError(
                f"x must be 2D (samples, features) or 3D (samples, time, features), "
                f"got shape {x.shape} with {x.ndim} dimensions"
            )

        if x.size == 0:
            raise ValueError(f"x is empty, shape {x.shape}")

        # Check for NaN or Inf values
        if np.any(np.isnan(x)):
            raise ValueError("x contains NaN values")

        if np.any(np.isinf(x)):
            warnings.warn("x contains infinite values. Evaluation may be unreliable.")

        # Check data characteristics
        EvaluateParameterValidator._check_data_range(x)

    @staticmethod
    def validate_target_data(y):
        """Validate target data y for evaluation"""
        if not isinstance(y, np.ndarray):
            raise TypeError(f"y must be a numpy array, got {type(y).__name__}")

        if y.ndim not in [1, 2, 3]:
            raise ValueError(
                f"y must be 1D, 2D, or 3D, got shape {y.shape} with {y.ndim} dimensions"
            )

        if y.size == 0:
            raise ValueError(f"y is empty, shape {y.shape}")

        # Check for NaN values
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")

        if np.any(np.isinf(y)):
            warnings.warn("y contains infinite values. Evaluation may be unreliable.")

    @staticmethod
    def _check_data_range(x):
        """Check if data values are in reasonable range"""
        x_abs_max = np.abs(x).max()
        if x_abs_max > 1e6:
            warnings.warn(
                f"Large values in x (max abs: {x_abs_max:.2e}). "
                f"Consider scaling/normalizing."
            )

    @staticmethod
    def validate_data_shapes(x, y, predictions_shape=None):
        """Validate that x and y shapes are compatible"""
        # Check sample dimension matches
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples mismatch: x has {x.shape[0]} samples, "
                f"y has {y.shape[0]} samples"
            )

        # If predictions are available, check compatibility
        if predictions_shape is not None:
            if predictions_shape[0] != x.shape[0]:
                raise ValueError(
                    f"Predictions sample mismatch: x has {x.shape[0]} samples, "
                    f"predictions have {predictions_shape[0]} samples"
                )

            # Check if predictions match target shape (excluding batch dimension)
            if len(predictions_shape) == len(y.shape):
                if predictions_shape[1:] != y.shape[1:]:
                    warnings.warn(
                        f"Shape mismatch: predictions {predictions_shape[1:]} vs "
                        f"targets {y.shape[1:]}. Some metrics may not work properly."
                    )

    @staticmethod
    def validate_metrics(metrics):
        """Validate metrics parameter"""
        if metrics is None:
            return

        valid_metrics = [
            "mse",
            "mean_squared_error",
            "mae",
            "mean_absolute_error",
            "r2",
            "r_squared",
            "r2_score",
            "accuracy",
            "mape",
            "mean_absolute_percentage_error",
            "msle",
            "mean_squared_logarithmic_error",
            "cosine",
            "cosine_similarity",
            "binary_crossentropy",
            "categorical_crossentropy",
            "explained_variance",
            "max_error",
            "median_absolute_error",
        ]

        if isinstance(metrics, str):
            # Check if metric is valid
            if metrics not in valid_metrics:
                # Try to suggest correct spelling
                suggestions = []
                for valid in valid_metrics:
                    if metrics.lower() in valid.lower():
                        suggestions.append(valid)

                if suggestions:
                    error_msg = (
                        f"Unknown metric '{metrics}'. Did you mean: {suggestions}?"
                    )
                else:
                    error_msg = f"Unknown metric '{metrics}'. Available metrics: {valid_metrics}"

                raise ValueError(error_msg)

        elif isinstance(metrics, list):
            if len(metrics) == 0:
                raise ValueError("metrics list cannot be empty")

            for i, metric in enumerate(metrics):
                if not isinstance(metric, str):
                    raise TypeError(
                        f"All metrics must be strings, got {type(metric).__name__} at index {i}"
                    )

                # Check if metric is valid
                if metric not in valid_metrics:
                    suggestions = []
                    for valid in valid_metrics:
                        if (
                            metric.lower() in valid.lower()
                            or valid.lower() in metric.lower()
                        ):
                            suggestions.append(valid)

                    if suggestions:
                        error_msg = f"Unknown metric '{metric}' at index {i}. Similar metrics: {suggestions}"
                    else:
                        error_msg = f"Unknown metric '{metric}' at index {i}. Available metrics: {valid_metrics}"

                    raise ValueError(error_msg)

        else:
            raise TypeError(
                f"metrics must be str, list, or None, got {type(metrics).__name__}"
            )

    @staticmethod
    def validate_model_state(model_fitted, model_compiled):
        """Validate model state before evaluation"""
        if not model_compiled:
            raise RuntimeError("Model must be compiled before evaluation")

        if not model_fitted:
            warnings.warn("Model not fitted. Evaluation results may not be meaningful.")

    @staticmethod
    def validate_return_dict(return_dict):
        """Validate return_dict parameter"""
        if not isinstance(return_dict, bool):
            raise TypeError(
                f"return_dict must be a boolean, got {type(return_dict).__name__}"
            )

    @staticmethod
    def validate_sample_weight(sample_weight, n_samples):
        """Validate sample_weight parameter"""
        if sample_weight is not None:
            if not isinstance(sample_weight, np.ndarray):
                raise TypeError(
                    f"sample_weight must be a numpy array, got {type(sample_weight).__name__}"
                )

            if sample_weight.ndim != 1:
                raise ValueError(
                    f"sample_weight must be 1D, got shape {sample_weight.shape}"
                )

            if len(sample_weight) != n_samples:
                raise ValueError(
                    f"sample_weight length {len(sample_weight)} "
                    f"does not match number of samples {n_samples}"
                )

            if np.any(sample_weight < 0):
                warnings.warn("sample_weight contains negative values")

            if np.any(np.isnan(sample_weight)):
                raise ValueError("sample_weight contains NaN values")

    @classmethod
    def validate_all(cls, x, y, **kwargs):
        """Validate all evaluate parameters"""
        # Validate main parameters
        cls.validate_input_data(x)
        cls.validate_target_data(y)

        # Validate data compatibility
        predictions_shape = kwargs.get("predictions_shape", None)
        cls.validate_data_shapes(x, y, predictions_shape)

        # Validate model state if provided
        if "model_fitted" in kwargs and "complue" in kwargs:
            cls.validate_model_state(kwargs["model_fitted"], kwargs["model_compiled"])

        # Validate metrics
        if "metrics" in kwargs:
            cls.validate_metrics(kwargs["metrics"])

        # Validate optional parameters
        if "return_dict" in kwargs:
            cls.validate_return_dict(kwargs["return_dict"])

        if "sample_weight" in kwargs:
            cls.validate_sample_weight(kwargs["sample_weight"], x.shape[0])

        # Check if evaluation makes sense
        if "predictions_shape" in kwargs:
            if kwargs["predictions_shape"][0] < 10:
                warnings.warn(
                    f"Small number of samples ({kwargs['predictions_shape'][0]}) "
                    f"for evaluation. Results may not be reliable."
                )


class AutoRCParameterValidator:
    """
    Validator that handles both RidgeSK class and instance.
    """

    @staticmethod
    def validate_optimizer(optimizer: Union[str, Any, type]) -> dict:
        """
        Validate optimizer parameter - handles strings, instances, and classes.

        """
        result = {
            "type": None,
            "name": None,
            "value": optimizer,
            "is_sklearn_compatible": False,
            "requires_instantiation": False,
        }

        # Case 1: String optimizer
        if isinstance(optimizer, str):
            result["type"] = "string"
            result["name"] = optimizer.lower()

            valid_strings = [
                "ridge",
                "sgd",
                "adam",
                "rmsprop",
                "adagrad",
                "lbfgs",
                "svd",
                "auto",
                "ridgesk",
            ]

            if result["name"] not in valid_strings:
                suggestions = [
                    opt
                    for opt in valid_strings
                    if result["name"] in opt or opt in result["name"]
                ]
                error_msg = f"Unknown optimizer '{optimizer}'"
                if suggestions:
                    error_msg += f". Did you mean: {suggestions}?"
                raise ValueError(error_msg)

            if result["name"] in ["ridge", "ridgesk"]:
                result["is_sklearn_compatible"] = True

        # Case 2: Object instance (has fit method)
        elif hasattr(optimizer, "fit"):
            result["type"] = "instance"
            result["name"] = optimizer.__class__.__name__

            # Check if it's sklearn-style
            sklearn_methods = ["fit", "predict", "score"]
            if all(hasattr(optimizer, method) for method in sklearn_methods):
                result["is_sklearn_compatible"] = True

            # Show hyperparameters
            if hasattr(optimizer, "alpha"):
                warnings.warn(f"Using {result['name']} with alpha={optimizer.alpha}")

        # Case 3: Class type (like RidgeSK class itself)
        elif hasattr(optimizer, "fit") or hasattr(optimizer, "__class__"):
            # Check if it looks like a valid optimizer
            optimizer_class_name = optimizer.__class__.__name__

            # Check for common optimizer methods
            if not (hasattr(optimizer, "fit") or hasattr(optimizer, "update")):
                warnings.warn(
                    f"Optimizer object {optimizer_class_name} may not have standard optimizer methods."
                )

            print(f"Using optimizer object: {optimizer_class_name}")
        # Case 4: Invalid type
        else:
            raise TypeError(
                f"Optimizer must be string, class, or instance with fit() method, "
                f"got {type(optimizer).__name__}"
            )

        return result

    @staticmethod
    def validate_metrics(metrics: Union[str, List[str], None]) -> List[str]:
        """Validate metrics parameter."""
        if metrics is None:
            return ["mse"]

        if isinstance(metrics, str):
            metrics_list = [metrics.lower()]
        elif isinstance(metrics, list):
            if len(metrics) == 0:
                raise ValueError("Metrics list cannot be empty")
            metrics_list = [m.lower() if isinstance(m, str) else m for m in metrics]
        else:
            raise TypeError("Metrics must be str, list, or None")

        valid_metrics = [
            "mse",
            "mean_squared_error",
            "mae",
            "mean_absolute_error",
            "r2",
            "r_squared",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "mape",
            "cosine",
        ]

        for i, metric in enumerate(metrics_list):
            if not isinstance(metric, str):
                raise TypeError(f"Metric at index {i} must be string")
            if metric not in valid_metrics:
                similar = [m for m in valid_metrics if metric in m or m in metric]
                error_msg = f"Unknown metric '{metric}'"
                if similar:
                    error_msg += f". Similar: {similar}"
                raise ValueError(error_msg)

        return metrics_list

    @staticmethod
    def validate_discard_transients(
        discard_transients: int, data_length: Optional[int] = None
    ) -> int:
        """Validate discard_transients parameter."""
        if not isinstance(discard_transients, int):
            raise TypeError("discard_transients must be integer")

        if discard_transients < 0:
            raise ValueError("discard_transients must be >= 0")

        if data_length and discard_transients >= data_length:
            raise ValueError(
                f"discard_transients ({discard_transients}) >= data length ({data_length})"
            )

        if discard_transients > 0:
            warnings.warn(f"Discarding first {discard_transients} time steps")

        return discard_transients

    @classmethod
    def validate_compile_params(cls, **kwargs) -> dict:
        """
        Validate all AutoRC_compile parameters.
        """
        validated = {}

        # Validate optimizer
        if "optimizer" in kwargs:
            optimizer_info = cls.validate_optimizer(kwargs["optimizer"])
            validated["optimizer_info"] = optimizer_info

            # If optimizer is a class, instantiate it
            if optimizer_info["requires_instantiation"]:
                try:
                    # Try to instantiate with default parameters
                    optimizer_instance = optimizer_info["value"]()
                    validated["optimizer"] = optimizer_instance
                    warnings.warn(
                        f"Instantiated {optimizer_info['name']} with default parameters"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Failed to instantiate {optimizer_info['name']}: {e}"
                    )
            else:
                validated["optimizer"] = optimizer_info["value"]

        # Validate metrics
        if "metrics" in kwargs:
            validated["metrics"] = cls.validate_metrics(kwargs["metrics"])

        # Validate discard_transients
        if "discard_transients" in kwargs:
            validated["discard_transients"] = cls.validate_discard_transients(
                kwargs["discard_transients"], kwargs.get("data_length")
            )

        # Store other parameters
        for key in [
            "learning_rate",
            "auto_tune",
            "n_trials",
            "cv_folds",
            "random_state",
            "verbose",
        ]:
            if key in kwargs:
                validated[key] = kwargs[key]

        return validated


class AutoRCPredictValidator:

    @staticmethod
    def validate_x(x: np.ndarray, method_name: str = "AutoRC_predict") -> np.ndarray:

        if x is None:
            raise ValueError(f"{method_name}: Input x cannot be None")

        if not isinstance(x, np.ndarray):
            raise TypeError(
                f"{method_name}: x must be a numpy array, " f"got {type(x).__name__}"
            )

        if x.size == 0:
            raise ValueError(f"{method_name}: x cannot be empty")

        # Check dimensions - should be 2D or 3D
        if x.ndim not in [2, 3]:
            raise ValueError(
                f"{method_name}: x must be 2D or 3D array, " f"got shape {x.shape}"
            )

        # Convert 2D to 3D (single batch)
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])

        # Check for NaN/inf values
        if np.any(np.isnan(x)):
            warnings.warn(f"{method_name}: Input x contains NaN values")

        if np.any(np.isinf(x)):
            warnings.warn(f"{method_name}: Input x contains infinite values")

        return x

    @staticmethod
    def validate_fb_scale(
        fb_scale: float, method_name: str = "AutoRC_predict"
    ) -> float:

        if not isinstance(fb_scale, (int, float)):
            raise TypeError(
                f"{method_name}: fb_scale must be a number, "
                f"got {type(fb_scale).__name__}"
            )

        fb_scale = float(fb_scale)

        # Warning for extreme values
        if fb_scale < 0:
            warnings.warn(
                f"{method_name}: Negative fb_scale ({fb_scale}) may cause instability"
            )
        elif fb_scale > 2.0:
            warnings.warn(
                f"{method_name}: Large fb_scale ({fb_scale}) may cause exploding feedback"
            )
        elif fb_scale == 0:
            warnings.warn(f"{method_name}: fb_scale=0, feedback is disabled")

        return fb_scale

    @staticmethod
    def validate_T_run(
        T_run: int, x_shape: Tuple[int, ...], method_name: str = "AutoRC_predict"
    ) -> int:
        if not isinstance(T_run, int):
            raise TypeError(
                f"{method_name}: T_run must be an integer, "
                f"got {type(T_run).__name__}"
            )

        if T_run <= 0:
            raise ValueError(f"{method_name}: T_run must be positive, got {T_run}")

        # Check if T_run matches input timesteps
        n_timesteps = x_shape[1]
        if T_run != n_timesteps:
            warnings.warn(
                f"{method_name}: T_run ({T_run}) doesn't match input timesteps ({n_timesteps})"
            )

        # Warning for large T_run
        if T_run > 10000:
            warnings.warn(
                f"{method_name}: Large T_run ({T_run}), prediction may be slow"
            )

        return T_run

    @staticmethod
    def validate_feedback_indices(
        feedback_indices: Optional[np.ndarray],
        n_input_states: Optional[int] = None,
        n_output_states: Optional[int] = None,
        method_name: str = "AutoRC_predict",
    ) -> Optional[np.ndarray]:
        """
        Validate feedback indices array

        Args:
            feedback_indices: Indices of states to use for feedback
            n_input_states: Number of input states
            n_output_states: Number of output states
            method_name: Name of the method for error messages

        Returns:
            Validated feedback_indices or None
        """
        if feedback_indices is None:
            return None

        if not isinstance(feedback_indices, np.ndarray):
            raise TypeError(
                f"{method_name}: feedback_indices must be numpy array, "
                f"got {type(feedback_indices).__name__}"
            )

        # Must be 1D array
        if feedback_indices.ndim != 1:
            raise ValueError(
                f"{method_name}: feedback_indices must be 1D, "
                f"got shape {feedback_indices.shape}"
            )

        # Must contain integers
        if not np.issubdtype(feedback_indices.dtype, np.integer):
            raise ValueError(f"{method_name}: feedback_indices must contain integers")

        # Check bounds based on context
        if n_output_states is not None:
            # If we know output states, check feedback indices are within output bounds
            if np.any(feedback_indices < 0):
                raise ValueError(
                    f"{method_name}: feedback_indices contains negative indices"
                )

            if np.any(feedback_indices >= n_output_states):
                raise ValueError(
                    f"{method_name}: feedback_indices contains indices "
                    f"outside output states [0, {n_output_states-1}]"
                )

        # Check for duplicates
        unique_indices = np.unique(feedback_indices)
        if len(unique_indices) != len(feedback_indices):
            warnings.warn(
                f"{method_name}: feedback_indices contains duplicates, using unique values"
            )
            feedback_indices = unique_indices

        # Sort for consistency
        feedback_indices = np.sort(feedback_indices)

        return feedback_indices

    @staticmethod
    def validate_model_state(model, method_name: str = "AutoRC_predict") -> None:

        # Check if model is compiled
        if not getattr(model, "compile", False):
            raise RuntimeError(
                f"{method_name}: Model must be compiled. Call AutoRC_compile() first."
            )

        # Check if model has reservoir and readout
        if not hasattr(model, "reservoir_layer") or model.reservoir_layer is None:
            raise RuntimeError(f"{method_name}: Model reservoir not initialized")

        if not hasattr(model, "readout") or model.readout is None:
            warnings.warn(
                f"{method_name}: Model readout not initialized. Prediction may not be accurate."
            )

    @staticmethod
    def validate_dimension_consistency(
        x: np.ndarray,
        model_input_dim: int,
        model_output_dim: int,
        method_name: str = "AutoRC_predict",
    ) -> None:

        n_batch, n_timesteps, n_states = x.shape

        # Check input dimension
        if n_states != model_input_dim:
            raise ValueError(
                f"{method_name}: Input dimension mismatch. "
                f"Expected {model_input_dim} states, got {n_states}"
            )

        # Warn if output dimension is different from input dimension
        if model_output_dim != model_input_dim:
            warnings.warn(
                f"{method_name}: Model input_dim ({model_input_dim}) != output_dim ({model_output_dim})"
            )

    @classmethod
    def validate_all(
        cls,
        x: np.ndarray,
        fb_scale: float,
        T_run: int,
        feedback_indices: Optional[np.ndarray] = None,
        model=None,
        model_input_dim: Optional[int] = None,
        model_output_dim: Optional[int] = None,
        method_name: str = "AutoRC_predict",
    ) -> dict:
        """
        Validate all AutoRC_predict parameters

        Returns:
            Dictionary of validated parameters
        """
        validated = {}

        # Validate model state
        if model:
            cls.validate_model_state(model, method_name)

        # Validate input data
        validated["x"] = cls.validate_x(x, method_name)

        # Validate fb_scale
        validated["fb_scale"] = cls.validate_fb_scale(fb_scale, method_name)

        # Validate T_run
        validated["T_run"] = cls.validate_T_run(
            T_run, validated["x"].shape, method_name
        )

        # Validate feedback indices
        validated["feedback_indices"] = cls.validate_feedback_indices(
            feedback_indices, model_input_dim, model_output_dim, method_name
        )

        # Validate dimension consistency
        if model_input_dim is not None:
            cls.validate_dimension_consistency(
                validated["x"],
                model_input_dim,
                model_output_dim if model_output_dim is not None else model_input_dim,
                method_name,
            )

        # Additional warnings based on parameters
        if validated["fb_scale"] > 0 and validated["feedback_indices"] is None:
            warnings.warn(
                f"{method_name}: fb_scale > 0 but no feedback_indices specified. "
                f"Will use all outputs for feedback."
            )

        return validated
