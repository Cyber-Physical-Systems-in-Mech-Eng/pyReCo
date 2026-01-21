from functools import wraps
import warnings
from pyreco.reservoir_validator import (
    AutoRCParameterValidator,
    AutoRCPredictValidator,
    CompileParameterValidator,
    EvaluateParameterValidator,
    FeedbackParameterValidator,
    FitParameterValidator,
    InputParameterValidator,
    OutputParameterValidator,
    PredictParameterValidator,
    RandomReservoirValidator,
    ReservoirParameterValidator,
    RidgeParameterValidator,
    SequenceParameterValidator,
    SequenceScalarParameterValidator,
    VisualizeParameterValidator,
)
import inspect
import numpy as np
from typing import Union


def validate_sequence_params(func):
    """Decorator for sequence_to_sequence parameter validation"""

    def wrapper(name, n_batch=50, n_states=2, n_time=3, **kwargs):
        # Combine parameters for validation
        all_params = {
            "name": name,
            "n_batch": n_batch,
            "n_states": n_states,
            "n_time": n_time,
            **kwargs,
        }

        # Run all validations
        SequenceParameterValidator.validate_all(**all_params)

        # Ensure n_batch is at least 2 (as in original function)
        n_batch = max(n_batch, 2)

        print(
            f"Generating sequence '{name}' with n_batch={n_batch}, n_states={n_states}, n_time={n_time}"
        )

        # Call original function
        result = func(name, n_batch=n_batch, n_states=n_states, n_time=n_time, **kwargs)

        # Validate returned data
        if result and len(result) == 4:
            X_train, X_test, y_train, y_test = result
            print("Generated data shapes:")
            print(f"  X_train: {X_train.shape}")
            print(f"  X_test: {X_test.shape}")
            print(f"  y_train: {y_train.shape}")
            print(f"  y_test: {y_test.shape}")

        return result

    return wrapper


def validate_reservoir_params(func):
    """Decorator to validate reservoir computer parameters before initialization"""

    def wrapper(self, **kwargs):
        # Run all validations
        ReservoirParameterValidator.validate_all(**kwargs)

        # Store validated parameters as instance variables
        self.num_nodes = kwargs.get("num_nodes", 100)
        self.density = kwargs.get("density", 0.8)
        self.activation = kwargs.get("activation", "tanh")
        self.leakage_rate = kwargs.get("leakage_rate", 0.5)
        self.spec_rad = kwargs.get("spec_rad", 0.9)
        self.fraction_input = kwargs.get("fraction_input", 1.0)
        self.fraction_output = kwargs.get("fraction_output", 1.0)
        self.n_time_in = kwargs.get("n_time_in", None)
        self.n_time_out = kwargs.get("n_time_out", None)
        self.n_states_in = kwargs.get("n_states_in", None)
        self.n_states_out = kwargs.get("n_states_out", None)
        self.metrics = kwargs.get("metrics", "mean_squared_error")
        self.optimizer = kwargs.get("optimizer", "ridge")
        self.init_res_sampling = kwargs.get("init_res_sampling", "random_normal")

        # Call the original __init__ method
        return func(self, **kwargs)

    return wrapper


def validate_input_params(func):
    """Decorator to validate input layer parameters"""

    def wrapper(self, **kwargs):
        # Run all validations
        InputParameterValidator.validate_all(**kwargs)

        # Store validated parameters
        self.input_shape = kwargs["input_shape"]
        self.n_time = self.input_shape[0]
        self.n_states = self.input_shape[1]

        # Store optional parameters with defaults
        self.name = kwargs.get("name", "input_layer")
        self.type = kwargs.get("type", "input")
        self.fraction_nonzero_entries = kwargs.get("fraction_nonzero_entries", 1.0)

        # Call the original __init__ method
        return func(self, **kwargs)

    return wrapper


def validate_random_reservoir_params(func):
    """Decorator for RandomReservoirLayer parameter validation"""

    def wrapper(self, **kwargs):
        # Run all validations
        RandomReservoirValidator.validate_all(**kwargs)

        # Store validated parameters
        self.nodes = kwargs["nodes"]
        self.density = kwargs.get("density", 0.1)
        self.activation = kwargs.get("activation", "tanh")
        self.leakage_rate = kwargs.get("leakage_rate", 0.5)
        self.fraction_input = kwargs.get("fraction_input", 0.8)
        self.spec_rad = kwargs.get("spec_rad", 0.9)
        self.init_res_sampling = kwargs.get("init_res_sampling", "random_normal")
        self.seed = kwargs.get("seed", None)

        # Call original __init__
        return func(self, **kwargs)

    return wrapper


def validate_output_params(func):
    """Decorator for OutputLayer parameter validation"""

    def wrapper(self, *args, **kwargs):
        # Get the function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(self, *args, **kwargs)
        bound_args.apply_defaults()

        # Convert to dict (excluding 'self')
        params = dict(bound_args.arguments)
        params.pop("self", None)

        # Run all validations
        OutputParameterValidator.validate_all(**params)

        # Store validated parameters as instance variables
        for key, value in params.items():
            setattr(self, key, value)

        # Set derived attributes
        self.output_shape = params["output_shape"]
        self.n_time = self.output_shape[0]
        self.n_states = self.output_shape[1]
        self.fraction_out = params.get("fraction_out", 1.0)
        self.type = "output"

        # Call original __init__
        return func(self, *args, **kwargs)

    return wrapper


def validate_feedback_params(func):
    """Decorator for FeedbackLayer parameter validation"""

    def wrapper(self, **kwargs):
        # Run all validations
        FeedbackParameterValidator.validate_all(**kwargs)

        # Store validated parameters
        self.shape = kwargs["feedback_shape"]
        self.n_time = self.shape[0]
        self.n_states = self.shape[1]

        # Set default values
        self.name = kwargs.get("name", "feedback_layer")
        self.type = "feedback"
        self.fraction_nonzero_entries = kwargs.get("fraction_nonzero_entries", 1.0)
        self.delay_steps = kwargs.get("delay_steps", 1)
        self.feedback_strength = kwargs.get("feedback_strength", 0.5)

        # Call original __init__
        return func(self, **kwargs)

    return wrapper


def validate_ridge_params(func):
    """Decorator for RidgeSK parameter validation"""

    def wrapper(self, **kwargs):
        # Run all validations
        RidgeParameterValidator.validate_all(**kwargs)

        # Store validated parameters
        self.name = kwargs.get("name", "")
        self.alpha = kwargs.get("alpha", 1.0)

        # Store optional parameters with defaults
        self.fit_intercept = kwargs.get("fit_intercept", True)
        self.normalize = kwargs.get("normalize", False)
        self.copy_X = kwargs.get("copy_X", True)
        self.max_iter = kwargs.get("max_iter", None)
        self.tol = kwargs.get("tol", 1e-3)
        self.solver = kwargs.get("solver", "auto")
        self.random_state = kwargs.get("random_state", None)

        # Call original __init__
        return func(self, **kwargs)

    return wrapper


def validate_compile_params(func):
    """Decorator for compile method parameter validation"""

    def wrapper(self, optimizer="ridge", metrics=None, discard_transients=0, **kwargs):
        # Combine parameters for validation
        all_params = {
            "optimizer": optimizer,
            "metrics": metrics,
            "discard_transients": discard_transients,
            **kwargs,
        }

        # Run all validations
        CompileParameterValidator.validate_all(**all_params)

        # Store validated parameters
        self.optimizer = optimizer  # Can be string or object
        self.metrics = metrics if metrics is not None else ["mse"]
        self.discard_transients = discard_transients

        # Determine optimizer name
        if isinstance(optimizer, str):
            self.optimizer_name = optimizer.lower()
        else:
            self.optimizer_name = optimizer.__class__.__name__.lower()

        # Store optional parameters
        self.learning_rate = kwargs.get("learning_rate", 0.001)
        self.loss = kwargs.get("loss", "mse")
        self.batch_size = kwargs.get("batch_size", 32)
        self.epochs = kwargs.get("epochs", 100)
        self.verbose = kwargs.get("verbose", 1)

        # Initialize optimizer if string was provided
        if isinstance(optimizer, str):
            self._initialize_optimizer()

        print(f"Model compiled with {self.optimizer_name} optimizer")

        # Call original compile method
        return func(
            self,
            optimizer=optimizer,
            metrics=metrics,
            discard_transients=discard_transients,
            **kwargs,
        )

    return wrapper


def validate_fit_params(func):
    """Decorator for fit method parameter validation"""

    def wrapper(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_init: int = 1,
        store_states: bool = False,
        **kwargs,
    ):

        # Combine all parameters
        all_params = {
            "x": x,
            "y": y,
            "n_init": n_init,
            "store_states": store_states,
            **kwargs,
        }

        # Run all validations - pass as keyword arguments only
        FitParameterValidator.validate_all(**all_params)  # No separate x, y

        # Store validated parameters
        self.x = x
        self.y = y
        self.n_init = n_init
        self.store_states = store_states

        # Store optional parameters with defaults
        self.validation_data = kwargs.get("validation_data", None)
        self.verbose = kwargs.get("verbose", 1)
        self.epochs = kwargs.get("epochs", 100)
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.callbacks = kwargs.get("callbacks", [])

        # Training state
        self.is_fitted = False
        self.training_history = {}
        self.stored_states = None if not store_states else []

        print(f"Starting fit with {x.shape[0]} samples, n_init={n_init}")

        # Call original fit method
        result = func(self, x, y, n_init=n_init, store_states=store_states, **kwargs)

        # Update training state
        self.is_fitted = True
        print("Fit completed successfully")

        return result

    return wrapper


def validate_visualize_params(func):
    """Decorator for model_visualize parameter validation"""

    def wrapper(
        self,
        save=False,
        file_name=None,
        file_type=None,
        Node_colors=None,
        Edge_Weights=None,
        **kwargs,
    ):

        # Apply defaults BEFORE validation
        if Edge_Weights is None:
            Edge_Weights = 0.7

        # Default colors
        default_colors = {
            "CWinp": "black",
            "CWres_inp": "lightcoral",
            "CWres_out": "lightgreen",
            "CWres_both": "orange",
            "CWres_internal": "lightblue",
            "CWout": "black",
            "Winp": "blue",
            "Wout": "red",
            "CWres": "grey",
        }

        # Merge user colors with defaults
        if Node_colors is None:
            final_colors = default_colors
        else:
            final_colors = default_colors.copy()
            # Only update non-None values from user
            for key, color in Node_colors.items():
                if color is not None:
                    final_colors[key] = color

        # Normalize file type
        normalized_file_type = None
        if file_type is not None:
            normalized_file_type = file_type.lower().strip()

        # Now validate everything (including defaults)
        all_params = {
            "save": save,
            "file_name": file_name,
            "file_type": normalized_file_type,
            "Node_colors": final_colors,  # Use merged colors for validation
            "Edge_Weights": Edge_Weights,
            **kwargs,
        }

        VisualizeParameterValidator.validate_all(**all_params)

        # Store everything
        self._visualize_save = save
        self._visualize_file_name = file_name
        self._visualize_file_type = normalized_file_type
        self._visualize_node_colors = final_colors
        self._visualize_edge_weights = Edge_Weights

        # Store optional parameters
        for key, default in [
            ("dpi", 300),
            ("figsize", (12, 8)),
            ("title", "Reservoir Network Visualization"),
            ("show_legend", True),
            ("node_size", 300),
            ("font_size", 10),
        ]:
            setattr(self, f"_visualize_{key}", kwargs.get(key, default))

        print(f"Visualization configured: save={save}")

        # Call function with processed parameters
        return func(
            self,
            save=save,
            file_name=file_name,
            file_type=file_type,
            Node_colors=final_colors,
            Edge_Weights=Edge_Weights,
            **kwargs,
        )

    return wrapper


def validate_predict_params(func):
    """Decorator for predict method parameter validation"""

    def wrapper(self, x: np.ndarray, **kwargs):
        # Check model state
        model_fitted = getattr(self, "is_fitted", False)
        model_compiled = getattr(self, "compiled", False)

        # Combine parameters for validation
        all_params = {
            "x": x,
            "model_fitted": model_fitted,
            "model_compiled": model_compiled,
            **kwargs,
        }

        # Add expected input shape if available
        if hasattr(self, "input_layer") and hasattr(self.input_layer, "input_shape"):
            all_params["expected_input_shape"] = self.input_layer.input_shape

        # Run all validations
        PredictParameterValidator.validate_all(**all_params)

        # Store validated parameters
        self._predict_x = x
        self._predict_batch_size = kwargs.get("batch_size", None)
        self._predict_verbose = kwargs.get("verbose", 0)
        self._predict_return_states = kwargs.get("return_states", False)
        self._predict_return_confidence = kwargs.get("return_confidence", False)

        # Track prediction
        self._last_prediction_input = x
        self._last_prediction_time = None  # Will be set after prediction

        print(f"Starting prediction with {x.shape[0]} samples")

        # Call original predict method
        result = func(self, x, **kwargs)

        return result

    return wrapper


def validate_sequence_scalar_params(func):
    """Decorator for sequence_to_scalar parameter validation"""

    def wrapper(name, n_batch=50, n_states=1, n_time_in=2, **kwargs):
        # n_time_out is always 1 for scalar output
        n_time_out = 1

        # Combine parameters for validation
        all_params = {
            "name": name,
            "n_batch": n_batch,
            "n_states": n_states,
            "n_time_in": n_time_in,
            "n_time_out": n_time_out,
            **kwargs,
        }

        # Run all validations
        SequenceScalarParameterValidator.validate_all(**all_params)

        # Ensure n_batch is at least 2 (as in original function)
        n_batch = max(n_batch, 2)

        print(f"Generating sequence-to-scalar '{name}' with:")
        print(
            f"n_batch={n_batch}, n_states={n_states}, n_time_in={n_time_in}, n_time_out={n_time_out}"
        )

        # Call original function
        result = func(
            name, n_batch=n_batch, n_states=n_states, n_time_in=n_time_in, **kwargs
        )

        # Validate returned data
        if result and len(result) == 4:
            X_train, X_test, y_train, y_test = result

            # Validate output is scalar
            SequenceScalarParameterValidator.validate_scalar_output_shape(y_train.shape)
            SequenceScalarParameterValidator.validate_scalar_output_shape(y_test.shape)

            print("Generated data shapes:")
            print(f"  X_train: {X_train.shape}")
            print(f"  X_test:  {X_test.shape}")
            print(f"  y_train: {y_train.shape} (should be scalar output)")
            print(f"  y_test:  {y_test.shape} (should be scalar output)")

            # Verify scalar output
            if y_train.shape[1] != 1 and len(y_train.shape) == 3:
                warnings.warn(f"y_train has {y_train.shape[1]} time steps, expected 1")
            if y_test.shape[1] != 1 and len(y_test.shape) == 3:
                warnings.warn(f"y_test has {y_test.shape[1]} time steps, expected 1")

        return result

    return wrapper


def validate_evaluate_params(func):
    """Decorator for evaluate method parameter validation"""

    def wrapper(
        self,
        x: np.ndarray,
        y: np.ndarray,
        metrics: Union[str, list, None] = None,
        **kwargs,
    ):

        # Check model state
        model_fitted = getattr(self, "is_fitted", False)
        model_compiled = getattr(self, "compiled", False)

        # Get predictions shape if model can predict
        predictions_shape = None
        if hasattr(self, "predict"):
            try:
                predictions = self.predict(x)
                predictions_shape = predictions.shape
            except:
                predictions_shape = None

        # Combine parameters for validation
        all_params = {
            "x": x,
            "y": y,
            "metrics": metrics,
            "model_fitted": model_fitted,
            "model_compiled": model_compiled,
            "predictions_shape": predictions_shape,
            **kwargs,
        }

        # Run all validations
        EvaluateParameterValidator.validate_all(**all_params)

        # Store validated parameters
        self._evaluate_x = x
        self._evaluate_y = y
        self._evaluate_metrics = metrics if metrics is not None else ["mse"]

        # Store optional parameters
        self._evaluate_return_dict = kwargs.get("return_dict", False)
        self._evaluate_sample_weight = kwargs.get("sample_weight", None)
        self._evaluate_verbose = kwargs.get("verbose", 1)

        # Track evaluation
        self._last_evaluation_input = x
        self._last_evaluation_time = None
        self._evaluation_count = getattr(self, "_evaluation_count", 0) + 1

        print(f"Starting evaluation with {x.shape[0]} samples")
        if metrics:
            print(f"Metrics: {metrics}")

        # Call original evaluate method
        result = func(self, x, y, metrics=metrics, **kwargs)

        # Store evaluation metadata
        from datetime import datetime

        self._last_evaluation_time = datetime.now()

        # Process result
        if isinstance(result, tuple):
            print(f"Evaluation completed. Results: {result}")
        elif isinstance(result, dict):
            print(f"Evaluation completed. Got {len(result)} metric(s)")

        return result

    return wrapper


def validate_autorc_compile_params(func):
    """
    Decorator that handles autoRC compile validation.
    """

    def wrapper(self, optimizer="ridge", metrics=None, discard_transients=0, **kwargs):
        # Combine all parameters
        all_params = {
            "optimizer": optimizer,
            "metrics": metrics,
            "discard_transients": discard_transients,
            **kwargs,
        }

        # Validate all parameters
        validated = AutoRCParameterValidator.validate_compile_params(**all_params)

        # Store validated parameters
        self.optimizer_name = validated["optimizer_info"]["name"]
        self.optimizer_type = validated["optimizer_info"]["type"]
        self.optimizer_obj = validated["optimizer"]

        self.metrics = validated.get("metrics", ["mse"])
        self.discard_transients = validated.get("discard_transients", 0)

        # Store optional parameters
        self.learning_rate = validated.get("learning_rate", 0.001)
        self.auto_tune = validated.get("auto_tune", False)
        self.n_trials = validated.get("n_trials", 100 if self.auto_tune else None)
        self.verbose = validated.get("verbose", 1)

        # Print compilation info
        if self.verbose > 0:
            print("✓ AutoRC Model Compiled")
            print(f"  Optimizer: {self.optimizer_name} ({self.optimizer_type})")
            print(f"  Metrics: {self.metrics}")
            print(f"  Discard transients: {self.discard_transients}")

            if self.optimizer_type == "instance":
                obj_name = type(self.optimizer_obj).__name__
                print(f"  Optimizer instance: {obj_name}")
            elif self.optimizer_type == "class":
                print(f"  Optimizer class: {self.optimizer_name}")

        # Call original method
        return func(
            self,
            optimizer=optimizer,
            metrics=metrics,
            discard_transients=discard_transients,
            **kwargs,
        )

    return wrapper


def validate_autorc_predict(func):
    """
    Decorator for AutoRC_predict method with validation
    """

    @wraps(func)
    def wrapper(
        self,
        x: np.ndarray,
        fb_scale: float,
        T_run: int,
        feedback_indices: np.ndarray = None,
        **kwargs,
    ) -> np.ndarray:

        method_name = func.__name__

        # Get model dimensions
        model_input_dim = getattr(self, "n_states", getattr(self, "n_inputs", None))
        model_output_dim = getattr(
            self, "n_states_out", getattr(self, "n_outputs", None)
        )

        # If model_output_dim not specified, assume same as input
        if model_output_dim is None:
            model_output_dim = model_input_dim

        # Validate all parameters
        validated_params = AutoRCPredictValidator.validate_all(
            x=x,
            fb_scale=fb_scale,
            T_run=T_run,
            feedback_indices=feedback_indices,
            model=self,
            model_input_dim=model_input_dim,
            model_output_dim=model_output_dim,
            method_name=method_name,
        )

        # Store prediction parameters for debugging
        self._last_prediction_params = {
            "fb_scale": validated_params["fb_scale"],
            "T_run": validated_params["T_run"],
            "feedback_indices": (
                validated_params["feedback_indices"].copy()
                if validated_params["feedback_indices"] is not None
                else None
            ),
            "input_shape": validated_params["x"].shape,
            "timestamp": np.datetime64("now"),
        }

        # Log prediction info if verbose
        verbose_level = getattr(self, "verbose", getattr(self, "_verbose", 0))
        if verbose_level > 0:
            print(f"\n{'='*60}")
            print("AUTORC PREDICTION STARTED")
            print(f"{'='*60}")
            print(f"  Method: {method_name}")
            print(f"  Batch size: {validated_params['x'].shape[0]}")
            print(f"  Timesteps: {validated_params['x'].shape[1]}")
            print(f"  Input states: {validated_params['x'].shape[2]}")
            print(f"  T_run: {validated_params['T_run']}")
            print(f"  Feedback scale: {validated_params['fb_scale']:.4f}")
            if validated_params["feedback_indices"] is not None:
                print(f"  Feedback indices: {validated_params['feedback_indices']}")
            else:
                print("Feedback indices: All outputs")
            print(f"{'='*60}\n")

        # Call the original prediction method
        result = func(
            self,
            x=validated_params["x"],
            fb_scale=validated_params["fb_scale"],
            T_run=validated_params["T_run"],
            feedback_indices=validated_params["feedback_indices"],
            **kwargs,
        )

        # Validate the result
        if result is not None:
            # Store result info
            # self._last_prediction_params["result_shape"] = result.shape

            if verbose_level > 0:
                print(f"\n{'='*60}")
                print("PREDICTION COMPLETED")
                print(f"{'='*60}")
                print(f"  Length shape: {len(result)}")
                print(f"{'='*60}")

        return result

    return wrapper
