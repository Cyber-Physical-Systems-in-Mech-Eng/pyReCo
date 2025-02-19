"""
Higher-level model definition for default models (built on custom models):

Wrapper for lower-level implementation of RCs. Instead of the Sequential-API-type syntax, this will provide
sklearn-ready models, which under the hood build Sequential-API-type models and ship them.

Currently contains a lot of duplicate code, which needs to be ported to the lower-level implementations.
"""

import numpy as np
from typing import Union
from abc import ABC, abstractmethod

from .custom_models import RC, CustomModel
from .layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from .metrics import mse, mae


class Model(ABC):

    def __init__(self, activation: str = "tanh", leakage_rate: float = 0.3):
        """
        Initializes a Model instance with basic architectural hyperparameters.

        Parameters
        ----------
        activation : str, optional
            The activation function used in the model (default is "tanh").
        leakage_rate : float, optional
            The leakage rate, which controls the memory effect in reservoir computing (default is 0.3).

        Notes
        -----
        The activation function determines how the model processes inputs, and the leakage rate
        influences how much of the previous state is retained over time.
        """
        # basic architectural hyperparameters
        self.activation: str = activation
        self.leakage_rate: float = leakage_rate

        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the model to the given training data.

        This is an abstract method and must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            The input training data, typically of shape (n_samples, n_features).
        y : np.ndarray
            The target values corresponding to `X`, typically of shape (n_samples,).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Notes
        -----
        Subclasses should define the specific training procedure based on the model type.
        """
        # fits the model to the given training data
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates predictions for new input data.

        This is an abstract method and must be implemented by subclasses.

        Parameters
        ----------
        X : np.ndarray
            The input data for which predictions are to be made, typically of shape 
            (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted outputs corresponding to `X`, with shape (n_samples, ...)
            depending on the specific model implementation.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.

        Notes
        -----
        Subclasses should define the specific prediction logic based on the model type.
        """
        # returns predictions for new input data
        pass

    def compile(self, optimizer: str = "ridge", metrics: list = ["mse"]):
        """
        Configures the model for training by setting the optimizer and evaluation metrics.

        Parameters
        ----------
        optimizer : str, optional
            The optimization algorithm to use (default is "ridge").
        metrics : list, optional
            A list of metric names to evaluate the model's performance (default is ["mse"]).

        Notes
        -----
        This method is similar to TensorFlow's `compile` function, allowing the user
        to specify the optimizer and metrics before training the model.
        """
        # sets up things like optimizer and metrics (like in TensorFlow)
        self.optimizer = optimizer
        self.metrics = metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: str | list = ["mse"]):
        """
        Evaluates the model performance on given input data using specified metrics.

        Parameters
        ----------
        X : np.ndarray
            The input data for evaluation, typically of shape (n_samples, n_features).
        y : np.ndarray
            The ground truth values corresponding to `X`, typically of shape (n_samples,).
        metrics : str or list, optional
            A metric name or a list of metric names to compute the model's performance (default is ["mse"]).

        Returns
        -------
        dict
            A dictionary mapping each metric to its computed value.

        Notes
        -----
        This method runs predictions on `X` and compares them to `y` using the specified metrics.
        If multiple metrics are provided, their results are returned in a dictionary.
        """
        # let model run predictions for input data X and return the metrics against the ground truth y
        pass

    # @abstractmethod
    def set_params(self, **get_params):
        """
        Sets model parameters dynamically. 

        This method is required for scikit-learn compatibility, allowing users to update 
        model attributes via keyword arguments.

        Parameters
        ----------
        **get_params : dict
            Keyword arguments where keys correspond to model attribute names and values 
            are the new values to be assigned.

        Returns
        -------
        self : object
            The model instance with updated parameters.

        Notes
        -----
        - This method updates the instance attributes dynamically using `setattr`.
        - It enables integration with scikit-learn's `set_params` interface.
    """
        # needed for scikit-learn compatibility
        for parameter, value in get_params.items():
            setattr(self, parameter, value)
        return self

    # @abstractmethod
    def get_params(self, deep=True):
        """
        Retrieves the model parameters as a dictionary.

        This method is required for scikit-learn compatibility and allows access to 
        the model's hyperparameters.

        Parameters
        ----------
        deep : bool, optional
            If True, recursively retrieves parameters of sub-objects that are estimators (not implemented here). 
            Default is True.

        Returns
        -------
        dict
            A dictionary containing the model's parameters.

        Notes
        -----
        - This method enables integration with scikit-learn's `get_params` interface.
        - The returned dictionary contains key-value pairs of model attributes.
        - Additional attributes (e.g., `optimizer`, `metrics_available`, `metrics`) can be included if needed.
        """
        # needed for scikit-learn compatibility
        return {
            "activation": self.activation,
            "leakage_rate": self.leakage_rate,
            # 'optimizer': self.optimizer,
            # 'metrics_available': self.metrics_available,
            # 'metrics': self.metrics,
        }


"""
A classical Reservoir Computer (basic vanilla version)
"""


class ReservoirComputer(Model):
    # implements a very classic random reservoir

    def __init__(
        self,
        num_nodes: int = 100,
        density: float = 0.8,
        activation: str = "tanh",
        leakage_rate: float = 0.5,
        spec_rad: float = 0.9,
        fraction_input: float = 1.0,
        fraction_output: float = 1.0,
        n_time_out=None,
        n_time_in=None,
        n_states_in=None,
        n_states_out=None,
        metrics: Union[str, list] = "mean_squared_error",
        optimizer: str = "ridge",
        init_res_sampling="random_normal",
    ):
        """
        Initializes a Reservoir Computer (RC) model.

        This class extends `Model` to create a reservoir computing system with a 
        randomly initialized reservoir and tunable hyperparameters.

        Parameters
        ----------
        num_nodes : int, optional
            Number of nodes in the reservoir, by default 100.
        density : float, optional
            Connection density of the reservoir network, by default 0.8.
        activation : str, optional
            Activation function for the reservoir layer, by default "tanh".
        leakage_rate : float, optional
            Leakage rate of the reservoir, controlling state update speed, by default 0.5.
        spec_rad : float, optional
            Spectral radius of the reservoir weight matrix, by default 0.9.
        fraction_input : float, optional
            Fraction of input nodes connected to the reservoir, by default 1.0.
        fraction_output : float, optional
            Fraction of reservoir nodes connected to the output, by default 1.0.
        n_time_out : int, optional
            Number of time steps in the output (if known), by default None.
        n_time_in : int, optional
            Number of time steps in the input (if known), by default None.
        n_states_in : int, optional
            Number of input states (features), by default None.
        n_states_out : int, optional
            Number of output states (targets), by default None.
        metrics : str or list, optional
            Evaluation metric(s) for model training, by default "mean_squared_error".
        optimizer : str, optional
            Optimization method used for training, by default "ridge".
        init_res_sampling : str, optional
            Initialization method for the reservoir weights, by default "random_normal".

        Attributes
        ----------
        model : RC
            The initialized reservoir computing model.
        trainable_weights : int
            Number of trainable parameters in the readout layer.
        
        Notes
        -----
        - The input/output shapes may be unknown at initialization.
        - The reservoir structure is generated randomly based on the given parameters.
        - Compatible with scikit-learn API.

        """
        # initialize parent class
        super().__init__(activation=activation, leakage_rate=leakage_rate)

        # initialize child class
        self.num_nodes = num_nodes  # former N
        self.density = density
        self.spec_rad = spec_rad
        self.fraction_input = fraction_input
        self.fraction_output = fraction_output

        # dimensionalities of the mapping problem
        self.n_time_out = n_time_out
        self.n_time_in = n_time_in
        self.n_states_in = n_states_in
        self.n_states_out = n_states_out

        self.optimizer = optimizer
        self.metrics = metrics
        self.init_res_sampling = init_res_sampling

        self.trainable_weights: int  # number of trainable weights

        # create a RC from a random reservoir. We do not know about the shapes of input and output at this stage
        self.model = RC()

    def fit(
        self, X: np.ndarray, y: np.ndarray, n_init: int = 1, store_states: bool = False
    ):
        """
        Fits the Reservoir Computer model to the given training data.

        This method computes the optimal model parameters (readout matrix) by training the 
        reservoir computer using the provided input and target data. The model is constructed 
        from the input data, a reservoir layer, and a readout layer. The readout layer is 
        trained using the least-squares method.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape [n_batch, n_time_in, n_states_in], where:
            - n_batch : Number of training samples in the batch.
            - n_time_in : Number of time steps for the input sequence.
            - n_states_in : Number of input states (features).
        y : np.ndarray
            Target data of shape [n_batch, n_time_out, n_states_out], where:
            - n_batch : Number of training samples in the batch.
            - n_time_out : Number of time steps for the output sequence.
            - n_states_out : Number of output states (targets).
        n_init : int, optional
            Number of times to sample the initial reservoir states, by default 1.
        store_states : bool, optional
            If True, stores the full time trace of reservoir states, which is memory-heavy, 
            by default False.

        Returns
        -------
        history : object
            History object that stores training information, depending on the implementation 
            of the `fit` method in the model class.

        Raises
        ------
        ValueError
            If the input or target data contains complex numbers.
        TypeError
            If the input or target data is of object type.

        Notes
        -----
        - This method constructs the model from the `InputLayer`, `RandomReservoirLayer`, and `ReadoutLayer`.
        - The input data X and target data y must be in a format suitable for univariate or multivariate 
        time series tasks.
        - The model is compiled using the specified optimizer and metrics before fitting.
        - The training process utilizes the reservoir's readout layer to compute the model's predictions.
        """
        # Computes the model weights (readout matrix) through fitting the training data.

        # expects data in particular format that is reasonable for univariate/multivariate time series data
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # - y target data of shape [n_batch, n_time_out, n_states_out]
        # - n_init: number of times that initial reservoir states are sampled.
        # - store_states returns the full time trace of reservoir states (memory-heavy!)
        # finds the optimal model parameters (W_out): trains dense layer at output

        # TODO call some helper function with in-depth dimensionality and sanity checks
        if np.iscomplexobj(X) or np.iscomplexobj(y):
            raise ValueError("Complex data not supported")

        # check for object data types
        if X.dtype == "O" or y.dtype == "O":
            raise TypeError("Data type 'object' not supported")

        # obtain the input and output shapes
        n_batch, self.n_time_in, self.n_states_in = X.shape[0], X.shape[1], X.shape[2]
        self.n_time_out, self.n_states_out = y.shape[1], y.shape[2]

        # translate into the shapes requested by the layered model API
        input_shape = (self.n_time_in, self.n_states_in)
        output_shape = (self.n_time_out, self.n_states_out)

        # compose a model from layers. The model was instantiated in the __init__
        self.model.add(InputLayer(input_shape=input_shape))
        self.model.add(
            RandomReservoirLayer(
                nodes=self.num_nodes,
                density=self.density,
                activation=self.activation,
                leakage_rate=self.leakage_rate,
                spec_rad=self.spec_rad,
                fraction_input=self.fraction_input,
                init_res_sampling=self.init_res_sampling,
            )
        )
        self.model.add(ReadoutLayer(output_shape, fraction_out=self.fraction_output))

        # compile the model
        self.model.compile(optimizer=self.optimizer, metrics=self.metrics)

        # fit to training data
        history = self.model.fit(X=X, y=y, n_init=n_init, store_states=store_states)

        self.trainable_weights = self.model.num_trainable_weights

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts output data for the given input data.

        This method generates predictions based on the input data using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape [n_batch, n_time_in, n_states_in], where:
            - n_batch : Number of samples in the batch.
            - n_time_in : Number of time steps in the input sequence.
            - n_states_in : Number of features (states) in the input data.

        Returns
        -------
        y_pred : np.ndarray
            Predicted output data of shape [n_batch, n_time_out, n_states_out], where:
            - n_batch : Number of samples in the batch.
            - n_time_out : Number of time steps in the predicted sequence.
            - n_states_out : Number of predicted states (outputs).

        Raises
        ------
        ValueError
            If the input data contains complex numbers.
        TypeError
            If the input data is of object type.

        Notes
        -----
        - This method assumes that the model has been trained and is ready for prediction.
        - The model's prediction logic is encapsulated in the `predict` method of the model's underlying layers.
        """
        # returns predictions for given data X
        # expects:
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # returns:
        # - y_pred predicted data of shape [n_batch, n_time_out, n_states_out]

        # just a dummy here. TODO insert the actual .predict function
        #

        # check for object data types in X
        if X.dtype == "O":
            raise TypeError("Data type 'object' not supported")

        # check for complex data types
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        y_pred = self.model.predict(X=X)

        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: list = ["mse"]):
        """
        Evaluates the model on the given input data and returns the specified metrics.

        This method runs predictions for the input data and compares the predictions
        with the ground truth to compute evaluation metrics.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape [n_batch, n_time_in, n_states_in], where:
            - n_batch : Number of samples in the batch.
            - n_time_in : Number of time steps in the input sequence.
            - n_states_in : Number of features (states) in the input data.

        y : np.ndarray
            Ground truth data of shape [n_batch, n_time_out, n_states_out], where:
            - n_batch : Number of samples in the batch.
            - n_time_out : Number of time steps in the target sequence.
            - n_states_out : Number of target states (outputs).

        metrics : list of str, optional, default=["mse"]
            List of metric names to evaluate. Available options include:
            - "mse" : Mean squared error
            - "mae" : Mean absolute error
            - "r2" : R-squared
            - and other metrics implemented in the model.

        Returns
        -------
        metric_values : dict
            A dictionary where the keys are the names of the metrics, and the values are the computed metric values.
            For example, {"mse": 0.05, "mae": 0.02}.

        Raises
        ------
        ValueError
            If the input data contains complex numbers.
        TypeError
            If the input data is of object type.

        Notes
        -----
        - This method assumes that the model has been trained and is ready for evaluation.
        - The evaluation is performed by calling the `evaluate` method of the model's underlying layers.
        """
        # let model run predictions for input data X and return the metrics against the ground truth y
        metric_values = self.model.evaluate(X=X, y=y, metrics=metrics)

        return metric_values

    def get_params(self, deep=True):
        """
        Gets the hyperparameters of the model, for compatibility with scikit-learn's API.

        This method returns a dictionary of all the hyperparameters (parameters) of the model, which can 
        be useful for model inspection, grid search, and other tools from scikit-learn. 

        Parameters
        ----------
        deep : bool, optional, default=True
            If True, the method will return parameters from sub-objects (e.g., nested models or layers). 
            If False, only top-level parameters will be included.

        Returns
        -------
        params : dict
            A dictionary containing the model's hyperparameters. Keys are parameter names (e.g., 
            'num_nodes', 'density', etc.) and values are the corresponding values of the parameters.

        Notes
        -----
        - This method is part of the scikit-learn compatibility, allowing the model to be used in grid 
        search or other tools that require parameter access.
        """
        # needed for scikit-learn compatibility
        return {
            "num_nodes": self.num_nodes,
            "density": self.density,
            "fraction_input": self.fraction_input,
            "fraction_output": self.fraction_output,
            "n_time_out": self.n_time_out,
            "n_time_in": self.n_time_in,
            "n_states_in": self.n_states_in,
            "n_states_out": self.n_states_out,
            "model": self.model,
        }

    def set_params(self, **get_params):
        """
        Sets the hyperparameters of the model, for compatibility with scikit-learn's API.

        This method allows you to update the hyperparameters of the model by passing them as keyword 
        arguments. This is useful for model tuning and other scikit-learn tools such as grid search.

        Parameters
        ----------
        get_params : keyword arguments
            The hyperparameters to be set. Each keyword corresponds to a parameter name, and its value 
            is the new value for that parameter.

        Returns
        -------
        self : object
            The model instance, with updated hyperparameters.

        Notes
        -----
        - This method is part of scikit-learn compatibility, allowing the model to be used in grid search 
        or other tools that require parameter setting.
        """
        # needed for scikit-learn compatibility
        for parameter, value in get_params.items():
            setattr(self, parameter, value)
        return self
