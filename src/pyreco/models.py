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

    def __init__(self, activation: str = 'tanh', leakage_rate: float = 0.3):
        # basic architectural hyperparameters
        self.activation: str = activation
        self.leakage_rate: float = leakage_rate

        pass

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        # fits the model to the given training data
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        # returns predictions for new input data
        pass

    def compile(self, optimizer: str = 'ridge', metrics: list = ['mse']):
        # sets up things like optimizer and metrics (like in TensorFlow)
        self.optimizer = optimizer
        self.metrics = metrics

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: str|list = ['mse']):
        # let model run predictions for input data X and return the metrics against the ground truth y
        pass

    # @abstractmethod
    def set_params(self, **get_params):
        # needed for scikit-learn compatibility
        for parameter, value in get_params.items():
            setattr(self, parameter, value)
        return self

    # @abstractmethod
    def get_params(self, deep=True):
        # needed for scikit-learn compatibility
        return {
            'activation': self.activation,
            'leakage_rate': self.leakage_rate,
            # 'optimizer': self.optimizer,
            # 'metrics_available': self.metrics_available,
            # 'metrics': self.metrics,
        }


"""
A classical Reservoir Computer (basic vanilla version)
"""


class ReservoirComputer(Model):
    # implements a very classic random reservoir

    def __init__(self,
                 num_nodes: int = 100,
                 density: float = 0.8,
                 activation: str = 'tanh',
                 leakage_rate: float = 0.5,
                 spec_rad: float = 0.9,
                 fraction_input: float = 1.0,
                 fraction_output: float = 1.0,
                 n_time_out=None,
                 n_time_in=None,
                 n_states_in=None,
                 n_states_out=None,
                 metrics: Union[str, list] = 'mean_squared_error',
                 optimizer: str = 'ridge',
                 init_res_sampling='random_normal'
                 ):
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

        # create a RC from a random reservoir. We do not know about the shapes of input and output at this stage
        self.model = RC()

    def fit(self, X: np.ndarray, y: np.ndarray, n_init: int = 1, store_states: bool = False):
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
        if X.dtype == 'O' or y.dtype == 'O':
            raise TypeError("Data type 'object' not supported")

        # obtain the input and output shapes
        n_batch, self.n_time_in, self.n_states_in = X.shape[0], X.shape[1], X.shape[2]
        self.n_time_out, self.n_states_out = y.shape[1], y.shape[2]

        # translate into the shapes requested by the layered model API
        input_shape = (self.n_time_in, self.n_states_in)
        output_shape = (self.n_time_out, self.n_states_out)

        # compose a model from layers. The model was instantiated in the __init__
        self.model.add(InputLayer(input_shape=input_shape))
        self.model.add(RandomReservoirLayer(nodes=self.num_nodes,
                                            density=self.density,
                                            activation=self.activation,
                                            leakage_rate=self.leakage_rate,
                                            spec_rad=self.spec_rad,
                                            fraction_input=self.fraction_input,
                                            init_res_sampling=self.init_res_sampling)
                       )
        self.model.add(ReadoutLayer(output_shape, fraction_out=self.fraction_output))

        # compile the model
        self.model.compile(optimizer=self.optimizer,
                           metrics=self.metrics)

        # fit to training data
        history = self.model.fit(X=X, y=y,
                                 n_init=n_init,
                                 store_states=store_states)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        # returns predictions for given data X
        # expects:
        # - X input data of shape [n_batch, n_time_in, n_states_in]
        # returns:
        # - y_pred predicted data of shape [n_batch, n_time_out, n_states_out]

        # just a dummy here. TODO insert the actual .predict function
        #

        # check for object data types in X
        if X.dtype == 'O':
            raise TypeError("Data type 'object' not supported")

        # check for complex data types
        if np.iscomplexobj(X):
            raise ValueError("Complex data not supported")

        y_pred = self.model.predict(X=X)

        return y_pred

    def evaluate(self, X: np.ndarray, y: np.ndarray, metrics: list = ['mse']):
        # let model run predictions for input data X and return the metrics against the ground truth y
        metric_values = self.model.evaluate(X=X, y=y, metrics=metrics)

        return metric_values

    def get_params(self, deep=True):
        # needed for scikit-learn compatibility
        return {
            'num_nodes': self.num_nodes,
            'density': self.density,
            'fraction_input': self.fraction_input,
            'fraction_output': self.fraction_output,
            'n_time_out': self.n_time_out,
            'n_time_in': self.n_time_in,
            'n_states_in': self.n_states_in,
            'n_states_out': self.n_states_out,
            'model': self.model
        }

    def set_params(self, **get_params):
        # needed for scikit-learn compatibility
        for parameter, value in get_params.items():
            setattr(self, parameter, value)
        return self
