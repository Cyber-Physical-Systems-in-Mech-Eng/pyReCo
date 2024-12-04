"""
We will have an abstract Layer class, from which the following layers inherit:

- InputLayer
- ReservoirLayer
    - RandomReservoir
    - RecurrenceReservoir
    - EvolvedReservoir
- ReadoutLayer

"""

from abc import ABC, abstractmethod
import numpy as np

from .utils_networks import gen_ER_graph, compute_density, get_num_nodes, compute_spec_rad


# implements the abstract base class
class Layer(ABC):

    @abstractmethod
    def __init__(self):
        self.weights = None  # every layer will have some weights (trainable or not)
        self.name: str = 'layer'
        pass


class InputLayer(Layer):
    # Shape of the read-in weights is: N x n_states, where N is the number of nodes in the reservoir, and n_states is
    # the state dimension of the input (irrespective if a time series or a vector was put in)
    # the actual read-in layer matrix will be created by mode.compile()!

    def __init__(self, input_shape):
        # input shape is (n_timesteps, n_states)
        super().__init__()
        self.shape = input_shape
        self.n_time = input_shape[0]
        self.n_states = input_shape[1]
        self.name = 'input_layer'


class ReadoutLayer(Layer):

    def __init__(self, output_shape, fraction_out=1.0):
        # expects output_shape = (n_timesteps, n_states)
        super().__init__()
        self.output_shape: tuple = output_shape
        self.n_time = output_shape[0]
        self.n_states = output_shape[1]

        self.fraction_out: float = fraction_out  # fraction of connections to the reservoir
        self.name = 'readout_layer'

        self.readout_nodes = []  # list of nodes that are linked to output


class ReservoirLayer(Layer):  # subclass for the specific reservoir layers

    def __init__(self, nodes, density, activation, leakage_rate, fraction_input,
                 init_res_sampling, seed: int = 42):
        super().__init__()
        self.nodes: int = nodes
        self.density: float = density
        self.spec_rad = None
        self.activation = activation
        self.leakage_rate = leakage_rate
        self.name = 'reservoir_layer'
        self.fraction_input = fraction_input
        self.weights = None

        # initial reservoir state (will be set later)
        self.initial_res_states = None
        self.init_res_sampling = init_res_sampling

    def activation_fun(self, x: np.ndarray):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            raise (ValueError(f'unknown activation function {self.activation}!'))

    def set_weights(self, network: np.ndarray):
        # set reservoir network from outside.
        # Updates all related parameters

        self.weights = network
        self.nodes = get_num_nodes(network)
        self.density = compute_density(network)
        self.spec_rad = compute_spec_rad(network)

    def set_initial_state(self, r_init: np.ndarray):
        # assigns an initial state to each of the reservoir nodes

        if r_init.shape[0] != self.nodes:
            raise (ValueError('initial reservoir state does not match the number of nodes in the reservoir!'))
        self.initial_res_states = r_init


class RandomReservoirLayer(ReservoirLayer):
    def __init__(self, nodes,
                 density: float = 0.1,
                 activation: str = 'tanh',
                 leakage_rate: float = 0.5,
                 fraction_input: float = 0.8,
                 spec_rad: float = 0.9,
                 init_res_sampling='random_normal',
                 seed=None):

        # Call the parent class's __init__ method
        super().__init__(nodes=nodes,
                         density=density,
                         activation=activation,
                         leakage_rate=leakage_rate,
                         fraction_input=fraction_input,
                         init_res_sampling=init_res_sampling,
                         seed=seed)

        # initialize subclass-specific attributes
        self.seed = seed
        self.spec_rad = spec_rad

        # generate a random ER graph using networkx
        self.weights = gen_ER_graph(nodes=nodes, density=density, spec_rad=self.spec_rad, directed=True, seed=seed)


# class ReccurrenceLayer(ReservoirLayer):
#     # To Do: accept a random seed
#     def __init__(self, nodes, density, activation: str = 'tanh', leakage_rate: float = 0.2):
#         # Call the parent class's __init__ method
#         super().__init__(nodes, density, activation, leakage_rate)
#
#         # Initialize subclass-specific attributes
#         # https://pyts.readthedocs.io/en/stable/generated/pyts.image.RecurrencePlot.html#pyts.image.RecurrencePlot
#         # https://tocsy.pik-potsdam.de/pyunicorn.php
#


