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
        """
        Initializes a layer with default attributes.

        Attributes
        ----------
        weights
            The weights associated with the layer, which can be trainable or non-trainable.
        name
            The name of the layer, default is 'layer'.
        """
        self.weights = None  # every layer will have some weights (trainable or not)
        self.name: str = 'layer'
        pass


class InputLayer(Layer):
    # Shape of the read-in weights is: N x n_states, where N is the number of nodes in the reservoir, and n_states is
    # the state dimension of the input (irrespective if a time series or a vector was put in)
    # the actual read-in layer matrix will be created by mode.compile()!

    def __init__(self, input_shape):
        """
        Initializes an input layer with a given shape.

        Parameters
        ----------
        input_shape
            The shape of the input, expected to be (n_timesteps, n_states).

        Attributes
        ----------
        shape
            The shape of the input.
        n_time
            The number of timesteps in the input.
        n_states
            The number of states (features) in the input.
        name
            The name of the layer, set to 'input_layer'.
        """
        # input shape is (n_timesteps, n_states)
        super().__init__()
        self.shape = input_shape
        self.n_time = input_shape[0]
        self.n_states = input_shape[1]
        self.name = 'input_layer'


class ReadoutLayer(Layer):

    def __init__(self, output_shape, fraction_out=1.0):
        """
        Initializes a readout layer with a specified output shape and connection fraction.

        Parameters
        ----------
        output_shape
            The shape of the output, expected to be (n_timesteps, n_states).
        fraction_out, optional
            The fraction of connections to the reservoir, by default 1.0.

        Attributes
        ----------
        output_shape
            The shape of the output.
        n_time
            The number of timesteps in the output.
        n_states
            The number of states (features) in the output.
        fraction_out
            The fraction of connections to the reservoir.
        name
            The name of the layer, set to 'readout_layer'.
        readout_nodes
            A list of nodes that are linked to the output.
        """
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
        """
        Initializes a reservoir layer with the specified parameters.

        Parameters
        ----------
        nodes
            The number of nodes in the reservoir.
        density
            The density of connections within the reservoir.
        activation
            The activation function used in the reservoir.
        leakage_rate
            The leakage rate for the reservoir update.
        fraction_input
            The fraction of input connections to the reservoir.
        init_res_sampling
            The method used to initialize the reservoir states.
        seed : int, optional
            The random seed for initialization, by default 42.

        Attributes
        ----------
        nodes
            The number of nodes in the reservoir.
        density
            The density of connections within the reservoir.
        spec_rad
            The spectral radius of the reservoir (to be set later).
        activation
            The activation function used in the reservoir.
        leakage_rate
            The leakage rate for the reservoir update.
        name
            The name of the layer, set to 'reservoir_layer'.
        fraction_input
            The fraction of input connections to the reservoir.
        weights
            The weight matrix of the reservoir (to be set later).
        initial_res_states
            The initial reservoir states (to be set later).
        init_res_sampling
            The method used to initialize the reservoir states.
        """
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
        """
        Applies the selected activation function to the input array.

        Parameters
        ----------
        x : np.ndarray
            The input array on which the activation function is applied.

        Returns
        -------
        np.ndarray
            The transformed array after applying the activation function.

        Raises
        ------
        ValueError
            If the specified activation function is not supported.

        Notes
        -----
        - Supports the following activation functions:
        - 'sigmoid': \( \sigma(x) = \frac{1}{1 + e^{-x}} \)
        - 'tanh': \( \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)
        """
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation == 'tanh':
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            raise (ValueError(f'unknown activation function {self.activation}!'))

    def set_weights(self, network: np.ndarray):
        """
        Sets the reservoir network weights and updates related parameters.

        Parameters
        ----------
        network : np.ndarray
            The adjacency matrix representing the reservoir network.

        Attributes Updated
        ------------------
        weights
            Stores the provided network weights.
        nodes
            Updates the number of nodes in the reservoir based on the network.
        density
            Updates the density of the network.
        spec_rad
            Updates the spectral radius of the network.
        """
        # set reservoir network from outside.
        # Updates all related parameters

        self.weights = network
        self.nodes = get_num_nodes(network)
        self.density = compute_density(network)
        self.spec_rad = compute_spec_rad(network)

    def set_initial_state(self, r_init: np.ndarray):
        """
        Assigns an initial state to each of the reservoir nodes.

        Parameters
        ----------
        r_init : np.ndarray
            The initial state vector for the reservoir nodes.

        Raises
        ------
        ValueError
            If the shape of `r_init` does not match the number of nodes in the reservoir.

        Attributes Updated
        ------------------
        initial_res_states
            Stores the assigned initial state for the reservoir nodes.
        """
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
        """
        Initializes a reservoir layer with a randomly generated Erdős-Rényi (ER) network.

        Parameters
        ----------
        nodes
            The number of nodes in the reservoir.
        density : float, optional
            The density of connections within the reservoir, by default 0.1.
        activation : str, optional
            The activation function used in the reservoir, by default 'tanh'.
        leakage_rate : float, optional
            The leakage rate for the reservoir update, by default 0.5.
        fraction_input : float, optional
            The fraction of input connections to the reservoir, by default 0.8.
        spec_rad : float, optional
            The spectral radius of the reservoir, by default 0.9.
        init_res_sampling : str, optional
            The method used to initialize the reservoir states, by default 'random_normal'.
        seed : int, optional
            The random seed for initialization, by default None.

        Attributes
        ----------
        seed
            The random seed for initializing the reservoir.
        spec_rad
            The spectral radius of the reservoir.
        weights
            The adjacency matrix representing the reservoir network, generated using
            an Erdős-Rényi (ER) graph.

        Notes
        -----
        - Calls the parent class's `__init__` method to initialize shared attributes.
        - Uses `gen_ER_graph` to generate a directed ER network with the specified parameters.
        """
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


