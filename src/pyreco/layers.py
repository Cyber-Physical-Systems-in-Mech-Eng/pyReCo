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
import networkx as nx

from pyreco.utils_networks import (
    gen_ER_graph,
    compute_density,
    get_num_nodes,
    compute_spec_rad,
    remove_nodes_from_graph,
)


# implements the abstract base class
class Layer(ABC):

    @abstractmethod
    def __init__(self):
        self.weights = None  # every layer will have some weights (trainable or not)
        self.name: str = "layer"
        self._is_compiled: bool = False  # flag to check if the overall model is compiled
        pass

    @abstractmethod
    def update_layer_properties(self):
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
        self.name = "input_layer"

        # some properties of the readin layer
        self.fraction_nonzero_entries: (
            float  # fraction of nonzero entries in the input layer
        )

    def remove_nodes(self, nodes: list):
        # removes a node from the input layer (i.e. if a reservoir node needs to be dropped)

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")
        if np.max(nodes) > self.weights.shape[0]:
            raise ValueError(
                "Node index exceeds the number of nodes in the input layer."
            )
        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")
        if not all(isinstance(x, int) for x in nodes):
            raise ValueError("All entries in the node list must be integers.")

        # remove nodes from [n_reservoir_nodes, n_states] matrix
        self.weights = np.delete(self.weights, nodes, axis=0)

        # update the properties of the input layer
        self.update_layer_properties()

    def update_layer_properties(self):

        # updates the properties of the input layer
        self.fraction_nonzero_entries = (
            np.count_nonzero(self.weights) / self.weights.size
        )


class ReadoutLayer(Layer):

    def __init__(self, output_shape, fraction_out=1.0):
        # expects output_shape = (n_timesteps, n_states)
        super().__init__()
        self.output_shape: tuple = output_shape
        self.n_time = output_shape[0]
        self.n_states = output_shape[1]

        self.fraction_out: float = (
            fraction_out  # fraction of connections to the reservoir
        )
        self.name = "readout_layer"

        self.readout_nodes = []  # list of nodes that are linked to output

    def remove_nodes(self, nodes: list):
        # remove nodes from the readout layer, when the user wants to delete specific reservoir nodes

        # # delete entries that equal the nodes to be removed
        # idx_del = np.where(np.isin(self.readout_nodes, nodes))[0]
        # self.readout_nodes = np.delete(self.readout_nodes, idx_del)
        # self.update_layer_properties()
        raise NotImplementedError("This method is not yet implemented.")

    def update_layer_properties(self):

        # updates the properties of the readout layer
        self.fraction_out = len(self.readout_nodes) / self.weights.shape[0]
        pass


class ReservoirLayer(Layer):  # subclass for the specific reservoir layers

    def __init__(
        self,
        nodes,
        density,
        activation,
        leakage_rate,
        fraction_input,
        init_res_sampling,
        seed: int = 42,
    ):
        super().__init__()
        self.nodes: int = nodes  # number of reservoir nodes
        self.density: float = density
        self.spec_rad = None
        self.activation = activation
        self.leakage_rate = leakage_rate
        self.name = "reservoir_layer"
        self.fraction_input = fraction_input
        self.weights = None
        self.input_receiving_nodes = None

        # initial reservoir state (will be set later)
        self.initial_res_states = None
        self.init_res_sampling = init_res_sampling

    def activation_fun(self, x: np.ndarray):
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "tanh":
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        else:
            raise (ValueError(f"unknown activation function {self.activation}!"))

    def set_weights(self, network: np.ndarray):
        # set reservoir network (adjacency matrix) from outside.
        # Updates all related parameters. 
        # MUST be having the same shape as the existing reservoir network.

        if not isinstance(network, np.ndarray):
            raise TypeError("Network must be a numpy array.")
        
        if network.ndim != 2:
            raise ValueError("Network must be a 2D numpy array.")
        
        if network.shape[0] != network.shape[1]:
            raise ValueError("Network must be square (same number of rows and columns).")
        
        # for now, we only accept matrices that have the same shape as the existing reservoir
        if self._is_compiled and network.shape != self.weights.shape:
            raise ValueError(f"Network must have the same shape as the existing reservoir ({self.weights.shape}).")

        # overwrite the existing adjacency matrix
        self.weights = network

        # update related reservoir properties
        self.nodes = get_num_nodes(self.weights)
        self.density = compute_density(self.weights)
        self.spec_rad = compute_spec_rad(self.weights)

    def set_initial_state(self, r_init: np.ndarray):
        # assigns an initial state to each of the reservoir nodes

        if r_init.shape[0] != self.nodes:
            raise (
                ValueError(
                    "initial reservoir state does not match the number of nodes in the reservoir!"
                )
            )
        self.initial_res_states = r_init

    def remove_nodes(self, nodes: list):
        """
        Remove specified nodes from the reservoir network.
        Parameters:
        nodes (list): A list of indices representing the nodes to be removed.
        1. Removes the specified nodes from the adjacency matrix.
        2. Updates the reservoir properties including the number of nodes, density, and spectral radius.
        """
        # remove nodes from the reservoir network

        if not isinstance(nodes, list):
            raise TypeError("Nodes must be provided as a list of indices.")

        if np.max(nodes) > self.nodes:
            raise ValueError("Node index exceeds the number of nodes in the reservoir.")

        if np.min(nodes) < 0:
            raise ValueError("Node index must be positive.")

        # 1. remove nodes from the adjacency matrix
        self.weights = remove_nodes_from_graph(graph=self.weights, nodes=nodes)

        # 2. update reservoir properties
        self.update_layer_properties()

        # 3. update the initial reservoir state
        self.initial_res_states = np.delete(self.initial_res_states, nodes, axis=0)

        # # 4. update the info about input-receiving nodes
        # self.input_receiving_nodes = np.delete(self.input_receiving_nodes, nodes)

    def get_spec_rad(self):
        return self.spec_rad

    def set_spec_rad(self, value):
        if value <= 0:
            raise ValueError("Spectral radius must be positive")
        if self.weights is not None:
            from pyreco.utils_networks import set_spec_rad
            self.weights = set_spec_rad(self.weights, value)
        self.spec_rad = value

    def get_leakage_rate(self):
        return self.leakage_rate

    def set_leakage_rate(self, value):
        if not (0 < value <= 1):
            raise ValueError("Leakage rate must be in (0, 1]")
        self.leakage_rate = value

    def get_activation(self):
        return self.activation

    def set_activation(self, value):
        if value not in ["tanh", "sigmoid"]:
            raise ValueError("Activation must be 'tanh' or 'sigmoid'")
        self.activation = value


class RandomReservoirLayer(ReservoirLayer):
    def __init__(
        self,
        nodes,
        density: float = 0.1,
        activation: str = "tanh",
        leakage_rate: float = 0.5,
        fraction_input: float = 0.8,
        spec_rad: float = 0.9,
        init_res_sampling="random_normal",
        seed=None,
    ):

        # Call the parent class's __init__ method
        super().__init__(
            nodes=nodes,
            density=density,
            activation=activation,
            leakage_rate=leakage_rate,
            fraction_input=fraction_input,
            init_res_sampling=init_res_sampling,
            seed=seed,
        )

        # initialize subclass-specific attributes
        self.seed = seed
        self.spec_rad = spec_rad

        # generate a random ER graph using networkx
        self.weights = gen_ER_graph(
            nodes=nodes,
            density=density,
            spec_rad=self.spec_rad,
            directed=True,
            seed=seed,
        )

    def update_layer_properties(self):
        # Updates the reservoir properties including the number of nodes, density, and spectral radius.
        self.nodes = get_num_nodes(self.weights)
        self.density = compute_density(self.weights)
        self.spec_rad = compute_spec_rad(self.weights)


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
