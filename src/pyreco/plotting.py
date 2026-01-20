"""
Some plotting capabilities
"""

import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

from .metrics import r2

def r2_scatter(y_true: np.ndarray, y_pred: np.ndarray, state_idx: int|tuple= None,
               title:str = None, xlabel:str=None, ylabel:str = None):
    # plots predictions against ground truth values as scatter plot
    # lets the user choose the output state to plot (if there are multiple states). If not provided, all states will
    # be plotted.
    # collapses the data along the time dimension to show time-dependent targets

    # expects arguments to be of 3D shape: [n_batch, n_timesteps, n_states]

    if y_true.ndim != y_pred.ndim:
        raise(ValueError('Inconsistent shapes! y_true and y_pred need to have the same shape!'))

    if (state_idx is not None) and (np.max(state_idx) >= y_true.ndim):
        raise(ValueError(f'Please select a valid state index, maximum being {y_true.ndim} for the given data'))

    # select the states to show in case supplied by the user
    if state_idx is not None:
        y_true = y_true[:, :, state_idx]
        y_pred = y_pred[:, :, state_idx]

    # now flatten into a vector (along states and along time if necessary)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    min_val = np.min(y_true)  # np.min([np.min(y_true), np.min(y_pred)])
    max_val = np.max(y_true)  # np.max([np.max(y_true), np.max(y_pred)])

    fig = plt.figure()
    plt.plot([min_val, max_val], [min_val, max_val],
             linestyle='solid',
             color='gray',
             label='perfect model')
    plt.plot(y_true, y_pred,
             linestyle='none',
             marker='.',
             markersize=5,
             color='black',
             label=rf'model predictions ($R^2=${r2(y_true, y_pred):.2f})')
    if xlabel is None:
        plt.xlabel('ground truth')
    else:
        plt.xlabel(xlabel)
    if ylabel is None:
        plt.ylabel('predictions')
    else:
        plt.ylabel(ylabel)
    plt.legend()
    if title is None:
        plt.title(rf'')
    else:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_reservoir_network_circle(G_Net, W_inp, W_out, n_inputs, n_outputs, 
                                       seed=None, save_path=None, Node_colors=None,
                                       Edge_Weights=None, has_feedback=False):
    
    """
    Draws a circular layout diagram for ESN/RC architecture.
    Handles ANY valid shape of W_inp and W_out safely.
    Allows custom colors.
    """

    CWinp = Node_colors['CWinp']
    CWres_inp = Node_colors['CWres_inp']
    CWres_out = Node_colors['CWres_out']
    CWres_both = Node_colors['CWres_both']
    CWres_internal = Node_colors['CWres_internal']
    CWout = Node_colors['CWout']
    Winp = Node_colors['Winp']
    Wout = Node_colors['Wout']
    CWres = Node_colors['CWres']
    # --------------------------------------------------
    # Setup
    # --------------------------------------------------
    if seed is not None:
        np.random.seed(seed)

    G = nx.DiGraph()

    # --------------------------------------------------
    # Create Nodes
    # --------------------------------------------------

    # Input nodes
    input_nodes = [f'I{i}' for i in range(n_inputs)]
    for node in input_nodes:
        G.add_node(node, type='input')

    # Reservoir nodes
    n_reservoir = G_Net.shape[0]
    reservoir_nodes = [f'R{i}' for i in range(n_reservoir)]
    for node in reservoir_nodes:
        G.add_node(node, type='reservoir')

    # Output nodes
    output_nodes = [f'O{i}' for i in range(n_outputs)]
    for node in output_nodes:
        G.add_node(node, type='output')

    # --------------------------------------------------
    # Add Input → Reservoir Edges
    # --------------------------------------------------
    input_connections = set()

    def add_single_input_vector(vec):
        """Handles W_inp as (N,) or (N,1) — single input only."""
        for j in range(len(vec)):
            w = vec[j]
            G.add_edge('I0', f'R{j}', weight=w, type='input')
            input_connections.add(f'R{j}')

    if W_inp.ndim == 1:
        add_single_input_vector(W_inp)

    elif W_inp.ndim == 2 and W_inp.shape[1] == 1:
        add_single_input_vector(W_inp[:, 0])

    else:
        # Standard multiple inputs: shape = (n_inputs, n_reservoir)
        for i in range(W_inp.shape[0]):
            for j in range(W_inp.shape[1]):
                w = W_inp[i, j]
                if abs(w) > 0.01:
                    G.add_edge(f'I{i}', f'R{j}', weight=w, type='input')
                    input_connections.add(f'R{j}')

    # --------------------------------------------------
    # Add Reservoir → Reservoir Edges
    # --------------------------------------------------
    for i in range(n_reservoir):
        for j in range(n_reservoir):
            w = G_Net[i, j]
            if abs(w) > 0.01:
                G.add_edge(f'R{i}', f'R{j}', weight=w, type='recurrent')

    # --------------------------------------------------
    # Add Reservoir → Output Edges
    # --------------------------------------------------
    output_connections = set()

    if W_out.ndim == 1:
        for i in range(len(W_out)):
            if abs(W_out[i]) > 0.01:
                G.add_edge(f'R{i}', 'O0', weight=W_out[i], type='output')
                output_connections.add(f'R{i}')

    else:
        for i in range(W_out.shape[0]):
            for j in range(W_out.shape[1]):
                w = W_out[i, j]
                if abs(w) > 0.01:
                    G.add_edge(f'R{i}', f'O{j}', weight=w, type='output')
                    output_connections.add(f'R{i}')

    # --------------------------------------------------
    # Reservoir Node Types (colors + labels)
    # --------------------------------------------------
    reservoir_colors = {}
    reservoir_labels = {}

    for node in reservoir_nodes:
        has_in = node in input_connections
        has_out = node in output_connections

        if has_in and has_out:
            reservoir_colors[node] = CWres_both
            reservoir_labels[node] = 'I+O'
        elif has_in:
            reservoir_colors[node] = CWres_inp
            reservoir_labels[node] = 'Inp'
        elif has_out:
            reservoir_colors[node] = CWres_out
            reservoir_labels[node] = 'Out'
        else:
            reservoir_colors[node] = CWres_internal
            reservoir_labels[node] = 'Int'

    # --------------------------------------------------
    # Node Positions
    # --------------------------------------------------
    pos = {}

    # Reservoir nodes inside a circle
    R = 2
    for i, node in enumerate(reservoir_nodes):
        theta = np.random.uniform(0, 2 * np.pi)
        r = R * np.sqrt(np.random.uniform(0, 1))
        pos[node] = (r * np.cos(theta), r * np.sin(theta))

    # Input column aligned vertically
    spacing = 1.5
    mid = (n_inputs - 1) * spacing / 2
    for k, node in enumerate(input_nodes):
        pos[node] = (-3, k * spacing - mid)

    # Output column aligned vertically
    mid = (n_outputs - 1) * spacing / 2
    for k, node in enumerate(output_nodes):
        pos[node] = (3, k * spacing - mid)

    # --------------------------------------------------
    # Plotting
    # --------------------------------------------------
    plt.figure(figsize=(14, 10))

    # Reservoir circle
    plt.gca().add_patch(
        plt.Circle((0, 0), R, color='Black', fill=False, linestyle='-', linewidth=1, alpha=0.5)
    )

    # Node colors list
    node_colors = []
    for node in G.nodes():
        t = G.nodes[node]['type']
        if t == 'input':
            node_colors.append(Winp)
        elif t == 'output':
            node_colors.append(Wout)
        else:
            node_colors.append(reservoir_colors[node])

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=150,
        node_color=node_colors
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        {k: v for k, v in pos.items() if k.startswith('R')},
        labels=reservoir_labels,
        font_size=6, font_weight='bold',
        )

    # Edge colors
    edge_colors = []
    for _, _, d in G.edges(data=True):
        if d['type'] == 'input':
            edge_colors.append(CWinp)
        elif d['type'] == 'recurrent':
            edge_colors.append(CWres)
        else:
            edge_colors.append(CWout)

    # Draw edges on TOP of nodes
    nx.draw_networkx_edges(
        G, pos, arrows=True, arrowsize=5,
        edge_color=edge_colors,
        width=Edge_Weights
    )

    # --------------------------------------------------
    # Add Feedback Loop Arrow (if has_feedback is True)
    # --------------------------------------------------
    if has_feedback and n_outputs > 0 and n_inputs > 0:        
        # Get positions of top nodes
        output_top_y = max([pos[node][1] for node in output_nodes])
        input_top_y = max([pos[node][1] for node in input_nodes])

        # Coordinates for the arrow path
        start_x, start_y = 3, output_top_y + 0.3
        mid_top_y = 2.6

        # Draw the arrow as 3 connected segments with ONE arrowhead at the end

        # 1. Vertical line up from output
        plt.plot([start_x, start_x], [start_y, mid_top_y],
                 color='purple', linewidth=2)

        # 2. Horizontal line across the top
        plt.plot([start_x, -3], [mid_top_y, mid_top_y],
                 color='purple', linewidth=2)

        # 3. Vertical line down to input WITH ARROWHEAD
        plt.arrow(-3, mid_top_y, 
                  0, (input_top_y + 0.3) - mid_top_y,
                  head_width=0.15, head_length=0.2,
                  fc='purple', ec='purple', linewidth=2,
                  length_includes_head=True)

        # Add feedback label
        plt.text(0, mid_top_y + 0.1, "Feedback Loop", 
                 fontsize=10, ha='center', color='purple', weight='bold')

    # Layer labels
    plt.text(-2.5, -1.5, "Input Layer", fontsize=14, ha='center', color='red')
    plt.text(0, 2.1, "Reservoir Layer", fontsize=14, ha='center', color='blue')
    plt.text(2.5, -1.5, "Output Layer", fontsize=14, ha='center', color='green')
    plt.text(-3, -2.25, "W_inp", fontsize=14, ha='center', color='red')
    plt.text(0, -2.3, "W_res", fontsize=14, ha='center', color='blue')
    plt.text(3, -2.25, "W_out", fontsize=14, ha='center', color='green')

    # Legend fixed (inside figure)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', markerfacecolor='orange', color='w', markersize=10, label='Input + Output'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='lightcoral', color='w', markersize=10, label='Input Only'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='lightgreen', color='w', markersize=10, label='Output Only'),
        plt.Line2D([0], [0], marker='o', markerfacecolor='lightblue', color='w', markersize=10, label='Internal Only'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)

    plt.title("Python Reservoir Computing Architecture - PyReCo Architecture" + 
              (" with Feedback" if has_feedback else ""))
    plt.axis('off')
    plt.tight_layout()
    # Save figure if filename is provided
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
