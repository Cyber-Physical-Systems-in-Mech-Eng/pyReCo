"""
Minimal working example: what happens to a NARMA10 reservoir when all of its edges are pruned, one edge
after the other, until every node is isolated?

A reservoir of 50 nodes with an initial density of 50% (about 1200 edges) is trained on NARMA10. Then the
``EdgePruner`` removes one edge per iteration. In every iteration it tries out *all* remaining edges
(``candidate_fraction=1.0``): each one is removed tentatively, the readout is retrained, and the
validation error is computed. The edge whose removal hurts least is removed for good. This is repeated
until no edge is left, which takes about 1200 iterations and some 700 000 retrained readouts.

The script answers three questions:

1. How does the performance (NRMSE on the training, validation and test data) change over the pruning
   iterations, and how many edges can be removed before it deteriorates?
2. How do the properties of the network change (density, spectral radius, degree, clustering, number of
   isolated nodes, connected components)?
3. How does the pruning change where the nodes work on their activation curve (mean slope of the
   activation, fraction of time in saturation)? With the random input bias switched on (``--input-bias``)
   this is the quantity that structural measures like the degree cannot see, see
   ``pyreco.edge_analyzer.operating_point_statistics``.

Because the run takes an hour or more, the result of every iteration is stored on disk while the pruning
runs (``<out>/records.json``, one record per iteration) and the run can be interrupted and resumed with
the same command: the network is rebuilt from the stored order of the removed edges. The pruner keeps no
checkpoints itself, so the script hooks into it and keeps only a compact record per iteration instead of
the candidate details of the whole history (which would grow to gigabytes).

Usage::

    python mwe_edge_pruning_to_isolation.py                   # run (or resume) and plot
    python mwe_edge_pruning_to_isolation.py --max-iterations 20   # only 20 more iterations (try it out)
    python mwe_edge_pruning_to_isolation.py --plot-only       # plots of the stored records
    python mwe_edge_pruning_to_isolation.py --input-bias 0.5 --seed 1   # another arm of the study

Output (in ``edge_pruning_to_isolation_bias<b>_seed<s>/``): ``records.json``, ``initial_weights.npy``,
``config.json``, ``pruning_performance.png``, ``pruning_network_properties.png``, ``pruning_steps.png``.
"""

import os

# one BLAS thread per process: the candidates are evaluated in parallel by worker processes, and the
# matrices are small (50 x 50), so more threads only compete with each other
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import argparse
import contextlib
import copy
import io
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components

from pyreco.custom_models import RC
from pyreco.datasets import load
from pyreco.edge_analyzer import operating_point_statistics
from pyreco.edge_pruning import EdgePruner
from pyreco.layers import InputLayer, RandomReservoirLayer, ReadoutLayer
from pyreco.optimizers import RidgeSK
from pyreco.utils_networks import extract_clustering_coefficient

# ------------------------------------------------------------------------------ settings
NODES = 50
DENSITY = 0.5  # initial density of the reservoir (fraction of the N (N - 1) possible edges)
SPEC_RAD = 0.9
LEAKAGE_RATE = 0.9
INPUT_SCALING = 0.3
WEIGHT_DISTRIBUTION = "uniform"  # signed weights; the all-positive default limits NARMA10 accuracy
RIDGE_ALPHA = 1e-6
N_IN = 30  # input window; the target belongs to the last input step (warm-up N_IN - 1)
N_TRAIN, N_VAL = 400, 250  # windows used in the pruning (every candidate retrains the readout on them)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0, help="seed of the reservoir, the read-in weights, the data")
    parser.add_argument(
        "--input-bias", type=float, default=0.0,
        help="gain of the random per-node input bias (0 = off). It sets the operating point of the nodes and "
             "can decide which edges matter: run the study with and without it and with several seeds.",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="prune at most this many edges in this call")
    parser.add_argument("--serial", action="store_true", help="evaluate the candidates in one process")
    parser.add_argument("--fresh", action="store_true", help="ignore stored records and start from scratch")
    parser.add_argument("--plot-only", action="store_true", help="only plot the stored records")
    return parser.parse_args()


# ----------------------------------------------------------------------------- the study
class Study:
    """Data, model, records and the files of one pruning run."""

    def __init__(self, seed, input_bias):
        self.config = dict(
            nodes=NODES, density=DENSITY, spec_rad=SPEC_RAD, leakage_rate=LEAKAGE_RATE, input_scaling=INPUT_SCALING,
            weight_distribution=WEIGHT_DISTRIBUTION, ridge_alpha=RIDGE_ALPHA, n_in=N_IN, n_train=N_TRAIN,
            n_val=N_VAL, seed=seed, input_bias=input_bias,
        )
        self.directory = Path(f"edge_pruning_to_isolation_bias{input_bias:g}_seed{seed}")
        self.records = []

        # NARMA10 in windows of N_IN inputs; the subsets are spread evenly over the windows
        x_train, y_train, x_val, y_val, self.x_test, self.y_test, _ = load(
            "narma10", n_samples=6000, n_in=N_IN, n_out=1, seed=seed + 1, val_fraction=0.2
        )
        pick = lambda n, size: np.linspace(0, size - 1, n).astype(int)
        i, j = pick(N_TRAIN, len(x_train)), pick(N_VAL, len(x_val))
        self.x_train, self.y_train, self.x_val, self.y_val = x_train[i], y_train[i], x_val[j], y_val[j]

    def build_model(self):
        """The unpruned, fitted reservoir computer."""
        model = RC(seed=self.config["seed"])
        model.add(InputLayer(input_shape=self.x_train.shape[1:]))
        model.add(RandomReservoirLayer(
            nodes=NODES, density=DENSITY, spec_rad=SPEC_RAD, leakage_rate=LEAKAGE_RATE,
            weight_distribution=WEIGHT_DISTRIBUTION, seed=self.config["seed"],
        ))
        model.add(ReadoutLayer(output_shape=self.y_train.shape[1:]))
        model.compile(
            optimizer=RidgeSK(alpha=RIDGE_ALPHA), discard_transients=N_IN - 1, metrics=["nrmse"],
            input_scaling=INPUT_SCALING, input_bias=self.config["input_bias"],
        )
        return model

    # ------------------------------------------------------------------------ records
    def snapshot(self, model, iteration, winner=None, candidate_scores=None):
        """All numbers that are stored for the state of the network after ``iteration`` removed edges."""
        A = model.reservoir_layer.weights
        B = (A != 0).astype(float)
        n_edges = int(B.sum())
        _, labels = connected_components(B, directed=False)
        component_sizes = np.bincount(labels)
        operating_point = operating_point_statistics(model, self.x_val)["node"]
        score = lambda x, y: float(model.evaluate(x, y, metrics="nrmse")[0])
        record = dict(
            iteration=iteration,
            edges=n_edges,
            train_nrmse=score(self.x_train, self.y_train),
            val_nrmse=score(self.x_val, self.y_val),
            test_nrmse=score(self.x_test, self.y_test),
            density=n_edges / (NODES * (NODES - 1)),
            spectral_radius=float(model.reservoir_layer.spec_rad),
            mean_degree=n_edges / NODES,  # mean in-degree = mean out-degree
            clustering=float(extract_clustering_coefficient(A)),
            isolated_nodes=int(np.sum(B.sum(0) + B.sum(1) == 0)),
            components=int(len(component_sizes)),
            largest_component=int(component_sizes.max()),
            mean_gain=float(operating_point["mean_gain"].mean()),
            saturation_fraction=float(operating_point["saturation_fraction"].mean()),
            winner=None if winner is None else [int(winner[0]), int(winner[1])],
            winner_weight=None if winner is None else float(winner[2]),
        )
        if candidate_scores is not None:
            record.update(
                n_candidates=len(candidate_scores), candidate_min=float(np.min(candidate_scores)),
                candidate_median=float(np.median(candidate_scores)), candidate_max=float(np.max(candidate_scores)),
            )
        return record

    def save(self):
        """Write the records; replacing the file in one step keeps the last complete state if the run is killed."""
        temporary = self.directory / "records.json.tmp"
        temporary.write_text(json.dumps(self.records))
        os.replace(temporary, self.directory / "records.json")

    def weights_after(self, n_removed, initial_weights):
        """The network after ``n_removed`` pruning iterations, rebuilt from the order of the removed edges."""
        A = initial_weights.copy()
        for record in self.records[1 : n_removed + 1]:
            A[tuple(record["winner"])] = 0.0
        return A


class CheckpointedEdgePruner(EdgePruner):
    """
    ``EdgePruner`` that stores a compact record after every iteration and drops the heavy history.

    It overrides the (private) hook ``_history_update_after_pruning``, which the pruner calls after it has
    removed an edge for good. The pruner itself evaluates the candidates, so the pruning is unchanged.
    """

    def __init__(self, study, **kwargs):
        super().__init__(**kwargs)
        self.study = study
        self._last_time = time.time()

    def _history_update_after_pruning(self, model, graph, idx_prune, candidates, removed_nodes):
        super()._history_update_after_pruning(model, graph, idx_prune, candidates, removed_nodes)

        study = self.study
        entry = self.history.pop(self._iter_count)  # keep the scores of the candidates, drop the rest
        scores = [candidate["score"] for candidate in entry["candidates"].values()]
        u, v = entry["winner"]
        previous_weights = study.weights_before_step
        record = study.snapshot(
            model, iteration=study.records[-1]["iteration"] + 1, winner=(u, v, previous_weights[u, v]),
            candidate_scores=scores,
        )
        record["seconds"] = time.time() - self._last_time
        self._last_time = time.time()
        study.records.append(record)
        study.weights_before_step = model.reservoir_layer.weights.copy()
        study.save()

        if record["iteration"] % 25 == 0 or record["edges"] == 0:
            print(
                f"iteration {record['iteration']:5d}: {record['edges']:5d} edges, val NRMSE {record['val_nrmse']:.3f}, "
                f"test NRMSE {record['test_nrmse']:.3f}, {record['seconds']:.1f} s/iteration",
                file=sys.__stdout__, flush=True,
            )


# ------------------------------------------------------------------------------ running
def run_pruning(study, args):
    """Run (or resume) the pruning, storing the records on the way."""
    study.directory.mkdir(exist_ok=True)
    config_file, weights_file, records_file = (
        study.directory / "config.json", study.directory / "initial_weights.npy", study.directory / "records.json"
    )

    model = study.build_model()
    resuming = records_file.exists() and not args.fresh
    if resuming:
        if json.loads(config_file.read_text()) != study.config:
            raise SystemExit(f"{study.directory} was made with other settings; use --fresh to start again.")
        study.records = json.loads(records_file.read_text())
        initial_weights = np.load(weights_file)
        assert np.array_equal(initial_weights, model.reservoir_layer.weights), "the rebuilt network differs"
        model.reservoir_layer.set_weights(study.weights_after(len(study.records) - 1, initial_weights))
        model.fit(study.x_train, study.y_train)
        print(f"resuming after {study.records[-1]['iteration']} iterations, {study.records[-1]['edges']} edges left")
    else:
        model.fit(study.x_train, study.y_train)
        np.save(weights_file, model.reservoir_layer.weights)
        config_file.write_text(json.dumps(study.config, indent=2))
        study.records = [study.snapshot(model, iteration=0)]
        study.save()
        print(
            f"start: {study.records[0]['edges']} edges (density {study.records[0]['density']:.3f}), "
            f"val NRMSE {study.records[0]['val_nrmse']:.3f}, test NRMSE {study.records[0]['test_nrmse']:.3f}"
        )

    edges = study.records[-1]["edges"]
    if edges == 0:
        print("all edges are pruned already")
        return
    study.weights_before_step = model.reservoir_layer.weights.copy()

    # prune until no edge is left (min_num_edges=0), or max_iterations edges in this call
    min_num_edges = 0 if args.max_iterations is None else max(0, edges - args.max_iterations)
    pruner = CheckpointedEdgePruner(
        study,
        candidate_fraction=1.0,  # try out all edges in every iteration
        performance_criterion="nrmse",
        stopping_criterion=["min_edges"],
        min_num_edges=min_num_edges,
        metrics=["nrmse"],
        return_best_model=False,  # the last model, not the best one
        parallel=not args.serial,
        seed=study.config["seed"],
    )
    start = time.time()
    # the pruner reports every candidate and shows a progress bar for the parallel evaluation: silence both
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        pruner.prune(copy.deepcopy(model), (study.x_train, study.y_train), (study.x_val, study.y_val))
    print(f"finished {len(study.records) - 1} iterations in total, {time.time() - start:.0f} s in this call")


# ------------------------------------------------------------------------------- plotting
def plot_results(study):
    records = study.records
    column = lambda name: np.array([r[name] if r.get(name) is not None else np.nan for r in records], dtype=float)
    iteration, edges = column("iteration"), column("edges")
    n_nodes = NODES
    path = study.directory

    def finish(ax, title, ylabel):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("pruning iteration (edges removed)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    # ---- performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, zoom in zip(axes, (False, True)):
        for name, color in (("train", "C0"), ("val", "C1"), ("test", "C2")):
            ax.plot(iteration, column(f"{name}_nrmse"), color=color, label=f"{name} NRMSE")
        ax.fill_between(iteration[1:], column("candidate_min")[1:], column("candidate_max")[1:],
                        color="C1", alpha=0.15, label="all candidates (val): min to max")
        ax.plot(iteration[1:], column("candidate_median")[1:], color="C1", ls=":", label="median candidate (val)")
        ax.axhline(1.0, color="grey", lw=0.8, ls="--", label="predict the mean (1.0)")
        if zoom:
            reference = column("val_nrmse")[0]
            limit = np.nanmax(np.where(column("val_nrmse") <= 1.5 * reference, iteration, 0)) * 1.5 + 5
            ax.set_xlim(0, min(limit, iteration[-1]))
            ax.set_ylim(0, 1.3 * np.nanmax(column("test_nrmse")[iteration <= ax.get_xlim()[1]]))
            finish(ax, "zoom on the first iterations", "NRMSE")
        else:
            finish(ax, "NARMA10: error while the edges are removed", "NRMSE")
            ax.legend(fontsize=8, loc="lower right")
        top = ax.secondary_xaxis("top", functions=(lambda i: edges[0] - i, lambda e: edges[0] - e))
        top.set_xlabel("edges left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path / "pruning_performance.png", dpi=150)
    plt.close(fig)

    # ---- network properties
    panels = [
        ("density", "density", "|E| / (N (N - 1))"),
        ("spectral_radius", "spectral radius", r"$\rho(A)$"),
        ("mean_degree", "mean in- (= out-) degree", "edges per node"),
        ("clustering", "average clustering coefficient", "C"),
        ("isolated_nodes", "isolated nodes (of %d)" % n_nodes, "nodes"),
        ("largest_component", "largest weakly connected component", "nodes"),
        ("mean_gain", "mean slope of the activation", r"$\langle g'(z) \rangle$"),
        ("saturation_fraction", "fraction of time in saturation", "fraction"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (name, title, ylabel) in zip(axes.flat, panels):
        ax.plot(iteration, column(name), color="C3" if name in ("mean_gain", "saturation_fraction") else "C0")
        finish(ax, title, ylabel)
    axes.flat[4].plot(iteration, column("components"), color="C2", label="components")
    axes.flat[4].legend(fontsize=8)
    fig.suptitle(f"network properties along the pruning (input bias {study.config['input_bias']:g}, "
                 f"seed {study.config['seed']})")
    fig.tight_layout()
    fig.savefig(path / "pruning_network_properties.png", dpi=150)
    plt.close(fig)

    # ---- what the pruner removes
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    weight = column("winner_weight")
    axes[0].plot(iteration, np.abs(weight), ".", ms=3)
    finish(axes[0], "weight of the removed edge", r"$|A_{uv}|$")
    spread = column("candidate_max") - column("candidate_min")
    axes[1].semilogy(iteration, np.maximum(spread, 1e-12), ".", ms=3)
    finish(axes[1], "spread of the candidates (max - min val NRMSE)", "NRMSE difference")
    axes[2].plot(iteration, column("seconds"), ".", ms=3)
    finish(axes[2], "time per iteration", "seconds")
    fig.tight_layout()
    fig.savefig(path / "pruning_steps.png", dpi=150)
    plt.close(fig)


def summarize(study):
    r = study.records
    start = r[0]
    last = r[-1]
    print(f"\n{len(r) - 1} iterations; edges {start['edges']} -> {last['edges']}, "
          f"isolated nodes {start['isolated_nodes']} -> {last['isolated_nodes']} of {NODES}")
    print(f"{'':28s} {'val':>8s} {'test':>8s}")
    print(f"{'unpruned':28s} {start['val_nrmse']:8.3f} {start['test_nrmse']:8.3f}")
    for tolerance in (1.1, 1.5, 2.0):
        # the sparsest network that is still within the tolerance in every iteration up to it
        sparsest = next((x for x in r if x["val_nrmse"] > tolerance * start["val_nrmse"]), None)
        keep = r[r.index(sparsest) - 1] if sparsest else last
        print(f"{'val within +%d%% up to' % round((tolerance - 1) * 100):28s} {keep['val_nrmse']:8.3f} "
              f"{keep['test_nrmse']:8.3f}   iteration {keep['iteration']}, {keep['edges']} edges "
              f"(density {keep['density']:.3f})")
    print(f"{'all edges pruned':28s} {last['val_nrmse']:8.3f} {last['test_nrmse']:8.3f}")


if __name__ == "__main__":
    args = parse_arguments()
    study = Study(args.seed, args.input_bias)
    if not args.plot_only:
        run_pruning(study, args)
    else:
        study.records = json.loads((study.directory / "records.json").read_text())
    summarize(study)
    plot_results(study)
    print(f"\nresults in {study.directory.resolve()}")
