import sys
import os
import platform
import numpy as np
import time
from matplotlib import pyplot as plt
import copy
import threading

# Platform-specific path setup
if platform.system() == "Linux":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, "src")
    sys.path.insert(0, src_path)

# Import RC implementations and components
import pyreco.custom_models_1 as rc_old
import pyreco.custom_models as rc_new
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from examples.deprecated.rc_profiler import RCProfiler


def setup_model(model_class, input_shape, output_shape):
    """Setup RC model with consistent parameters."""
    model = model_class.RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(
        RandomReservoirLayer(
            nodes=100,
            density=0.1,
            activation="sigmoid",
            leakage_rate=0.1,
            fraction_input=0.5,
        )
    )
    model.add(ReadoutLayer(output_shape, fraction_out=0.99))
    model.compile(optimizer="ridge", metrics=["mse"])
    return model


def run_training_benchmark(model, X, y, n_init=1):
    """Run training benchmark and collect metrics."""
    profiler = RCProfiler(model)

    # Start monitoring
    monitor = threading.Thread(target=profiler.start_monitoring)
    monitor.daemon = True
    monitor.start()

    try:
        start_time = time.time()
        start_mem = profiler.process.memory_info().rss / 1024 / 1024

        # Train model
        with profiler.profile_section("Model Training"):
            history = model.fit(X, y, n_init=n_init)
            mse = model.evaluate(X, y, metrics=["mse"])[0]

        metrics = {
            "time": time.time() - start_time,
            "memory": profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            "mse": mse,
            "cpu": np.mean(profiler.metrics["cpu"]),
        }

    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)

    return metrics


def run_prediction_benchmark(model, X, y):
    """Run prediction benchmark and collect metrics."""
    profiler = RCProfiler(model)

    # Start monitoring
    monitor = threading.Thread(target=profiler.start_monitoring)
    monitor.daemon = True
    monitor.start()

    try:
        start_time = time.time()
        start_mem = profiler.process.memory_info().rss / 1024 / 1024

        # Make predictions
        with profiler.profile_section("Model Prediction"):
            y_pred = model.predict(X)
            mse = model.evaluate(X, y, metrics=["mse"])[0]

        metrics = {
            "time": time.time() - start_time,
            "memory": profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            "mse": mse,
            "cpu": np.mean(profiler.metrics["cpu"]),
        }

    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)

    return metrics


def run_pruning_benchmark(model, X, y):
    """Run pruning benchmark and collect metrics."""
    profiler = RCProfiler(model)

    # Start monitoring
    monitor = threading.Thread(target=profiler.start_monitoring)
    monitor.daemon = True
    monitor.start()

    try:
        start_time = time.time()
        start_mem = profiler.process.memory_info().rss / 1024 / 1024

        # Run pruning
        with profiler.profile_section("Model Pruning"):
            if isinstance(model, rc_old.RC):
                history = model.fit_prune(X, y, loss_metric="mse", max_perf_drop=0.1)
            else:  # New implementation
                history, best_model = model.fit_prune(
                    X, y, loss_metric="mse", max_perf_drop=0.1
                )

        # Get final MSE
        mse = model.evaluate(X, y, metrics=["mse"])[0]

        metrics = {
            "time": time.time() - start_time,
            "memory": profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            "mse": mse,
            "cpu": np.mean(profiler.metrics["cpu"]),
            "history": history,
            "nodes_removed": (
                history["num_nodes"][0] - history["num_nodes"][-1]
                if "num_nodes" in history
                else None
            ),
            "mse_progression": (
                history["pruned_nodes_scores"]
                if "pruned_nodes_scores" in history
                else None
            ),
        }

    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)

    return metrics


def run_all_benchmarks(num_runs=5):
    """Run all benchmarks and collect results."""
    # Generate test data
    X_train, X_test, y_train, y_test = sequence_to_sequence(
        name="sincos2", n_batch=10, n_states=1, n_time=200
    )

    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])

    results = {
        "single_init": {"old": [], "new": []},
        "multi_init": {"old": [], "new": []},
        "prediction": {"old": [], "new": []},
        "pruning": {"old": [], "new": []},
    }

    # Run benchmarks
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")

        for model_type in ["old", "new"]:
            model_class = rc_old if model_type == "old" else rc_new

            # Single init training
            model = setup_model(model_class, input_shape, output_shape)
            results["single_init"][model_type].append(
                run_training_benchmark(model, X_train, y_train, n_init=1)
            )

            # Multi init training
            model = setup_model(model_class, input_shape, output_shape)
            results["multi_init"][model_type].append(
                run_training_benchmark(model, X_train, y_train, n_init=5)
            )

            # Prediction
            model = setup_model(model_class, input_shape, output_shape)
            model.fit(X_train, y_train)  # Train first
            results["prediction"][model_type].append(
                run_prediction_benchmark(model, X_test, y_test)
            )

            # Pruning
            model = setup_model(model_class, input_shape, output_shape)
            results["pruning"][model_type].append(
                run_pruning_benchmark(model, X_train, y_train)
            )

    return results


def plot_benchmark_results(results, save_path_prefix=None):
    """Create plots for each operation type."""
    operations = {
        "single_init": "Training (Single Init)",
        "multi_init": "Training (Multiple Init)",
        "prediction": "Prediction",
        "pruning": "Pruning",
    }

    metrics = ["time", "memory", "cpu", "mse"]
    metric_labels = ["Time (s)", "Memory (MB)", "CPU Usage (%)", "MSE"]

    for op_name, op_title in operations.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"{op_title} Performance Comparison")

        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]

            old_data = [r[metric] for r in results[op_name]["old"]]
            new_data = [r[metric] for r in results[op_name]["new"]]

            ax.boxplot([old_data, new_data], labels=["Old", "New"])
            ax.set_title(label)
            ax.grid(True)

            # Calculate improvement
            improvement = (
                (np.mean(old_data) - np.mean(new_data)) / np.mean(old_data) * 100
            )
            ax.text(
                0.5,
                0.95,
                f"Improvement: {improvement:+.1f}%",
                transform=ax.transAxes,
                ha="center",
            )

        plt.tight_layout()

        if save_path_prefix:
            plt.savefig(f"{save_path_prefix}_{op_name}.png")
        plt.show()


if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility

    print("Starting benchmarks...")
    results = run_all_benchmarks(num_runs=5)

    print("\nGenerating plots...")
    plot_benchmark_results(results, save_path_prefix="rc_benchmark")

    print("Benchmarking completed!")
