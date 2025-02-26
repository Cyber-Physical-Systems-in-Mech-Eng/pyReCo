
import sys
import os
import platform
import numpy as np
import time
from matplotlib import pyplot as plt
import copy
import threading

# Platform-specific path setup
if platform.system() == 'Linux':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    src_path = os.path.join(project_root, 'src')
    sys.path.insert(0, src_path)

# Import RC implementations and components
import pyreco.custom_models_1 as rc_old
import pyreco.custom_models as rc_new
from pyreco.layers import InputLayer, ReadoutLayer, RandomReservoirLayer
from pyreco.utils_data import sequence_to_sequence
from rc_profiler import RCProfiler

def setup_model(model_class, input_shape, output_shape):
    """
    Set up a Reservoir Computing (RC) model with consistent parameters.

    This function initializes an RC model with predefined layers, including an 
    input layer, a randomly initialized reservoir layer, and a readout layer. 
    The model is then compiled with a ridge regression optimizer.

    Parameters
    ----------
    model_class : class
        The class of the reservoir computing model to be instantiated.
    input_shape : tuple
        The shape of the input data.
    output_shape : tuple
        The shape of the output data.

    Returns
    -------
    model : object
        The initialized and compiled RC model.

    Notes
    -----
    - The reservoir layer consists of 100 nodes with a 10% connectivity density.
    - The activation function used in the reservoir is 'sigmoid'.
    - A leakage rate of 0.1 is applied to the reservoir layer.
    - The readout layer retains 99% of the nodes from the reservoir.
    - The model is compiled with a ridge regression optimizer and tracks mean squared error (MSE).

    Examples
    --------
    >>> model = setup_model(MyRCModel, input_shape=(10,), output_shape=(1,))
    >>> model.summary()
    """
    #"""Setup RC model with consistent parameters."""
    model = model_class.RC()
    model.add(InputLayer(input_shape=input_shape))
    model.add(RandomReservoirLayer(
        nodes=100,
        density=0.1,
        activation='sigmoid',
        leakage_rate=0.1,
        fraction_input=0.5
    ))
    model.add(ReadoutLayer(output_shape, fraction_out=0.99))
    model.compile(optimizer='ridge', metrics=['mse'])
    return model

def run_training_benchmark(model, X, y, n_init=1):
    """
    Run a training benchmark for a Reservoir Computing (RC) model and collect performance metrics.

    This function profiles the training process of an RC model, measuring execution time, 
    memory usage, CPU utilization, and mean squared error (MSE). It employs a profiler 
    to monitor system resource consumption throughout training.

    Parameters
    ----------
    model : object
        The RC model to be trained.
    X : array-like
        The input data for training.
    y : array-like
        The target output data.
    n_init : int, optional
        The number of times the model is initialized before training (default is 1).

    Returns
    -------
    metrics : dict
        A dictionary containing the following performance metrics:
        - 'time' : float
            Total training duration in seconds.
        - 'memory' : float
            Change in memory usage (MB) during training.
        - 'mse' : float
            Mean squared error of the trained model on the given dataset.
        - 'cpu' : float
            Average CPU utilization percentage during training.

    Notes
    -----
    - The function starts a separate monitoring thread to track system resource usage.
    - The training process is executed within a profiling context.
    - The function ensures proper cleanup by stopping monitoring before returning.

    Examples
    --------
    >>> model = setup_model(MyRCModel, input_shape=(10,), output_shape=(1,))
    >>> metrics = run_training_benchmark(model, X_train, y_train)
    >>> print(metrics)
    """
    #"""Run training benchmark and collect metrics."""
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
            mse = model.evaluate(X, y, metrics=['mse'])[0]
        
        metrics = {
            'time': time.time() - start_time,
            'memory': profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            'mse': mse,
            'cpu': np.mean(profiler.metrics['cpu'])
        }
        
    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)
    
    return metrics

def run_prediction_benchmark(model, X, y):
    """
    Run a prediction benchmark for a Reservoir Computing (RC) model and collect performance metrics.

    This function profiles the inference process of an RC model, measuring execution time, 
    memory usage, CPU utilization, and mean squared error (MSE). It employs a profiler 
    to monitor system resource consumption throughout prediction.

    Parameters
    ----------
    model : object
        The trained RC model used for predictions.
    X : array-like
        The input data for making predictions.
    y : array-like
        The true target values for evaluation.

    Returns
    -------
    metrics : dict
        A dictionary containing the following performance metrics:
        - 'time' : float
            Total prediction duration in seconds.
        - 'memory' : float
            Change in memory usage (MB) during inference.
        - 'mse' : float
            Mean squared error between predicted and true values.
        - 'cpu' : float
            Average CPU utilization percentage during inference.

    Notes
    -----
    - The function starts a separate monitoring thread to track system resource usage.
    - The prediction process is executed within a profiling context.
    - The function ensures proper cleanup by stopping monitoring before returning.

    Examples
    --------
    >>> model = setup_model(MyRCModel, input_shape=(10,), output_shape=(1,))
    >>> metrics = run_prediction_benchmark(model, X_test, y_test)
    >>> print(metrics)
    """
    #"""Run prediction benchmark and collect metrics."""
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
            mse = model.evaluate(X, y, metrics=['mse'])[0]
        
        metrics = {
            'time': time.time() - start_time,
            'memory': profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            'mse': mse,
            'cpu': np.mean(profiler.metrics['cpu'])
        }
        
    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)
    
    return metrics

def run_pruning_benchmark(model, X, y):
    """
    Run a pruning benchmark for a Reservoir Computing (RC) model and collect performance metrics.

    This function profiles the pruning process of an RC model, measuring execution time, 
    memory usage, CPU utilization, and mean squared error (MSE). It also tracks pruning-specific 
    metrics, such as the number of removed nodes and the MSE progression during pruning.

    Parameters
    ----------
    model : object
        The RC model to be pruned. Must support `fit_prune()` for pruning.
    X : array-like
        The input training data.
    y : array-like
        The target values corresponding to `X`.

    Returns
    -------
    metrics : dict
        A dictionary containing the following performance and pruning-related metrics:
        - 'time' : float
            Total pruning duration in seconds.
        - 'memory' : float
            Change in memory usage (MB) during pruning.
        - 'mse' : float
            Final mean squared error after pruning.
        - 'cpu' : float
            Average CPU utilization percentage during pruning.
        - 'history' : dict
            The history of pruning iterations and associated performance metrics.
        - 'nodes_removed' : int or None
            The total number of nodes removed during pruning, if tracked.
        - 'mse_progression' : list or None
            MSE progression during the pruning process, if available.

    Notes
    -----
    - The function starts a separate monitoring thread to track system resource usage.
    - The pruning process is executed within a profiling context.
    - The function supports both old and new implementations of the RC model's `fit_prune()` method.
    - The function ensures proper cleanup by stopping monitoring before returning.

    Examples
    --------
    >>> model = setup_model(MyRCModel, input_shape=(10,), output_shape=(1,))
    >>> metrics = run_pruning_benchmark(model, X_train, y_train)
    >>> print(metrics)
    """
    #"""Run pruning benchmark and collect metrics."""
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
                history = model.fit_prune(X, y, loss_metric='mse', max_perf_drop=0.1)
            else:  # New implementation
                history, best_model = model.fit_prune(X, y, loss_metric='mse', max_perf_drop=0.1)
        
        # Get final MSE
        mse = model.evaluate(X, y, metrics=['mse'])[0]
        
        metrics = {
            'time': time.time() - start_time,
            'memory': profiler.process.memory_info().rss / 1024 / 1024 - start_mem,
            'mse': mse,
            'cpu': np.mean(profiler.metrics['cpu']),
            'history': history,
            'nodes_removed': history['num_nodes'][0] - history['num_nodes'][-1] if 'num_nodes' in history else None,
            'mse_progression': history['pruned_nodes_scores'] if 'pruned_nodes_scores' in history else None
        }
        
    finally:
        profiler.stop_monitoring()
        monitor.join(timeout=1.0)
    
    return metrics

def run_all_benchmarks(num_runs=5):
    """
    Run multiple benchmark tests on both old and new RC models and collect performance metrics.

    This function evaluates different aspects of reservoir computing models, including:
    - Training with a single initialization
    - Training with multiple initializations
    - Prediction performance
    - Pruning efficiency

    The benchmarking is repeated multiple times (`num_runs`) for statistical robustness. 
    The results are collected and returned as a dictionary.

    Parameters
    ----------
    num_runs : int, optional
        The number of times to repeat each benchmark (default is 5).

    Returns
    -------
    results : dict
        A dictionary containing benchmark results for both old (`rc_old`) and new (`rc_new`) 
        model implementations. The structure is:
        - 'single_init' : dict
            Results for training with a single initialization.
        - 'multi_init' : dict
            Results for training with multiple initializations.
        - 'prediction' : dict
            Results for prediction performance.
        - 'pruning' : dict
            Results for pruning efficiency.
        
        Each category contains:
        - 'old' : list
            List of results from the old RC model.
        - 'new' : list
            List of results from the new RC model.

    Notes
    -----
    - The function generates synthetic test data using `sequence_to_sequence()`.
    - Models are initialized and trained separately for each benchmark run.
    - Pruning is performed only after training the models.
    - The results provide insights into computational efficiency and prediction accuracy.

    Examples
    --------
    >>> results = run_all_benchmarks(num_runs=3)
    >>> print(results['single_init']['new'])
    """
    #"""Run all benchmarks and collect results."""
    # Generate test data
    X_train, X_test, y_train, y_test = sequence_to_sequence(
        name='sincos2', n_batch=10, n_states=1, n_time=200)
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = (y_train.shape[1], y_train.shape[2])
    
    results = {
        'single_init': {'old': [], 'new': []},
        'multi_init': {'old': [], 'new': []},
        'prediction': {'old': [], 'new': []},
        'pruning': {'old': [], 'new': []}
    }
    
    # Run benchmarks
    for run in range(num_runs):
        print(f"\nRun {run + 1}/{num_runs}")
        
        for model_type in ['old', 'new']:
            model_class = rc_old if model_type == 'old' else rc_new
            
            # Single init training
            model = setup_model(model_class, input_shape, output_shape)
            results['single_init'][model_type].append(
                run_training_benchmark(model, X_train, y_train, n_init=1))
            
            # Multi init training
            model = setup_model(model_class, input_shape, output_shape)
            results['multi_init'][model_type].append(
                run_training_benchmark(model, X_train, y_train, n_init=5))
            
            # Prediction
            model = setup_model(model_class, input_shape, output_shape)
            model.fit(X_train, y_train)  # Train first
            results['prediction'][model_type].append(
                run_prediction_benchmark(model, X_test, y_test))
            
            # Pruning
            model = setup_model(model_class, input_shape, output_shape)
            results['pruning'][model_type].append(
                run_pruning_benchmark(model, X_train, y_train))
    
    return results

def plot_benchmark_results(results, save_path_prefix=None):
    """
    Generate and display box plots comparing benchmark results for old and new RC models.

    This function visualizes the performance of different reservoir computing models 
    across various benchmark categories, including training, prediction, and pruning. 
    It creates separate plots for execution time, memory usage, CPU utilization, and 
    mean squared error (MSE). 

    Parameters
    ----------
    results : dict
        A dictionary containing benchmark results from `run_all_benchmarks()`. 
        The expected structure is:
        - 'single_init' : dict
            Results for training with a single initialization.
        - 'multi_init' : dict
            Results for training with multiple initializations.
        - 'prediction' : dict
            Results for prediction performance.
        - 'pruning' : dict
            Results for pruning efficiency.
        
        Each category contains:
        - 'old' : list of dict
            List of results from the old RC model.
        - 'new' : list of dict
            List of results from the new RC model.
        
        Each result dictionary must contain the keys: 'time', 'memory', 'cpu', and 'mse'.

    save_path_prefix : str, optional
        If provided, the generated plots will be saved as PNG files with this prefix. 
        Each operation type (e.g., 'single_init', 'pruning') will be appended to the filename.

    Notes
    -----
    - Uses box plots to visualize distributions for both old and new models.
    - The percentage improvement from the old model to the new model is annotated in each plot.
    - Each operation type is visualized in a 2x2 grid layout.

    Examples
    --------
    >>> results = run_all_benchmarks(num_runs=3)
    >>> plot_benchmark_results(results, save_path_prefix="benchmark_plot")
    """
    #"""Create plots for each operation type."""
    operations = {
        'single_init': 'Training (Single Init)',
        'multi_init': 'Training (Multiple Init)',
        'prediction': 'Prediction',
        'pruning': 'Pruning'
    }
    
    metrics = ['time', 'memory', 'cpu', 'mse']
    metric_labels = ['Time (s)', 'Memory (MB)', 'CPU Usage (%)', 'MSE']
    
    for op_name, op_title in operations.items():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{op_title} Performance Comparison')
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            
            old_data = [r[metric] for r in results[op_name]['old']]
            new_data = [r[metric] for r in results[op_name]['new']]
            
            ax.boxplot([old_data, new_data], labels=['Old', 'New'])
            ax.set_title(label)
            ax.grid(True)
            
            # Calculate improvement
            improvement = ((np.mean(old_data) - np.mean(new_data)) / np.mean(old_data) * 100)
            ax.text(0.5, 0.95, f'Improvement: {improvement:+.1f}%', 
                   transform=ax.transAxes, ha='center')
        
        plt.tight_layout()
        
        if save_path_prefix:
            plt.savefig(f'{save_path_prefix}_{op_name}.png')
        plt.show()

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    
    print("Starting benchmarks...")
    results = run_all_benchmarks(num_runs=5)
    
    print("\nGenerating plots...")
    plot_benchmark_results(results, save_path_prefix='rc_benchmark')
    
    print("Benchmarking completed!")
