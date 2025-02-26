import cProfile
import pstats
import memory_profiler
import psutil
import GPUtil
import time
from contextlib import contextmanager
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

class RCProfiler:
    """
    Profiler for Reservoir Computing implementations to track CPU, memory, GPU usage and pruning metrics.
    """
    def __init__(self, model, log_file="rc_profile.log"):
        """
        Initialize the profiling class for the reservoir computing model.

        This constructor sets up profiling tools, logging, and metric tracking 
        for monitoring resource usage and pruning statistics.

        Parameters
        ----------
        model : object
            The reservoir computing model to be profiled.
        log_file : str, optional
            Path to the log file where profiling data will be stored (default is "rc_profile.log").

        Attributes
        ----------
        model : object
            The reservoir computing model being profiled.
        log_file : str
            File path for logging profiling information.
        process : psutil.Process
            Process object for retrieving system resource usage.
        profiler : cProfile.Profile
            Profiler instance for measuring execution performance.
        metrics : dict
            Dictionary storing profiling metrics, including:
            - 'memory': List of memory usage measurements (MB).
            - 'cpu': List of CPU usage percentages.
            - 'gpu': List of GPU usage percentages.
            - 'timestamps': List of timestamps for resource monitoring.
            - 'reservoir_states': List of tracked reservoir states during training.
            - 'pruning': Dictionary storing pruning-specific metrics:
                - 'nodes_removed': List of removed nodes during pruning.
                - 'performance_changes': List of performance changes per pruning step.
                - 'network_sizes': List of reservoir network sizes at each step.
                - 'memory_per_node': List of memory usage per removed node.
                - 'time_per_iteration': List of execution times per pruning iteration.
                - 'spectral_radius': List of spectral radii of the reservoir matrix.

        start_time : float
            Timestamp indicating when profiling began.

        Examples
        --------
        >>> model = ReservoirComputingModel()
        >>> profiler = RCProfiler(model)
        """
        self.model = model
        self.log_file = log_file
        self.process = psutil.Process()
        self.profiler = cProfile.Profile()
        self.metrics = {
            'memory': [],
            'cpu': [],
            'gpu': [],
            'timestamps': [],
            'reservoir_states': [],
            # Adding pruning-specific metrics
            'pruning': {
                'nodes_removed': [],
                'performance_changes': [],
                'network_sizes': [],
                'memory_per_node': [],
                'time_per_iteration': [],
                'spectral_radius': []
            }
        }
        self.start_time = time.time()
    
    @contextmanager
    def profile_section(self, section_name):
        """
        Context manager for profiling specific sections of code.

        This context manager measures the execution time and memory usage of a code 
        block and logs the results to a specified log file.

        Parameters
        ----------
        section_name : str
            The name of the section being profiled, used for logging.

        Yields
        ------
        None
            Execution proceeds within the context block, and profiling data is logged 
            upon exit.

        Notes
        -----
        - Captures the execution time of the code inside the `with` block.
        - Measures the memory usage at the beginning and end of execution.
        - Logs profiling results to `self.log_file`, including duration and memory change.

        Examples
        --------
        >>> obj = SomeClass()
        >>> with obj.profile_section("Data Processing"):
        ...     data = process_large_dataset()
        """
        #"""Context manager for profiling specific sections of code."""
        start_time = time.time()
        start_mem = self.process.memory_info().rss / 1024 / 1024  # MB
        
        yield
        
        end_time = time.time()
        end_mem = self.process.memory_info().rss / 1024 / 1024
        
        with open(self.log_file, 'a') as f:
            f.write(f"\n=== {section_name} ===\n")
            f.write(f"Duration: {end_time - start_time:.2f} seconds\n")
            f.write(f"Memory change: {end_mem - start_mem:.2f} MB\n")
    
    def profile_method(self, method_name):
        """
        Decorator for profiling class methods.

        This decorator wraps a method execution inside a profiling section, allowing 
        performance metrics to be collected and analyzed.

        Parameters
        ----------
        method_name : str
            The name of the method being profiled, used for logging and tracking.

        Returns
        -------
        function
            A decorator function that wraps the target method for profiling.

        Notes
        -----
        - This decorator uses `self.profile_section(method_name)` to collect profiling data.
        - The original function's behavior remains unchanged aside from being profiled.
        - The `@wraps(func)` decorator ensures that metadata from the original function 
        (such as its docstring and name) is preserved.

        Examples
        --------
        >>> obj = SomeClass()
        >>> @obj.profile_method("expensive_computation")
        ... def expensive_computation():
        ...     # Some computationally intensive task
        ...     pass
        >>> expensive_computation()
        """
        #"""Decorator for profiling class methods."""
        def decorator(func):
            """
            Decorator function that wraps a method for profiling.

            This function takes another function as input and returns a wrapped version 
            that executes inside a profiling section.

            Parameters
            ----------
            func : function
                The function to be wrapped and profiled.

            Returns
            -------
            function
                A wrapped function that collects profiling data upon execution.

            Notes
            -----
            - Uses `self.profile_section(method_name)` to measure execution performance.
            - Ensures that metadata (such as docstrings and function names) is preserved 
            using `@wraps(func)`.
            - The profiling mechanism is determined by `self.profile_section`.

            Examples
            --------
            >>> obj = SomeClass()
            >>> @obj.profile_method("my_method")
            ... def my_method(x):
            ...     return x ** 2
            >>> my_method(3)
            9
            """
            @wraps(func)
            def wrapper(*args, **kwargs):
                """
                Wrapper function for profiling a method.

                This function executes the decorated method within a profiling section, 
                collecting relevant performance metrics.

                Parameters
                ----------
                *args : tuple
                    Positional arguments passed to the wrapped method.
                **kwargs : dict
                    Keyword arguments passed to the wrapped method.

                Returns
                -------
                object
                    The return value of the wrapped method.

                Notes
                -----
                - The profiling section is managed using `self.profile_section(method_name)`.
                - Performance data such as execution time and resource usage may be collected 
                depending on the implementation of `profile_section`.

                Examples
                --------
                >>> obj = SomeClass()
                >>> @obj.profile_method("compute")
                ... def compute(x):
                ...     return x * 2
                >>> compute(5)
                10
                """
                with self.profile_section(method_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_monitoring(self, interval=1.0):
        """
        Start continuous resource monitoring.

        This method continuously records system resource usage, including memory, 
        CPU, and GPU utilization, at regular intervals until `stop_monitoring()` is called.

        Parameters
        ----------
        interval : float, optional
            The time interval (in seconds) between resource usage recordings. Default is 1.0.

        Notes
        -----
        - Memory usage is recorded in megabytes (MB).
        - CPU usage is recorded as a percentage (%).
        - GPU usage is recorded as the average load percentage across all available GPUs.
        - GPU monitoring requires `GPUtil` to be installed; if unavailable, GPU usage is recorded as 0.
        - Monitoring runs in a blocking loop and should be executed in a separate thread if 
        non-blocking behavior is desired.

        Examples
        --------
        >>> import threading
        >>> monitor_thread = threading.Thread(target=obj.start_monitoring, args=(1.0,))
        >>> monitor_thread.start()
        >>> # Run some operations
        >>> obj.stop_monitoring()
        >>> monitor_thread.join()
        """
        #"""Start continuous resource monitoring."""
        self.monitoring = True
        while self.monitoring:
            timestamp = time.time()
            
            # Memory
            mem_usage = self.process.memory_info().rss / 1024 / 1024
            self.metrics['memory'].append(mem_usage)
            
            # CPU
            cpu_percent = self.process.cpu_percent()
            self.metrics['cpu'].append(cpu_percent)
            
            # GPU if available
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = sum(gpu.load * 100 for gpu in gpus) / len(gpus)
                    self.metrics['gpu'].append(gpu_usage)
            except:
                self.metrics['gpu'].append(0)
                
            self.metrics['timestamps'].append(timestamp)
            
            time.sleep(interval)
    
    def stop_monitoring(self):
        """
        Stop continuous resource monitoring.

        This method sets the `monitoring` flag to `False`, signaling any ongoing 
        monitoring process to stop.

        Notes
        -----
        - This method does not forcibly terminate monitoring but relies on the 
        monitoring loop to check the `monitoring` flag and stop accordingly.

        Examples
        --------
        >>> obj.stop_monitoring()
        """
        #"""Stop continuous resource monitoring."""
        self.monitoring = False
    
    def profile_fit(self, X, y, **kwargs):
        """
        Profile the fit method of the Reservoir Computing (RC) model.

        This method enables the profiler, tracks the execution of the model fitting process, 
        and records relevant reservoir states if available. Profiling is disabled after execution.

        Parameters
        ----------
        X : array-like
            Input features for training the RC model.
        y : array-like
            Target values corresponding to `X`.
        **kwargs : dict, optional
            Additional keyword arguments passed to the model's `fit` method.

        Returns
        -------
        history : object
            The training history returned by the `fit` method of the RC model.

        Notes
        -----
        - The method enables profiling before fitting and disables it afterward.
        - If `history` contains `res_states`, they are stored in `self.metrics['reservoir_states']`.

        Examples
        --------
        >>> history = obj.profile_fit(X_train, y_train, epochs=100)
        >>> print(history)
        """
        #"""Profile the fit method of the RC model."""
        self.profiler.enable()
        
        with self.profile_section("Model Fitting"):
            history = self.model.fit(X, y, **kwargs)
            
            # Track reservoir states if available
            if hasattr(history, 'res_states'):
                self.metrics['reservoir_states'] = history['res_states']
        
        self.profiler.disable()
        return history

    def profile_fit_prune(self, X, y, loss_metric='mse', max_perf_drop=0.1):
        """
        Profile the `fit_prune` method with detailed performance metrics.

        This function enables profiling, logs pruning details, and collects 
        performance-related metrics while pruning the model.

        Parameters
        ----------
        X : array-like
            Input feature data for training.
        y : array-like
            Target output data corresponding to `X`.
        loss_metric : str, optional
            The loss metric used for evaluating pruning effectiveness, by default 'mse'.
        max_perf_drop : float, optional
            Maximum allowed performance drop due to pruning, by default 0.1.

        Returns
        -------
        history : object
            The training history returned by the original `fit_prune` method.

        Notes
        -----
        - Replaces the original `fit_prune` method with an instrumented version.
        - Injects a `track_pruning_step` function into `self.model` for logging.
        - Collects metrics such as:
            - Nodes removed per step
            - Performance changes per pruning iteration
            - Evolution of network size
            - Memory usage per pruning iteration
            - Execution time per pruning step
            - Spectral radius of the reservoir layer (if computable)
        - Logs pruning information using `self.log_pruning_step`.
        - Restores the original `fit_prune` method after execution.
        """
        #"""Profile the fit_prune method with detailed metrics."""
        self.profiler.enable()
        start_time = time.time()
        
        with open(self.log_file, 'a') as f:
            f.write("\n=== Starting Pruning Process ===\n")
        
        # Store initial state
        initial_state = {
            'memory': self.process.memory_info().rss / 1024 / 1024,
            'network_size': self.model.reservoir_layer.weights.shape[0]
        }
        
        # Wrap the original fit_prune method to collect metrics
        original_fit_prune = self.model.fit_prune
        
        @wraps(original_fit_prune)
        def instrumented_fit_prune(*args, **kwargs):
            """
            Instrument the pruning process to track and log performance metrics.

            This function wraps the `fit_prune` method of the model to collect various 
            performance metrics during pruning, including network size, memory usage, 
            performance changes, and spectral radius. The tracked data is stored in 
            `self.metrics['pruning']`, and each pruning step is logged.

            Parameters
            ----------
            *args : tuple
                Positional arguments passed to the original `fit_prune` method.
            **kwargs : dict
                Keyword arguments passed to the original `fit_prune` method.

            Returns
            -------
            history : object
                The training history returned by the original `fit_prune` method.

            Notes
            -----
            - A nested function `track_pruning_step` is injected into `self.model` to track
            pruning steps.
            - Metrics tracked include:
                - Nodes removed
                - Performance changes per pruning step
                - Network size evolution
                - Memory usage per node
                - Execution time per pruning iteration
                - Spectral radius of the reservoir matrix (if computable)
            - The original `fit_prune` method is temporarily replaced and restored after execution.
            - Logs each pruning step via `self.log_pruning_step`.
            """
            iteration = 0
            last_perf = None
            
            def track_pruning_step(node_idx, current_performance):
                """
                Track a single pruning step by logging relevant performance metrics.

                Parameters
                ----------
                node_idx : int
                    Index of the removed node.
                current_performance : float
                    The model's performance after the pruning step.

                Notes
                -----
                - Updates `self.metrics['pruning']` with collected data.
                - Computes spectral radius if applicable.
                - Logs pruning step using `self.log_pruning_step`.
                """
                nonlocal iteration, last_perf
                
                current_time = time.time()
                current_memory = self.process.memory_info().rss / 1024 / 1024
                current_size = self.model.reservoir_layer.weights.shape[0]
                
                # Calculate metrics
                perf_change = (current_performance - last_perf) if last_perf is not None else 0
                last_perf = current_performance
                
                # Get spectral radius
                try:
                    spec_rad = np.max(np.abs(np.linalg.eigvals(
                        self.model.reservoir_layer.weights)))
                except:
                    spec_rad = None
                
                # Store metrics
                self.metrics['pruning']['nodes_removed'].append(node_idx)
                self.metrics['pruning']['performance_changes'].append(perf_change)
                self.metrics['pruning']['network_sizes'].append(current_size)
                self.metrics['pruning']['memory_per_node'].append(
                    current_memory - initial_state['memory'])
                self.metrics['pruning']['time_per_iteration'].append(
                    current_time - start_time)
                if spec_rad is not None:
                    self.metrics['pruning']['spectral_radius'].append(spec_rad)
                
                # Log step
                self.log_pruning_step(iteration, node_idx, current_performance, 
                                    current_size, current_memory)
                
                iteration += 1
            
            # Inject tracking into model
            self.model._track_pruning = track_pruning_step
            
            # Run original fit_prune
            history = original_fit_prune(*args, **kwargs)
            
            # Clean up
            delattr(self.model, '_track_pruning')
            
            return history
        
        # Temporarily replace method
        self.model.fit_prune = instrumented_fit_prune
        
        try:
            history = self.model.fit_prune(X, y, loss_metric=loss_metric, 
                                         max_perf_drop=max_perf_drop)
        finally:
            # Restore original method
            self.model.fit_prune = original_fit_prune
        
        self.profiler.disable()
        return history
    
    def profile_predict(self, X, **kwargs):
        """
        Profile the predict method of the Reservoir Computing (RC) model.

        This method enables profiling, executes the model's prediction function 
        within a profiling section, and then disables profiling. The profiling data 
        can be later retrieved using `get_profile_stats()`.

        Parameters
        ----------
        X : array-like
            Input data for the model prediction.
        **kwargs : dict, optional
            Additional keyword arguments passed to the model's `predict` method.

        Returns
        -------
        array-like
            The predicted outputs from the RC model.

        Notes
        -----
        - Profiling is enabled before executing `self.model.predict(X, **kwargs)`.
        - The profiling section is labeled as `"Model Prediction"`.
        - Profiling is disabled after the prediction completes.
        - The collected profiling data can be retrieved using `get_profile_stats()`.

        Examples
        --------
        >>> predictions = obj.profile_predict(X_test)
        >>> print(predictions.shape)
        """
        #"""Profile the predict method of the RC model."""
        self.profiler.enable()
        
        with self.profile_section("Model Prediction"):
            predictions = self.model.predict(X, **kwargs)
        
        self.profiler.disable()
        return predictions
    
    def log_pruning_step(self, iteration, node_idx, performance, network_size, memory_usage):
        """
        Log information about a pruning step.

        This method records details of a single pruning iteration, including the 
        removed node, the updated network size, performance metrics, and memory usage.
        The log is appended to the file specified by `self.log_file`.

        Parameters
        ----------
        iteration : int
            The current pruning iteration index.
        node_idx : int
            The index of the node that was removed in this pruning step.
        performance : float
            The performance metric after the pruning step.
        network_size : int
            The current size of the network after pruning.
        memory_usage : float
            The measured memory usage (in MB) after pruning.

        Notes
        -----
        - The log is stored in `self.log_file` in an appended format.
        - Performance values are logged with four decimal places.
        - Memory usage values are logged with two decimal places.

        Examples
        --------
        >>> obj.log_pruning_step(3, 42, 0.8912, 1200, 512.34)
        """
        #"""Log information about a pruning step."""
        with open(self.log_file, 'a') as f:
            f.write(f"""
Pruning Step {iteration}:
- Node Removed: {node_idx}
- Network Size: {network_size}
- Performance: {performance:.4f}
- Memory Usage: {memory_usage:.2f} MB
""")
    
    def plot_metrics(self, save_path=None):
        """
        Plot collected performance metrics, including pruning-related metrics if available.

        This method visualizes various system and model performance metrics such as memory 
        usage, CPU and GPU utilization, and pruning effects on network size and performance.
        If pruning-related metrics are available, an extended visualization is created with 
        additional subplots.

        Parameters
        ----------
        save_path : str, optional
            The file path to save the generated plot. If None, the plot is displayed without saving.

        Notes
        -----
        - Metrics are extracted from `self.metrics`, which includes timestamps, memory, CPU, 
        and GPU usage, as well as pruning statistics (if applicable).
        - If pruning metrics are available, a 3x2 grid of subplots is used. Otherwise, 
        a simpler 3-row layout is used.
        - Time-based metrics are plotted relative to the first recorded timestamp.

        Examples
        --------
        >>> obj.plot_metrics()
        >>> obj.plot_metrics(save_path="metrics_plot.png")  # Save the plot to a file
        """
        #"""Plot all collected metrics including pruning metrics if available."""
        if self.metrics['pruning']['nodes_removed']:
            # Create a figure with two rows of subplots
            fig, axes = plt.subplots(3, 2, figsize=(15, 12))
            fig.suptitle('RC Performance and Pruning Analysis')
            
            # Original metrics
            timestamps = np.array(self.metrics['timestamps']) - self.metrics['timestamps'][0]
            
            # Memory usage
            ax = axes[0, 0]
            ax.plot(timestamps, self.metrics['memory'])
            ax.set_ylabel('Memory Usage (MB)')
            ax.set_xlabel('Time (s)')
            ax.grid(True)
            
            # Network size evolution
            ax = axes[0, 1]
            ax.plot(self.metrics['pruning']['network_sizes'])
            ax.set_ylabel('Network Size')
            ax.set_xlabel('Pruning Step')
            ax.grid(True)
            
            # CPU usage
            ax = axes[1, 0]
            ax.plot(timestamps, self.metrics['cpu'])
            ax.set_ylabel('CPU Usage (%)')
            ax.set_xlabel('Time (s)')
            ax.grid(True)
            
            # Performance changes
            ax = axes[1, 1]
            ax.plot(self.metrics['pruning']['performance_changes'])
            ax.set_ylabel('Performance Change')
            ax.set_xlabel('Pruning Step')
            ax.grid(True)
            
            # GPU usage
            ax = axes[2, 0]
            ax.plot(timestamps, self.metrics['gpu'])
            ax.set_ylabel('GPU Usage (%)')
            ax.set_xlabel('Time (s)')
            ax.grid(True)
            
            # Spectral radius if available
            ax = axes[2, 1]
            if self.metrics['pruning']['spectral_radius']:
                ax.plot(self.metrics['pruning']['spectral_radius'])
                ax.set_ylabel('Spectral Radius')
                ax.set_xlabel('Pruning Step')
                ax.grid(True)
            
        else:
            # Original plotting code for non-pruning metrics
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
            
            timestamps = np.array(self.metrics['timestamps']) - self.metrics['timestamps'][0]
            
            ax1.plot(timestamps, self.metrics['memory'])
            ax1.set_ylabel('Memory Usage (MB)')
            ax1.grid(True)
            
            ax2.plot(timestamps, self.metrics['cpu'])
            ax2.set_ylabel('CPU Usage (%)')
            ax2.grid(True)
            
            ax3.plot(timestamps, self.metrics['gpu'])
            ax3.set_ylabel('GPU Usage (%)')
            ax3.set_xlabel('Time (s)')
            ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def get_profile_stats(self):
        """
        Retrieve and log detailed profiling statistics.

        This method processes profiling data collected by `self.profiler`, sorts the 
        statistics by cumulative time, and appends the formatted profiling report to 
        the specified log file. The profiling results are also returned as a 
        `pstats.Stats` object for further inspection.

        Returns
        -------
        pstats.Stats
            A `Stats` object containing profiling statistics.

        Notes
        -----
        - The profiling data is sorted by cumulative time before being logged.
        - The log is appended to `self.log_file` in text format.
        - The `print_stats()` method is used to write the profiling output to the log.

        Examples
        --------
        >>> stats = obj.get_profile_stats()
        >>> stats.print_stats(10)  # Print top 10 functions by cumulative time
        """
        #"""Get detailed profiling statistics."""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        with open(self.log_file, 'a') as f:
            f.write("\n=== Detailed Profile Stats ===\n")
            stats.stream = f
            stats.print_stats()
        
        return stats

