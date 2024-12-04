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
        """Context manager for profiling specific sections of code."""
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
        """Decorator for profiling class methods."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile_section(method_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def start_monitoring(self, interval=1.0):
        """Start continuous resource monitoring."""
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
        """Stop continuous resource monitoring."""
        self.monitoring = False
    
    def profile_fit(self, X, y, **kwargs):
        """Profile the fit method of the RC model."""
        self.profiler.enable()
        
        with self.profile_section("Model Fitting"):
            history = self.model.fit(X, y, **kwargs)
            
            # Track reservoir states if available
            if hasattr(history, 'res_states'):
                self.metrics['reservoir_states'] = history['res_states']
        
        self.profiler.disable()
        return history

    def profile_fit_prune(self, X, y, loss_metric='mse', max_perf_drop=0.1):
        """Profile the fit_prune method with detailed metrics."""
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
            iteration = 0
            last_perf = None
            
            def track_pruning_step(node_idx, current_performance):
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
        """Profile the predict method of the RC model."""
        self.profiler.enable()
        
        with self.profile_section("Model Prediction"):
            predictions = self.model.predict(X, **kwargs)
        
        self.profiler.disable()
        return predictions
    
    def log_pruning_step(self, iteration, node_idx, performance, network_size, memory_usage):
        """Log information about a pruning step."""
        with open(self.log_file, 'a') as f:
            f.write(f"""
Pruning Step {iteration}:
- Node Removed: {node_idx}
- Network Size: {network_size}
- Performance: {performance:.4f}
- Memory Usage: {memory_usage:.2f} MB
""")
    
    def plot_metrics(self, save_path=None):
        """Plot all collected metrics including pruning metrics if available."""
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
        """Get detailed profiling statistics."""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        with open(self.log_file, 'a') as f:
            f.write("\n=== Detailed Profile Stats ===\n")
            stats.stream = f
            stats.print_stats()
        
        return stats

