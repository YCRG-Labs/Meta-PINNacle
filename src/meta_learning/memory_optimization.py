"""Memory optimization utilities for large-scale meta-learning.

This module implements memory optimization techniques including gradient checkpointing,
mixed precision training, and efficient task distribution for meta-learning models.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.checkpoint import checkpoint
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
import gc
import psutil
import time

from .task import Task, TaskData, TaskBatch
from ..utils.distributed_utils import get_rank, is_main_process


class GradientCheckpointing:
    """Gradient checkpointing for memory-efficient meta-learning.
    
    This class implements gradient checkpointing to reduce memory usage during
    meta-learning by trading computation for memory.
    """
    
    def __init__(self, enabled: bool = True, preserve_rng_state: bool = True):
        """Initialize gradient checkpointing.
        
        Args:
            enabled: Whether to enable gradient checkpointing
            preserve_rng_state: Whether to preserve RNG state during checkpointing
        """
        self.enabled = enabled
        self.preserve_rng_state = preserve_rng_state
        self.checkpointed_functions = {}
    
    def checkpoint_function(self, func: Callable, *args, **kwargs):
        """Apply gradient checkpointing to a function.
        
        Args:
            func: Function to checkpoint
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function output with gradient checkpointing
        """
        if not self.enabled or not torch.is_grad_enabled():
            return func(*args, **kwargs)
        
        return checkpoint(func, *args, preserve_rng_state=self.preserve_rng_state, **kwargs)
    
    def checkpoint_sequential(self, functions: List[Callable], segments: int, *args):
        """Apply gradient checkpointing to a sequence of functions.
        
        Args:
            functions: List of functions to checkpoint
            segments: Number of segments for checkpointing
            *args: Input arguments
            
        Returns:
            Output of sequential function application
        """
        if not self.enabled or not torch.is_grad_enabled():
            # Apply functions sequentially without checkpointing
            output = args
            for func in functions:
                output = func(*output) if isinstance(output, tuple) else func(output)
            return output
        
        # Apply checkpointing
        return torch.utils.checkpoint.checkpoint_sequential(
            functions, segments, *args, preserve_rng_state=self.preserve_rng_state
        )
    
    def wrap_model_forward(self, model: nn.Module) -> nn.Module:
        """Wrap model forward pass with gradient checkpointing.
        
        Args:
            model: Model to wrap
            
        Returns:
            Model with checkpointed forward pass
        """
        if not self.enabled:
            return model
        
        original_forward = model.forward
        
        def checkpointed_forward(*args, **kwargs):
            return self.checkpoint_function(original_forward, *args, **kwargs)
        
        model.forward = checkpointed_forward
        return model


class MixedPrecisionTraining:
    """Mixed precision training for meta-learning models.
    
    This class implements mixed precision training using automatic mixed precision (AMP)
    to reduce memory usage and potentially speed up training.
    """
    
    def __init__(self, enabled: bool = True, init_scale: float = 2.**16,
                 growth_factor: float = 2.0, backoff_factor: float = 0.5,
                 growth_interval: int = 2000):
        """Initialize mixed precision training.
        
        Args:
            enabled: Whether to enable mixed precision
            init_scale: Initial loss scaling factor
            growth_factor: Factor to multiply scale by during growth
            backoff_factor: Factor to multiply scale by during backoff
            growth_interval: Number of steps between scale growth
        """
        self.enabled = enabled and torch.cuda.is_available()
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval
            )
        else:
            self.scaler = None
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for mixed precision training.
        
        Args:
            loss: Loss tensor
            
        Returns:
            Scaled loss tensor
        """
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def backward(self, loss: torch.Tensor):
        """Perform backward pass with mixed precision.
        
        Args:
            loss: Loss tensor
        """
        if self.enabled and self.scaler is not None:
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
        else:
            loss.backward()
    
    def step(self, optimizer: torch.optim.Optimizer):
        """Perform optimizer step with mixed precision.
        
        Args:
            optimizer: Optimizer to step
        """
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
    
    def autocast_context(self):
        """Get autocast context for mixed precision.
        
        Returns:
            Autocast context manager
        """
        if self.enabled:
            return autocast()
        else:
            # Return a dummy context manager
            from contextlib import nullcontext
            return nullcontext()


class TaskDistributionOptimizer:
    """Optimizer for efficient task distribution and load balancing.
    
    This class implements strategies for distributing tasks across GPUs
    to minimize memory usage and maximize throughput.
    """
    
    def __init__(self, world_size: int = 1, memory_limit_gb: float = 10.0):
        """Initialize task distribution optimizer.
        
        Args:
            world_size: Number of processes/GPUs
            memory_limit_gb: Memory limit per GPU in GB
        """
        self.world_size = world_size
        self.memory_limit_bytes = memory_limit_gb * 1024**3
        self.task_memory_estimates = {}
        self.gpu_memory_usage = [0.0] * world_size
    
    def estimate_task_memory(self, task: Task, model_size_mb: float) -> float:
        """Estimate memory usage for a task.
        
        Args:
            task: Task to estimate memory for
            model_size_mb: Model size in MB
            
        Returns:
            Estimated memory usage in bytes
        """
        # Base model memory
        base_memory = model_size_mb * 1024**2
        
        # Data memory (inputs, outputs, collocation points)
        data_memory = 0
        if task.support_data:
            data_memory += task.support_data.inputs.numel() * 4  # float32
            data_memory += task.support_data.outputs.numel() * 4
            if task.support_data.collocation_points is not None:
                data_memory += task.support_data.collocation_points.numel() * 4
        
        if task.query_data:
            data_memory += task.query_data.inputs.numel() * 4
            data_memory += task.query_data.outputs.numel() * 4
            if task.query_data.collocation_points is not None:
                data_memory += task.query_data.collocation_points.numel() * 4
        
        # Gradient memory (approximately 2x model size for gradients + optimizer states)
        gradient_memory = base_memory * 2
        
        # Activation memory (estimated based on network depth and data size)
        activation_memory = data_memory * len(getattr(task, 'network_layers', [64, 64, 64])) * 0.1
        
        total_memory = base_memory + data_memory + gradient_memory + activation_memory
        
        # Cache estimate
        task_key = f"{task.problem_type}_{len(task.support_data) if task.support_data else 0}"
        self.task_memory_estimates[task_key] = total_memory
        
        return total_memory
    
    def distribute_tasks_by_memory(self, tasks: List[Task], model_size_mb: float) -> List[List[Task]]:
        """Distribute tasks across GPUs based on memory constraints.
        
        Args:
            tasks: List of tasks to distribute
            model_size_mb: Model size in MB
            
        Returns:
            List of task lists for each GPU
        """
        if self.world_size == 1:
            return [tasks]
        
        # Estimate memory for each task
        task_memories = []
        for task in tasks:
            memory = self.estimate_task_memory(task, model_size_mb)
            task_memories.append((task, memory))
        
        # Sort tasks by memory usage (largest first for better packing)
        task_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute tasks using first-fit decreasing algorithm
        gpu_tasks = [[] for _ in range(self.world_size)]
        gpu_memory = [0.0] * self.world_size
        
        for task, memory in task_memories:
            # Find GPU with minimum memory usage that can fit this task
            best_gpu = -1
            min_memory = float('inf')
            
            for gpu_idx in range(self.world_size):
                if gpu_memory[gpu_idx] + memory <= self.memory_limit_bytes:
                    if gpu_memory[gpu_idx] < min_memory:
                        min_memory = gpu_memory[gpu_idx]
                        best_gpu = gpu_idx
            
            if best_gpu != -1:
                gpu_tasks[best_gpu].append(task)
                gpu_memory[best_gpu] += memory
            else:
                # If no GPU can fit the task, assign to GPU with minimum memory
                min_gpu = np.argmin(gpu_memory)
                gpu_tasks[min_gpu].append(task)
                gpu_memory[min_gpu] += memory
                
                if is_main_process():
                    print(f"Warning: Task with {memory/1024**3:.2f}GB memory assigned to GPU {min_gpu} "
                          f"(current usage: {gpu_memory[min_gpu]/1024**3:.2f}GB)")
        
        # Update internal memory tracking
        self.gpu_memory_usage = gpu_memory
        
        if is_main_process():
            print("Task distribution summary:")
            for gpu_idx in range(self.world_size):
                print(f"  GPU {gpu_idx}: {len(gpu_tasks[gpu_idx])} tasks, "
                      f"{gpu_memory[gpu_idx]/1024**3:.2f}GB estimated memory")
        
        return gpu_tasks
    
    def get_optimal_batch_size(self, task: Task, model_size_mb: float, 
                              available_memory_gb: float) -> int:
        """Get optimal batch size for a task given memory constraints.
        
        Args:
            task: Task to optimize batch size for
            model_size_mb: Model size in MB
            available_memory_gb: Available memory in GB
            
        Returns:
            Optimal batch size
        """
        available_memory_bytes = available_memory_gb * 1024**3
        
        # Estimate memory per sample
        if task.support_data and len(task.support_data) > 0:
            sample_memory = self.estimate_task_memory(task, model_size_mb) / len(task.support_data)
        else:
            # Default estimate
            sample_memory = model_size_mb * 1024**2 * 0.1  # 10% of model size per sample
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory_bytes * 0.8 / sample_memory)  # Use 80% of available memory
        
        # Ensure minimum batch size of 1
        return max(1, optimal_batch_size)


class MemoryMonitor:
    """Memory usage monitor for meta-learning training.
    
    This class monitors memory usage during training and provides
    alerts and optimization suggestions.
    """
    
    def __init__(self, alert_threshold: float = 0.9, log_frequency: int = 100):
        """Initialize memory monitor.
        
        Args:
            alert_threshold: Memory usage threshold for alerts (0-1)
            log_frequency: Frequency of memory logging (in steps)
        """
        self.alert_threshold = alert_threshold
        self.log_frequency = log_frequency
        self.memory_history = []
        self.step_count = 0
        self.peak_memory = 0.0
        
    def log_memory_usage(self, step: int, additional_info: Optional[Dict[str, Any]] = None):
        """Log current memory usage.
        
        Args:
            step: Current training step
            additional_info: Additional information to log
        """
        memory_info = self.get_memory_info()
        
        if additional_info:
            memory_info.update(additional_info)
        
        memory_info['step'] = step
        self.memory_history.append(memory_info)
        
        # Update peak memory
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024**3
            self.peak_memory = max(self.peak_memory, current_memory)
        
        # Log periodically
        if step % self.log_frequency == 0 and is_main_process():
            self._log_memory_status(memory_info)
        
        # Check for memory alerts
        self._check_memory_alerts(memory_info)
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory information.
        
        Returns:
            Dictionary with memory information
        """
        memory_info = {}
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
            memory_info['gpu_cached'] = torch.cuda.memory_reserved() / 1024**3  # GB
            memory_info['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
            
            # Get total GPU memory
            device = torch.cuda.current_device()
            total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
            memory_info['gpu_total'] = total_memory
            memory_info['gpu_utilization'] = memory_info['gpu_allocated'] / total_memory
        
        # CPU memory
        process = psutil.Process()
        memory_info['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        memory_info['cpu_percent'] = process.memory_percent()
        
        # System memory
        system_memory = psutil.virtual_memory()
        memory_info['system_total'] = system_memory.total / 1024**3  # GB
        memory_info['system_available'] = system_memory.available / 1024**3  # GB
        memory_info['system_utilization'] = system_memory.percent / 100.0
        
        return memory_info
    
    def _log_memory_status(self, memory_info: Dict[str, float]):
        """Log memory status.
        
        Args:
            memory_info: Memory information dictionary
        """
        print(f"Memory Status (Step {memory_info['step']}):")
        
        if 'gpu_allocated' in memory_info:
            print(f"  GPU: {memory_info['gpu_allocated']:.2f}GB / {memory_info['gpu_total']:.2f}GB "
                  f"({memory_info['gpu_utilization']*100:.1f}%)")
        
        print(f"  CPU: {memory_info['cpu_memory']:.2f}GB ({memory_info['cpu_percent']:.1f}%)")
        print(f"  System: {memory_info['system_total'] - memory_info['system_available']:.2f}GB / "
              f"{memory_info['system_total']:.2f}GB ({memory_info['system_utilization']*100:.1f}%)")
    
    def _check_memory_alerts(self, memory_info: Dict[str, float]):
        """Check for memory usage alerts.
        
        Args:
            memory_info: Memory information dictionary
        """
        # GPU memory alert
        if 'gpu_utilization' in memory_info and memory_info['gpu_utilization'] > self.alert_threshold:
            if is_main_process():
                print(f"WARNING: High GPU memory usage: {memory_info['gpu_utilization']*100:.1f}%")
                print("Consider enabling gradient checkpointing or reducing batch size.")
        
        # System memory alert
        if memory_info['system_utilization'] > self.alert_threshold:
            if is_main_process():
                print(f"WARNING: High system memory usage: {memory_info['system_utilization']*100:.1f}%")
                print("Consider reducing the number of concurrent tasks.")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary.
        
        Returns:
            Memory usage summary
        """
        if not self.memory_history:
            return {'message': 'No memory history available'}
        
        # Calculate statistics
        gpu_usage = [info.get('gpu_utilization', 0) for info in self.memory_history]
        cpu_usage = [info.get('cpu_percent', 0) / 100.0 for info in self.memory_history]
        system_usage = [info.get('system_utilization', 0) for info in self.memory_history]
        
        summary = {
            'peak_gpu_memory_gb': self.peak_memory,
            'gpu_utilization': {
                'mean': np.mean(gpu_usage),
                'max': np.max(gpu_usage),
                'std': np.std(gpu_usage)
            },
            'cpu_utilization': {
                'mean': np.mean(cpu_usage),
                'max': np.max(cpu_usage),
                'std': np.std(cpu_usage)
            },
            'system_utilization': {
                'mean': np.mean(system_usage),
                'max': np.max(system_usage),
                'std': np.std(system_usage)
            },
            'n_measurements': len(self.memory_history)
        }
        
        return summary
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    def reset(self):
        """Reset memory monitoring."""
        self.memory_history.clear()
        self.step_count = 0
        self.peak_memory = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


class MemoryOptimizedMetaLearning:
    """Memory-optimized meta-learning wrapper.
    
    This class wraps meta-learning models with memory optimization techniques
    including gradient checkpointing, mixed precision, and efficient task distribution.
    """
    
    def __init__(self, base_model, config: Dict[str, Any]):
        """Initialize memory-optimized meta-learning.
        
        Args:
            base_model: Base meta-learning model
            config: Memory optimization configuration
        """
        self.base_model = base_model
        self.config = config
        
        # Initialize optimization components
        self.gradient_checkpointing = GradientCheckpointing(
            enabled=config.get('gradient_checkpointing', False)
        )
        
        self.mixed_precision = MixedPrecisionTraining(
            enabled=config.get('mixed_precision', False)
        )
        
        self.task_optimizer = TaskDistributionOptimizer(
            world_size=config.get('world_size', 1),
            memory_limit_gb=config.get('memory_limit_gb', 10.0)
        )
        
        self.memory_monitor = MemoryMonitor(
            alert_threshold=config.get('alert_threshold', 0.9),
            log_frequency=config.get('log_frequency', 100)
        )
        
        # Wrap model with gradient checkpointing if enabled
        if self.gradient_checkpointing.enabled:
            self.base_model = self.gradient_checkpointing.wrap_model_forward(self.base_model)
    
    def optimize_task_distribution(self, tasks: List[Task]) -> List[List[Task]]:
        """Optimize task distribution across GPUs.
        
        Args:
            tasks: List of tasks to distribute
            
        Returns:
            Optimized task distribution
        """
        # Estimate model size
        model_size_mb = sum(p.numel() * 4 for p in self.base_model.parameters()) / 1024**2
        
        return self.task_optimizer.distribute_tasks_by_memory(tasks, model_size_mb)
    
    def memory_efficient_forward(self, *args, **kwargs):
        """Memory-efficient forward pass.
        
        Args:
            *args: Forward pass arguments
            **kwargs: Forward pass keyword arguments
            
        Returns:
            Forward pass output
        """
        with self.mixed_precision.autocast_context():
            if self.gradient_checkpointing.enabled:
                return self.gradient_checkpointing.checkpoint_function(
                    self.base_model.forward, *args, **kwargs
                )
            else:
                return self.base_model.forward(*args, **kwargs)
    
    def memory_efficient_backward(self, loss: torch.Tensor):
        """Memory-efficient backward pass.
        
        Args:
            loss: Loss tensor
        """
        self.mixed_precision.backward(loss)
    
    def memory_efficient_step(self, optimizer: torch.optim.Optimizer):
        """Memory-efficient optimizer step.
        
        Args:
            optimizer: Optimizer to step
        """
        self.mixed_precision.step(optimizer)
    
    def monitor_memory(self, step: int, additional_info: Optional[Dict[str, Any]] = None):
        """Monitor memory usage.
        
        Args:
            step: Current training step
            additional_info: Additional information to log
        """
        self.memory_monitor.log_memory_usage(step, additional_info)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get memory usage summary.
        
        Returns:
            Memory usage summary
        """
        return self.memory_monitor.get_memory_summary()
    
    def cleanup_memory(self):
        """Clean up memory."""
        self.memory_monitor.clear_cache()


def create_memory_optimized_model(base_model, optimization_config: Dict[str, Any]):
    """Create memory-optimized version of a meta-learning model.
    
    Args:
        base_model: Base meta-learning model
        optimization_config: Memory optimization configuration
        
    Returns:
        Memory-optimized model
    """
    return MemoryOptimizedMetaLearning(base_model, optimization_config)