"""Meta-learning utilities that integrate with existing PINNacle src/utils."""

import torch
import numpy as np
import time
import os
import json
from typing import Dict, List, Optional, Any, Tuple, Union
from contextlib import contextmanager

from .task import Task, TaskData, TaskBatch
from .config import MetaLearningConfig


class MetaLearningLogger:
    """Logger for meta-learning training that extends PINNacle's logging patterns."""
    
    def __init__(self, log_dir: str, experiment_name: str):
        """Initialize logger.
        
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_meta_log.txt")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'meta_train_loss': [],
            'meta_val_loss': [],
            'adaptation_times': [],
            'meta_iteration_times': [],
            'memory_usage': []
        }
    
    def log_meta_iteration(self, iteration: int, train_loss: float, 
                          val_loss: Optional[float] = None, 
                          adaptation_time: Optional[float] = None,
                          memory_usage: Optional[float] = None):
        """Log meta-learning iteration results."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to file
        with open(self.log_file, 'a') as f:
            log_msg = f"[{timestamp}] Iteration {iteration}: train_loss={train_loss:.6f}"
            if val_loss is not None:
                log_msg += f", val_loss={val_loss:.6f}"
            if adaptation_time is not None:
                log_msg += f", adapt_time={adaptation_time:.4f}s"
            if memory_usage is not None:
                log_msg += f", memory={memory_usage:.2f}MB"
            f.write(log_msg + "\n")
        
        # Store metrics
        self.metrics['meta_train_loss'].append(train_loss)
        if val_loss is not None:
            self.metrics['meta_val_loss'].append(val_loss)
        if adaptation_time is not None:
            self.metrics['adaptation_times'].append(adaptation_time)
        if memory_usage is not None:
            self.metrics['memory_usage'].append(memory_usage)
    
    def save_metrics(self):
        """Save metrics to JSON file."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_message(self, message: str):
        """Log a general message."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")


class MetaLearningTimer:
    """Timer utility for measuring meta-learning performance."""
    
    def __init__(self):
        self.timers = {}
        self.start_times = {}
    
    def start(self, name: str):
        """Start timing for named operation."""
        self.start_times[name] = time.time()
    
    def stop(self, name: str) -> float:
        """Stop timing and return elapsed time."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        if name not in self.timers:
            self.timers[name] = []
        self.timers[name].append(elapsed)
        
        del self.start_times[name]
        return elapsed
    
    @contextmanager
    def time_context(self, name: str):
        """Context manager for timing operations."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def get_average_time(self, name: str) -> float:
        """Get average time for named operation."""
        if name not in self.timers or not self.timers[name]:
            return 0.0
        return np.mean(self.timers[name])
    
    def get_total_time(self, name: str) -> float:
        """Get total time for named operation."""
        if name not in self.timers:
            return 0.0
        return sum(self.timers[name])
    
    def reset(self, name: Optional[str] = None):
        """Reset timers."""
        if name is None:
            self.timers.clear()
            self.start_times.clear()
        else:
            if name in self.timers:
                del self.timers[name]
            if name in self.start_times:
                del self.start_times[name]


class GradientUtils:
    """Utilities for gradient computation in meta-learning."""
    
    @staticmethod
    def compute_meta_gradients(model: torch.nn.Module, 
                              loss: torch.Tensor,
                              create_graph: bool = True,
                              allow_unused: bool = True) -> Dict[str, torch.Tensor]:
        """Compute gradients for meta-learning.
        
        Args:
            model: Neural network model
            loss: Loss tensor
            create_graph: Whether to create computation graph for higher-order derivatives
            allow_unused: Whether to allow unused parameters
            
        Returns:
            Dictionary mapping parameter names to gradients
        """
        grads = torch.autograd.grad(
            loss, 
            model.parameters(), 
            create_graph=create_graph,
            allow_unused=allow_unused
        )
        
        grad_dict = {}
        for (name, param), grad in zip(model.named_parameters(), grads):
            if grad is not None:
                grad_dict[name] = grad
        
        return grad_dict
    
    @staticmethod
    def apply_gradients(model: torch.nn.Module, 
                       gradients: Dict[str, torch.Tensor],
                       learning_rate: float) -> Dict[str, torch.Tensor]:
        """Apply gradients to model parameters.
        
        Args:
            model: Neural network model
            gradients: Dictionary of gradients
            learning_rate: Learning rate for gradient update
            
        Returns:
            Dictionary of updated parameters
        """
        updated_params = {}
        
        for name, param in model.named_parameters():
            if name in gradients:
                updated_params[name] = param - learning_rate * gradients[name]
            else:
                updated_params[name] = param.clone()
        
        return updated_params
    
    @staticmethod
    def clip_gradients(gradients: Dict[str, torch.Tensor], 
                      max_norm: float) -> Dict[str, torch.Tensor]:
        """Clip gradients by norm.
        
        Args:
            gradients: Dictionary of gradients
            max_norm: Maximum gradient norm
            
        Returns:
            Dictionary of clipped gradients
        """
        # Compute total norm
        total_norm = 0.0
        for grad in gradients.values():
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            clipped_grads = {}
            for name, grad in gradients.items():
                if grad is not None:
                    clipped_grads[name] = grad * clip_coef
                else:
                    clipped_grads[name] = grad
            return clipped_grads
        
        return gradients


class MemoryUtils:
    """Utilities for memory management in meta-learning."""
    
    @staticmethod
    def get_memory_usage() -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    @staticmethod
    def clear_cache():
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    @contextmanager
    def memory_efficient_forward(self, model: torch.nn.Module):
        """Context manager for memory-efficient forward passes."""
        # Store original training mode
        training_mode = model.training
        
        try:
            # Use evaluation mode to save memory
            model.eval()
            with torch.no_grad():
                yield
        finally:
            # Restore original training mode
            model.train(training_mode)


class TaskUtils:
    """Utilities for task manipulation and processing."""
    
    @staticmethod
    def create_task_batch(tasks: List[Task], batch_size: int, 
                         shuffle: bool = True) -> List[TaskBatch]:
        """Create batches from list of tasks.
        
        Args:
            tasks: List of tasks
            batch_size: Size of each batch
            shuffle: Whether to shuffle tasks before batching
            
        Returns:
            List of TaskBatch objects
        """
        if shuffle:
            tasks = tasks.copy()
            np.random.shuffle(tasks)
        
        batches = []
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            batches.append(TaskBatch(batch_tasks))
        
        return batches
    
    @staticmethod
    def balance_task_distribution(tasks: List[Task], 
                                 balance_key: str = 'problem_type') -> List[Task]:
        """Balance task distribution by specified key.
        
        Args:
            tasks: List of tasks
            balance_key: Key to balance by (e.g., 'problem_type')
            
        Returns:
            Balanced list of tasks
        """
        # Group tasks by balance key
        groups = {}
        for task in tasks:
            if balance_key == 'problem_type':
                key = task.problem_type
            else:
                key = task.metadata.get(balance_key, 'unknown')
            
            if key not in groups:
                groups[key] = []
            groups[key].append(task)
        
        # Find minimum group size
        min_size = min(len(group) for group in groups.values())
        
        # Sample equally from each group
        balanced_tasks = []
        for group in groups.values():
            sampled = np.random.choice(group, min_size, replace=False)
            balanced_tasks.extend(sampled)
        
        return balanced_tasks
    
    @staticmethod
    def split_tasks(tasks: List[Task], 
                   train_ratio: float = 0.7,
                   val_ratio: float = 0.15,
                   test_ratio: float = 0.15) -> Tuple[List[Task], List[Task], List[Task]]:
        """Split tasks into train/validation/test sets.
        
        Args:
            tasks: List of tasks
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            
        Returns:
            Tuple of (train_tasks, val_tasks, test_tasks)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        n_tasks = len(tasks)
        n_train = int(n_tasks * train_ratio)
        n_val = int(n_tasks * val_ratio)
        
        # Shuffle tasks
        shuffled_tasks = tasks.copy()
        np.random.shuffle(shuffled_tasks)
        
        train_tasks = shuffled_tasks[:n_train]
        val_tasks = shuffled_tasks[n_train:n_train + n_val]
        test_tasks = shuffled_tasks[n_train + n_val:]
        
        return train_tasks, val_tasks, test_tasks


class ConfigUtils:
    """Utilities for configuration management."""
    
    @staticmethod
    def save_config(config: MetaLearningConfig, filepath: str):
        """Save configuration to file.
        
        Args:
            config: Configuration object
            filepath: Path to save configuration
        """
        config_dict = config.to_dict()
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @staticmethod
    def load_config(filepath: str, config_class: type) -> MetaLearningConfig:
        """Load configuration from file.
        
        Args:
            filepath: Path to configuration file
            config_class: Configuration class to instantiate
            
        Returns:
            Configuration object
        """
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert dtype string back to torch.dtype
        if 'dtype' in config_dict:
            dtype_str = config_dict['dtype']
            if dtype_str == 'torch.float32':
                config_dict['dtype'] = torch.float32
            elif dtype_str == 'torch.float64':
                config_dict['dtype'] = torch.float64
        
        return config_class(**config_dict)
    
    @staticmethod
    def merge_configs(base_config: MetaLearningConfig, 
                     override_config: Dict[str, Any]) -> MetaLearningConfig:
        """Merge configuration with overrides.
        
        Args:
            base_config: Base configuration
            override_config: Dictionary of overrides
            
        Returns:
            Merged configuration
        """
        config_dict = base_config.to_dict()
        config_dict.update(override_config)
        
        return type(base_config)(**config_dict)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility.
    
    This extends PINNacle's existing seed setting patterns.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = 'auto') -> torch.device:
    """Get torch device, extending PINNacle's device handling.
    
    Args:
        device_str: Device specification ('auto', 'cpu', 'cuda', 'cuda:0', etc.)
        
    Returns:
        torch.device object
    """
    if device_str == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        return torch.device(device_str)


def create_meta_learning_directory_structure(base_dir: str, experiment_name: str) -> Dict[str, str]:
    """Create directory structure for meta-learning experiments.
    
    This follows PINNacle's existing directory patterns.
    
    Args:
        base_dir: Base directory for experiments
        experiment_name: Name of the experiment
        
    Returns:
        Dictionary mapping directory types to paths
    """
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    directories = {
        'experiment': experiment_dir,
        'models': os.path.join(experiment_dir, 'models'),
        'logs': os.path.join(experiment_dir, 'logs'),
        'results': os.path.join(experiment_dir, 'results'),
        'plots': os.path.join(experiment_dir, 'plots'),
        'configs': os.path.join(experiment_dir, 'configs')
    }
    
    # Create all directories
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return directories