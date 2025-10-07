"""Task and TaskData classes for few-shot learning data structures."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import torch
import numpy as np


@dataclass
class TaskData:
    """Data for a single task in meta-learning.
    
    This class represents the data structure used for both support and query sets
    in few-shot learning scenarios for physics-informed neural networks.
    """
    inputs: torch.Tensor  # Input coordinates (x, t, etc.)
    outputs: torch.Tensor  # Target values u(x, t)
    collocation_points: torch.Tensor  # Points for physics loss computation
    boundary_data: Optional[torch.Tensor] = None  # Boundary condition data
    initial_data: Optional[torch.Tensor] = None  # Initial condition data
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return the number of data points."""
        return len(self.inputs)
    
    def to_device(self, device: torch.device) -> 'TaskData':
        """Move all tensors to specified device."""
        return TaskData(
            inputs=self.inputs.to(device),
            outputs=self.outputs.to(device),
            collocation_points=self.collocation_points.to(device),
            boundary_data=self.boundary_data.to(device) if self.boundary_data is not None else None,
            initial_data=self.initial_data.to(device) if self.initial_data is not None else None,
            metadata=self.metadata
        )
    
    def sample(self, n_samples: int, replace: bool = False) -> 'TaskData':
        """Sample n points from this TaskData.
        
        Args:
            n_samples: Number of samples to draw
            replace: Whether to sample with replacement
            
        Returns:
            New TaskData with sampled points
        """
        if n_samples > len(self) and not replace:
            raise ValueError(f"Cannot sample {n_samples} points without replacement from {len(self)} points")
        
        indices = np.random.choice(len(self), n_samples, replace=replace)
        
        return TaskData(
            inputs=self.inputs[indices],
            outputs=self.outputs[indices],
            collocation_points=self.collocation_points[indices] if len(self.collocation_points) > 0 else self.collocation_points,
            boundary_data=self.boundary_data[indices] if self.boundary_data is not None else None,
            initial_data=self.initial_data[indices] if self.initial_data is not None else None,
            metadata=self.metadata.copy()
        )


@dataclass
class Task:
    """Represents a single physics task for meta-learning.
    
    A task contains both support and query data, along with problem-specific
    parameters and metadata. This is the fundamental unit for few-shot learning
    in physics-informed neural networks.
    """
    problem_type: str  # Type of physics problem (e.g., 'heat_1d', 'burgers_2d')
    parameters: Dict[str, float]  # Problem parameters (e.g., diffusivity, viscosity)
    support_data: TaskData  # Data for adaptation/training
    query_data: TaskData  # Data for evaluation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def sample_support(self, n_samples: int, replace: bool = False) -> TaskData:
        """Sample n points from support set.
        
        Args:
            n_samples: Number of support samples to draw
            replace: Whether to sample with replacement
            
        Returns:
            TaskData with sampled support points
        """
        return self.support_data.sample(n_samples, replace)
    
    def sample_query(self, n_samples: int, replace: bool = False) -> TaskData:
        """Sample n points from query set.
        
        Args:
            n_samples: Number of query samples to draw
            replace: Whether to sample with replacement
            
        Returns:
            TaskData with sampled query points
        """
        return self.query_data.sample(n_samples, replace)
    
    def to_device(self, device: torch.device) -> 'Task':
        """Move all task data to specified device."""
        return Task(
            problem_type=self.problem_type,
            parameters=self.parameters.copy(),
            support_data=self.support_data.to_device(device),
            query_data=self.query_data.to_device(device),
            metadata=self.metadata.copy()
        )
    
    def get_parameter_vector(self) -> torch.Tensor:
        """Convert parameters dict to tensor vector for model conditioning."""
        param_values = [self.parameters[key] for key in sorted(self.parameters.keys())]
        return torch.tensor(param_values, dtype=torch.float32)


@dataclass
class TaskBatch:
    """Batch of tasks for efficient meta-learning training."""
    tasks: List[Task]
    
    def __len__(self) -> int:
        """Return number of tasks in batch."""
        return len(self.tasks)
    
    def to_device(self, device: torch.device) -> 'TaskBatch':
        """Move all tasks to specified device."""
        return TaskBatch([task.to_device(device) for task in self.tasks])
    
    def sample_support_batch(self, n_samples: int) -> List[TaskData]:
        """Sample support data from all tasks in batch."""
        return [task.sample_support(n_samples) for task in self.tasks]
    
    def sample_query_batch(self, n_samples: int) -> List[TaskData]:
        """Sample query data from all tasks in batch."""
        return [task.sample_query(n_samples) for task in self.tasks]


class TaskDistribution:
    """Manages distribution of tasks for meta-learning.
    
    This class handles the generation and management of task distributions
    for training, validation, and testing in meta-learning scenarios.
    """
    
    def __init__(self, problem_type: str, parameter_ranges: Dict[str, tuple]):
        """Initialize task distribution.
        
        Args:
            problem_type: Type of physics problem
            parameter_ranges: Dict mapping parameter names to (min, max) ranges
        """
        self.problem_type = problem_type
        self.parameter_ranges = parameter_ranges
    
    def sample_parameters(self, n_tasks: int) -> List[Dict[str, float]]:
        """Sample parameters for n tasks from the distribution.
        
        Args:
            n_tasks: Number of parameter sets to sample
            
        Returns:
            List of parameter dictionaries
        """
        parameter_sets = []
        for _ in range(n_tasks):
            params = {}
            for param_name, (min_val, max_val) in self.parameter_ranges.items():
                params[param_name] = np.random.uniform(min_val, max_val)
            parameter_sets.append(params)
        return parameter_sets
    
    def create_task_batch(self, n_tasks: int) -> TaskBatch:
        """Create a batch of tasks by sampling from the distribution.
        
        Args:
            n_tasks: Number of tasks to create
            
        Returns:
            TaskBatch containing sampled tasks
        """
        # This is a placeholder - actual implementation will depend on specific PDE
        # and will be implemented in the parametric PDE extensions
        raise NotImplementedError("Task creation must be implemented by specific PDE classes")