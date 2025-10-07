"""Unified parametric PDE module for meta-learning tasks.

This module provides a unified interface to all parametric PDE implementations
for few-shot learning scenarios in physics-informed neural networks.
"""

from typing import Dict, List, Type, Union
import numpy as np

from .parametric_heat import (
    ParametricHeat1D, 
    ParametricHeat2D_VaryingCoef, 
    HeatTaskDistribution
)
from .parametric_burgers import (
    ParametricBurgers1D, 
    ParametricBurgers2D, 
    BurgersTaskDistribution
)
from .parametric_poisson_ns import (
    ParametricPoisson2D_Classic, 
    ParametricNS2D_LidDriven,
    PoissonTaskDistribution,
    NavierStokesTaskDistribution
)
from .parametric_chaotic import (
    ParametricGrayScottEquation,
    ParametricKuramotoSivashinskyEquation,
    ChaoticPDETaskDistribution
)
from ..meta_learning.task import Task, TaskDistribution


# Registry of all parametric PDE classes
PARAMETRIC_PDE_REGISTRY = {
    # Heat equations
    'heat_1d': ParametricHeat1D,
    'heat_2d_varying_coef': ParametricHeat2D_VaryingCoef,
    
    # Burgers equations
    'burgers_1d': ParametricBurgers1D,
    'burgers_2d': ParametricBurgers2D,
    
    # Poisson equations
    'poisson_2d_classic': ParametricPoisson2D_Classic,
    
    # Navier-Stokes equations
    'ns_2d_lid_driven': ParametricNS2D_LidDriven,
    
    # Chaotic PDEs
    'gray_scott': ParametricGrayScottEquation,
    'kuramoto_sivashinsky': ParametricKuramotoSivashinskyEquation,
}

# Registry of task distributions
TASK_DISTRIBUTION_REGISTRY = {
    'heat_1d': lambda **kwargs: HeatTaskDistribution('1d', **kwargs),
    'heat_2d_varying_coef': lambda **kwargs: HeatTaskDistribution('2d', **kwargs),
    'burgers_1d': lambda **kwargs: BurgersTaskDistribution('1d', **kwargs),
    'burgers_2d': lambda **kwargs: BurgersTaskDistribution('2d', **kwargs),
    'poisson_2d_classic': lambda **kwargs: PoissonTaskDistribution(**kwargs),
    'ns_2d_lid_driven': lambda **kwargs: NavierStokesTaskDistribution(**kwargs),
    'gray_scott': lambda **kwargs: ChaoticPDETaskDistribution('gray_scott', **kwargs),
    'kuramoto_sivashinsky': lambda **kwargs: ChaoticPDETaskDistribution('kuramoto_sivashinsky', **kwargs),
}


def get_parametric_pde_class(problem_type: str) -> Type:
    """Get parametric PDE class by problem type.
    
    Args:
        problem_type: Type of PDE problem
        
    Returns:
        Parametric PDE class
        
    Raises:
        ValueError: If problem type is not found
    """
    if problem_type not in PARAMETRIC_PDE_REGISTRY:
        available_types = list(PARAMETRIC_PDE_REGISTRY.keys())
        raise ValueError(f"Unknown problem type: {problem_type}. "
                        f"Available types: {available_types}")
    
    return PARAMETRIC_PDE_REGISTRY[problem_type]


def create_task_distribution(problem_type: str, **kwargs) -> TaskDistribution:
    """Create task distribution for given problem type.
    
    Args:
        problem_type: Type of PDE problem
        **kwargs: Additional parameters for the task distribution
        
    Returns:
        TaskDistribution instance
        
    Raises:
        ValueError: If problem type is not found
    """
    if problem_type not in TASK_DISTRIBUTION_REGISTRY:
        available_types = list(TASK_DISTRIBUTION_REGISTRY.keys())
        raise ValueError(f"Unknown problem type: {problem_type}. "
                        f"Available types: {available_types}")
    
    return TASK_DISTRIBUTION_REGISTRY[problem_type](**kwargs)


def generate_meta_learning_tasks(problem_types: List[str], 
                                n_train: int = 100, 
                                n_val: int = 20, 
                                n_test: int = 50,
                                **kwargs) -> Dict[str, Dict[str, List[Task]]]:
    """Generate complete meta-learning task distributions for multiple problem types.
    
    Args:
        problem_types: List of problem types to generate tasks for
        n_train: Number of training tasks per problem type
        n_val: Number of validation tasks per problem type
        n_test: Number of test tasks per problem type
        **kwargs: Additional parameters passed to task distributions
        
    Returns:
        Dictionary mapping problem types to task distributions
        
    Example:
        >>> tasks = generate_meta_learning_tasks(
        ...     ['heat_1d', 'burgers_1d', 'poisson_2d_classic'],
        ...     n_train=100, n_val=20, n_test=50
        ... )
        >>> print(f"Generated {len(tasks['heat_1d']['train'])} heat training tasks")
    """
    all_tasks = {}
    
    for problem_type in problem_types:
        # Create task distribution
        task_dist = create_task_distribution(problem_type, **kwargs)
        
        # Generate tasks
        train_tasks = task_dist.pde_instance.generate_train_tasks(n_train)
        val_tasks = task_dist.pde_instance.generate_val_tasks(n_val)
        test_tasks = task_dist.pde_instance.generate_test_tasks(n_test)
        
        all_tasks[problem_type] = {
            'train': train_tasks,
            'val': val_tasks,
            'test': test_tasks
        }
    
    return all_tasks


def get_problem_parameter_ranges(problem_type: str) -> Dict[str, tuple]:
    """Get parameter ranges for a given problem type.
    
    Args:
        problem_type: Type of PDE problem
        
    Returns:
        Dictionary mapping parameter names to (min, max) ranges
    """
    parameter_ranges = {
        'heat_1d': {'diffusivity': (0.01, 0.1)},
        'heat_2d_varying_coef': {'diffusivity': (0.1, 2.0)},
        'burgers_1d': {'viscosity': (0.001, 0.1), 'ic_variant': (0, 4)},
        'burgers_2d': {'viscosity': (0.001, 0.01)},
        'poisson_2d_classic': {'boundary_value': (0.5, 2.0), 'source_strength': (0.1, 5.0)},
        'ns_2d_lid_driven': {'reynolds': (100, 10000), 'lid_velocity': (1, 8)},
        'gray_scott': {
            'b': (0.02, 0.08), 'd': (0.05, 0.15),
            'epsilon_u': (5e-6, 2e-5), 'epsilon_v': (2e-6, 1e-5)
        },
        'kuramoto_sivashinsky': {
            'alpha': (50/16, 200/16), 'beta': (50/(16*16), 200/(16*16)),
            'gamma': (50/(16**4), 200/(16**4))
        }
    }
    
    if problem_type not in parameter_ranges:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    return parameter_ranges[problem_type]


def create_benchmark_task_suite() -> Dict[str, Dict[str, List[Task]]]:
    """Create a complete benchmark suite with all parametric PDEs.
    
    This creates the standard task distribution used in the meta-learning
    benchmark with 100 train, 20 val, 50 test tasks per PDE type.
    
    Returns:
        Complete benchmark task suite
    """
    problem_types = [
        'heat_1d',
        'heat_2d_varying_coef', 
        'burgers_1d',
        'burgers_2d',
        'poisson_2d_classic',
        'ns_2d_lid_driven',
        'gray_scott',
        'kuramoto_sivashinsky'
    ]
    
    return generate_meta_learning_tasks(
        problem_types=problem_types,
        n_train=100,
        n_val=20, 
        n_test=50
    )


def validate_task_distributions(tasks: Dict[str, Dict[str, List[Task]]]) -> Dict[str, Dict[str, int]]:
    """Validate task distributions and return statistics.
    
    Args:
        tasks: Task distributions to validate
        
    Returns:
        Statistics about the task distributions
    """
    stats = {}
    
    for problem_type, task_sets in tasks.items():
        stats[problem_type] = {}
        
        for split_name, task_list in task_sets.items():
            stats[problem_type][split_name] = len(task_list)
            
            # Validate task structure
            for i, task in enumerate(task_list):
                assert task.problem_type == problem_type, \
                    f"Task {i} has wrong problem type: {task.problem_type} != {problem_type}"
                assert len(task.support_data) > 0, f"Task {i} has empty support data"
                assert len(task.query_data) > 0, f"Task {i} has empty query data"
                assert len(task.support_data.collocation_points) > 0, \
                    f"Task {i} has empty collocation points"
    
    return stats


# Convenience functions for specific problem types
def create_heat_tasks(variant: str = '1d', **kwargs) -> Dict[str, List[Task]]:
    """Create heat equation tasks."""
    problem_type = f'heat_{variant}'
    task_dist = create_task_distribution(problem_type, **kwargs)
    return {
        'train': task_dist.pde_instance.generate_train_tasks(100),
        'val': task_dist.pde_instance.generate_val_tasks(20),
        'test': task_dist.pde_instance.generate_test_tasks(50)
    }


def create_burgers_tasks(variant: str = '1d', **kwargs) -> Dict[str, List[Task]]:
    """Create Burgers equation tasks."""
    problem_type = f'burgers_{variant}'
    task_dist = create_task_distribution(problem_type, **kwargs)
    return {
        'train': task_dist.pde_instance.generate_train_tasks(100),
        'val': task_dist.pde_instance.generate_val_tasks(20),
        'test': task_dist.pde_instance.generate_test_tasks(50)
    }


def create_poisson_tasks(**kwargs) -> Dict[str, List[Task]]:
    """Create Poisson equation tasks."""
    task_dist = create_task_distribution('poisson_2d_classic', **kwargs)
    return {
        'train': task_dist.pde_instance.generate_train_tasks(100),
        'val': task_dist.pde_instance.generate_val_tasks(20),
        'test': task_dist.pde_instance.generate_test_tasks(50)
    }


def create_navier_stokes_tasks(**kwargs) -> Dict[str, List[Task]]:
    """Create Navier-Stokes equation tasks."""
    task_dist = create_task_distribution('ns_2d_lid_driven', **kwargs)
    return {
        'train': task_dist.pde_instance.generate_train_tasks(100),
        'val': task_dist.pde_instance.generate_val_tasks(20),
        'test': task_dist.pde_instance.generate_test_tasks(50)
    }


def create_chaotic_tasks(variant: str = 'gray_scott', **kwargs) -> Dict[str, List[Task]]:
    """Create chaotic PDE tasks."""
    task_dist = create_task_distribution(variant, **kwargs)
    return {
        'train': task_dist.pde_instance.generate_train_tasks(100),
        'val': task_dist.pde_instance.generate_val_tasks(20),
        'test': task_dist.pde_instance.generate_test_tasks(50)
    }