"""Parametric Burgers equation tasks for meta-learning."""

import numpy as np
import torch
import scipy
from typing import List, Dict, Tuple, Optional

import deepxde as dde
from . import baseclass
from ..meta_learning.task import Task, TaskData, TaskDistribution


class ParametricBurgers1D(baseclass.BaseTimePDE):
    """Parametric 1D Burgers equation for meta-learning.
    
    Extends Burgers1D to generate tasks with varying viscosity parameters
    and different initial conditions for few-shot learning scenarios.
    """

    def __init__(self, 
                 geom=[-1, 1], 
                 time=[0, 1], 
                 viscosity_range=(0.001, 0.1),
                 initial_condition_variants=5,
                 n_support_points=50,
                 n_query_points=1000,
                 n_collocation_points=2000):
        """Initialize parametric 1D Burgers equation.
        
        Args:
            geom: Spatial domain [x_min, x_max]
            time: Time domain [t_min, t_max]
            viscosity_range: Range of viscosity values ν ∈ [ν_min, ν_max]
            initial_condition_variants: Number of different initial condition types
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.geom_bounds = geom
        self.time_bounds = time
        self.viscosity_range = viscosity_range
        self.initial_condition_variants = initial_condition_variants
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension
        self.output_dim = 1
        
        # Geometry setup
        self.geom = dde.geometry.Interval(*geom)
        timedomain = dde.geometry.TimeDomain(*time)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        self.bbox = geom + time
        
        # Current parameters (will be set for each task)
        self.current_viscosity = 0.01 / np.pi
        self.current_ic_variant = 0
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points()

    def _setup_pde(self):
        """Setup 1D Burgers equation PDE."""
        def burgers_pde(x, u):
            """Burgers equation: ∂u/∂t + u ∂u/∂x = ν ∂²u/∂x²"""
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t + u * u_x - self.current_viscosity * u_xx

        self.pde = burgers_pde
        self.set_pdeloss(num=1)

    def _setup_boundary_conditions(self):
        """Setup boundary and initial conditions."""
        def ic_func(x):
            """Initial condition based on current variant."""
            return self.get_initial_condition(x, self.current_ic_variant)

        self.add_bcs([{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': (lambda _, on_boundary: on_boundary),
            'type': 'dirichlet'
        }])

    def get_initial_condition(self, x: np.ndarray, variant: int) -> np.ndarray:
        """Get initial condition based on variant.
        
        Args:
            x: Spatial coordinates
            variant: Initial condition variant (0-4)
            
        Returns:
            Initial condition values
        """
        if variant == 0:
            # Original sin wave
            return np.sin(-np.pi * x[:, 0:1])
        elif variant == 1:
            # Gaussian pulse
            return np.exp(-10 * (x[:, 0:1] - 0.3)**2)
        elif variant == 2:
            # Step function
            return np.where(x[:, 0:1] < 0, 1.0, -1.0)
        elif variant == 3:
            # Double sin wave
            return 0.5 * (np.sin(-2 * np.pi * x[:, 0:1]) + np.sin(-np.pi * x[:, 0:1]))
        elif variant == 4:
            # Triangular wave
            return 2 * np.abs(x[:, 0:1]) - 1
        else:
            # Default to sin wave
            return np.sin(-np.pi * x[:, 0:1])

    def set_parameters(self, viscosity: float, ic_variant: int):
        """Set parameters for current task.
        
        Args:
            viscosity: Viscosity parameter ν
            ic_variant: Initial condition variant
        """
        self.current_viscosity = viscosity
        self.current_ic_variant = ic_variant
        self._setup_pde()
        self._setup_boundary_conditions()

    def generate_reference_solution(self, viscosity: float, ic_variant: int, 
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution using finite difference method.
        
        Args:
            viscosity: Viscosity parameter
            ic_variant: Initial condition variant
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) with [x, t, u] values
        """
        # Simple finite difference solution for reference
        nx, nt = 100, 100
        x = np.linspace(self.geom_bounds[0], self.geom_bounds[1], nx)
        t = np.linspace(self.time_bounds[0], self.time_bounds[1], nt)
        dx = x[1] - x[0]
        dt = t[1] - t[0]
        
        # Initialize solution
        u = np.zeros((nt, nx))
        
        # Set initial condition
        x_ic = x.reshape(-1, 1)
        u[0, :] = self.get_initial_condition(x_ic, ic_variant).flatten()
        
        # Apply boundary conditions
        u[:, 0] = 0
        u[:, -1] = 0
        
        # Time stepping (simple explicit scheme)
        for n in range(nt - 1):
            for i in range(1, nx - 1):
                # Burgers equation discretization
                u_x = (u[n, i+1] - u[n, i-1]) / (2 * dx)
                u_xx = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / (dx**2)
                u[n+1, i] = u[n, i] - dt * u[n, i] * u_x + dt * viscosity * u_xx
        
        # Sample random points from the solution
        t_mesh, x_mesh = np.meshgrid(t, x, indexing='ij')
        x_flat = x_mesh.flatten()
        t_flat = t_mesh.flatten()
        u_flat = u.flatten()
        
        # Randomly sample points
        indices = np.random.choice(len(x_flat), min(n_points, len(x_flat)), replace=False)
        
        return np.column_stack([x_flat[indices], t_flat[indices], u_flat[indices]])

    def create_task_data(self, viscosity: float, ic_variant: int,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for specific parameters.
        
        Args:
            viscosity: Viscosity parameter
            ic_variant: Initial condition variant
            n_support: Number of support points
            n_query: Number of query points
            n_collocation: Number of collocation points
            
        Returns:
            Tuple of (support_data, query_data)
        """
        # Generate reference solution
        ref_data = self.generate_reference_solution(viscosity, ic_variant, n_support + n_query)
        
        # Split into support and query
        support_indices = np.random.choice(len(ref_data), n_support, replace=False)
        query_indices = np.setdiff1d(np.arange(len(ref_data)), support_indices)[:n_query]
        
        # Create support data
        support_inputs = torch.tensor(ref_data[support_indices, :2], dtype=torch.float32)
        support_outputs = torch.tensor(ref_data[support_indices, 2:3], dtype=torch.float32)
        
        # Create query data
        query_inputs = torch.tensor(ref_data[query_indices, :2], dtype=torch.float32)
        query_outputs = torch.tensor(ref_data[query_indices, 2:3], dtype=torch.float32)
        
        # Generate collocation points
        x_col = np.random.uniform(self.geom_bounds[0], self.geom_bounds[1], n_collocation)
        t_col = np.random.uniform(self.time_bounds[0], self.time_bounds[1], n_collocation)
        collocation_points = torch.tensor(np.column_stack([x_col, t_col]), dtype=torch.float32)
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'viscosity': viscosity, 'ic_variant': ic_variant}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'viscosity': viscosity, 'ic_variant': ic_variant}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying viscosity and initial conditions.
        
        Args:
            n_tasks: Number of training tasks to generate
            
        Returns:
            List of Task objects for training
        """
        tasks = []
        
        for i in range(n_tasks):
            # Sample viscosity from specified range
            viscosity = np.random.uniform(*self.viscosity_range)
            
            # Sample initial condition variant
            ic_variant = np.random.randint(0, self.initial_condition_variants)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                viscosity, ic_variant,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='burgers_1d',
                parameters={'viscosity': viscosity, 'ic_variant': ic_variant},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'geom_bounds': self.geom_bounds,
                    'time_bounds': self.time_bounds,
                    'task_id': i
                }
            )
            
            tasks.append(task)
        
        return tasks

    def generate_val_tasks(self, n_tasks: int) -> List[Task]:
        """Generate validation tasks."""
        np.random.seed(42)
        tasks = self.generate_train_tasks(n_tasks)
        np.random.seed()
        return tasks

    def generate_test_tasks(self, n_tasks: int) -> List[Task]:
        """Generate test tasks."""
        np.random.seed(123)
        tasks = self.generate_train_tasks(n_tasks)
        np.random.seed()
        return tasks


class ParametricBurgers2D(baseclass.BaseTimePDE):
    """Parametric 2D Burgers equation for meta-learning.
    
    Extends Burgers2D to generate tasks with varying viscosity parameters
    and different initial conditions.
    """

    def __init__(self, 
                 L=4, 
                 T=1, 
                 viscosity_range=(0.001, 0.01),
                 n_support_points=100,
                 n_query_points=2000,
                 n_collocation_points=4000):
        """Initialize parametric 2D Burgers equation.
        
        Args:
            L: Spatial domain size [0, L] x [0, L]
            T: Time domain [0, T]
            viscosity_range: Range of viscosity values
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.L = L
        self.T = T
        self.viscosity_range = viscosity_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension (u and v components)
        self.output_dim = 2
        
        # Geometry setup
        self.bbox = [0, L, 0, L, 0, T]
        self.geom = dde.geometry.Rectangle([0, 0], [L, L])
        timedomain = dde.geometry.TimeDomain(0, T)
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        
        # Current viscosity
        self.current_viscosity = 0.001
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points(mul=4)

    def _setup_pde(self):
        """Setup 2D Burgers equation PDE."""
        def burgers_pde_2d(x, u):
            """2D Burgers equation system."""
            u1, u2 = u[:, 0:1], u[:, 1:2]

            u1_x = dde.grad.jacobian(u, x, i=0, j=0)
            u1_y = dde.grad.jacobian(u, x, i=0, j=1)
            u1_t = dde.grad.jacobian(u, x, i=0, j=2)
            u1_xx = dde.grad.hessian(u, x, i=0, j=0, component=0)
            u1_yy = dde.grad.hessian(u, x, i=1, j=1, component=0)

            u2_x = dde.grad.jacobian(u, x, i=1, j=0)
            u2_y = dde.grad.jacobian(u, x, i=1, j=1)
            u2_t = dde.grad.jacobian(u, x, i=1, j=2)
            u2_xx = dde.grad.hessian(u, x, i=0, j=0, component=1)
            u2_yy = dde.grad.hessian(u, x, i=1, j=1, component=1)
            
            return [
                u1_t + u1 * u1_x + u2 * u1_y - self.current_viscosity * (u1_xx + u1_yy),
                u2_t + u1 * u2_x + u2 * u2_y - self.current_viscosity * (u2_xx + u2_yy)
            ]

        self.pde = burgers_pde_2d
        self.set_pdeloss(num=2)

    def _setup_boundary_conditions(self):
        """Setup boundary and initial conditions."""
        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], 0)

        def boundary_xb(x, on_boundary):
            return on_boundary and (np.isclose(x[0], 0) or np.isclose(x[0], self.L))

        def boundary_yb(x, on_boundary):
            return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], self.L))

        # Simple initial conditions (can be made parametric)
        def ic_func_u(x):
            return np.sin(np.pi * x[:, 0:1] / self.L) * np.sin(np.pi * x[:, 1:2] / self.L)

        def ic_func_v(x):
            return np.cos(np.pi * x[:, 0:1] / self.L) * np.cos(np.pi * x[:, 1:2] / self.L)

        self.add_bcs([
            {
                'component': 0,
                'function': ic_func_u,
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 1,
                'function': ic_func_v,
                'bc': boundary_ic,
                'type': 'ic'
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 0,
                'bc': boundary_xb,
            },
            {
                'component': 0,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
            {
                'component': 1,
                'type': 'periodic',
                'component_x': 1,
                'bc': boundary_yb,
            },
        ])

    def set_viscosity(self, viscosity: float):
        """Set viscosity parameter."""
        self.current_viscosity = viscosity
        self._setup_pde()

    def generate_reference_solution(self, viscosity: float, n_points: int = 10000) -> np.ndarray:
        """Generate reference solution (simplified approach).
        
        Args:
            viscosity: Viscosity parameter
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 5) with [x, y, t, u, v] values
        """
        # Generate random points in domain
        x = np.random.uniform(0, self.L, n_points)
        y = np.random.uniform(0, self.L, n_points)
        t = np.random.uniform(0, self.T, n_points)
        
        # Simple analytical approximation (replace with numerical solution if needed)
        u = np.sin(np.pi * x / self.L) * np.sin(np.pi * y / self.L) * np.exp(-2 * viscosity * np.pi**2 * t / self.L**2)
        v = np.cos(np.pi * x / self.L) * np.cos(np.pi * y / self.L) * np.exp(-2 * viscosity * np.pi**2 * t / self.L**2)
        
        return np.column_stack([x, y, t, u, v])

    def create_task_data(self, viscosity: float,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create task data for specific viscosity."""
        # Generate reference solution
        ref_data = self.generate_reference_solution(viscosity, n_support + n_query)
        
        # Split into support and query
        support_indices = np.random.choice(len(ref_data), n_support, replace=False)
        query_indices = np.setdiff1d(np.arange(len(ref_data)), support_indices)[:n_query]
        
        # Create support data
        support_inputs = torch.tensor(ref_data[support_indices, :3], dtype=torch.float32)
        support_outputs = torch.tensor(ref_data[support_indices, 3:5], dtype=torch.float32)
        
        # Create query data
        query_inputs = torch.tensor(ref_data[query_indices, :3], dtype=torch.float32)
        query_outputs = torch.tensor(ref_data[query_indices, 3:5], dtype=torch.float32)
        
        # Generate collocation points
        x_col = np.random.uniform(0, self.L, n_collocation)
        y_col = np.random.uniform(0, self.L, n_collocation)
        t_col = np.random.uniform(0, self.T, n_collocation)
        collocation_points = torch.tensor(
            np.column_stack([x_col, y_col, t_col]), 
            dtype=torch.float32
        )
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'viscosity': viscosity}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'viscosity': viscosity}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying viscosity."""
        tasks = []
        
        for i in range(n_tasks):
            # Sample viscosity
            viscosity = np.random.uniform(*self.viscosity_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                viscosity,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='burgers_2d',
                parameters={'viscosity': viscosity},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'L': self.L,
                    'T': self.T,
                    'task_id': i
                }
            )
            
            tasks.append(task)
        
        return tasks

    def generate_val_tasks(self, n_tasks: int) -> List[Task]:
        """Generate validation tasks."""
        np.random.seed(42)
        tasks = self.generate_train_tasks(n_tasks)
        np.random.seed()
        return tasks

    def generate_test_tasks(self, n_tasks: int) -> List[Task]:
        """Generate test tasks."""
        np.random.seed(123)
        tasks = self.generate_train_tasks(n_tasks)
        np.random.seed()
        return tasks


class BurgersTaskDistribution(TaskDistribution):
    """Task distribution for parametric Burgers equations."""
    
    def __init__(self, problem_variant: str = '1d', **kwargs):
        """Initialize Burgers task distribution.
        
        Args:
            problem_variant: '1d' or '2d' Burgers equation variant
            **kwargs: Additional parameters for the specific variant
        """
        if problem_variant == '1d':
            super().__init__('burgers_1d', {'viscosity': (0.001, 0.1), 'ic_variant': (0, 4)})
            self.pde_class = ParametricBurgers1D
        elif problem_variant == '2d':
            super().__init__('burgers_2d', {'viscosity': (0.001, 0.01)})
            self.pde_class = ParametricBurgers2D
        else:
            raise ValueError(f"Unknown problem variant: {problem_variant}")
        
        self.pde_instance = self.pde_class(**kwargs)
    
    def create_task_batch(self, n_tasks: int) -> List[Task]:
        """Create batch of Burgers equation tasks."""
        return self.pde_instance.generate_train_tasks(n_tasks)
    
    def create_test_tasks(self, n_tasks: int) -> List[Task]:
        """Create test tasks."""
        return self.pde_instance.generate_test_tasks(n_tasks)