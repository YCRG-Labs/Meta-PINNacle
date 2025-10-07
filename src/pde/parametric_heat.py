"""Parametric Heat equation tasks for meta-learning."""

import numpy as np
import torch
from scipy import interpolate
from typing import List, Dict, Tuple, Optional

import deepxde as dde
from . import baseclass
from ..meta_learning.task import Task, TaskData, TaskDistribution
from ..utils.func_cache import cache_tensor
from ..utils.geom import CSGMultiDifference
from ..utils.random import generate_heat_2d_coef


class ParametricHeat2D_VaryingCoef(baseclass.BaseTimePDE):
    """Parametric Heat equation with varying diffusivity for meta-learning.
    
    Extends Heat2D_VaryingCoef to generate tasks with different diffusivity parameters
    for few-shot learning scenarios. Each task represents a different heat equation
    with varying thermal diffusivity coefficients.
    """

    def __init__(self, 
                 bbox=[0, 1, 0, 1, 0, 5], 
                 A=200, 
                 m=(1, 5, 1),
                 diffusivity_range=(0.1, 2.0),
                 n_support_points=50,
                 n_query_points=1000,
                 n_collocation_points=2000):
        """Initialize parametric heat equation.
        
        Args:
            bbox: Bounding box [x_min, x_max, y_min, y_max, t_min, t_max]
            A: Amplitude parameter for source term
            m: Frequency parameters for source term
            diffusivity_range: Range of diffusivity values (min, max)
            n_support_points: Number of points in support set
            n_query_points: Number of points in query set
            n_collocation_points: Number of collocation points for physics loss
        """
        super().__init__()
        
        # Store parameters
        self.bbox = bbox
        self.A = A
        self.m = m
        self.diffusivity_range = diffusivity_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension
        self.output_dim = 1
        
        # Geometry setup
        self.geom = dde.geometry.Rectangle(xmin=(self.bbox[0], self.bbox[2]), 
                                          xmax=(self.bbox[1], self.bbox[3]))
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        
        # Load base coefficient field (will be scaled by diffusivity parameter)
        self.base_coef = np.loadtxt("ref/heat_2d_coef_256.dat")
        
        # Initialize with default diffusivity (will be overridden for each task)
        self.current_diffusivity = 1.0
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points(mul=4)

    def _setup_pde(self):
        """Setup PDE with current diffusivity parameter."""
        
        @cache_tensor
        def coef(x):
            """Coefficient field scaled by current diffusivity."""
            base_values = interpolate.griddata(
                self.base_coef[:, 0:2], 
                self.base_coef[:, 2], 
                x.detach().cpu().numpy()[:, 0:2], 
                method='nearest'
            )
            return torch.Tensor(base_values * self.current_diffusivity).unsqueeze(dim=1)

        def heat_pde(x, u):
            """Heat equation PDE residual."""
            u_xx = dde.grad.hessian(u, x, i=0, j=0) + dde.grad.hessian(u, x, i=1, j=1)
            u_t = dde.grad.jacobian(u, x, i=0, j=2)

            def f(x):
                """Source term."""
                return self.A * torch.sin(self.m[0] * torch.pi * x[:, 0:1]) * \
                       torch.sin(self.m[1] * torch.pi * x[:, 1:2]) * \
                       torch.sin(self.m[2] * torch.pi * x[:, 2:3])

            return [u_t - coef(x) * u_xx - f(x)]

        self.pde = heat_pde
        self.set_pdeloss(num=1)

    def _setup_boundary_conditions(self):
        """Setup boundary and initial conditions."""
        def boundary_t0(x, on_initial):
            return on_initial and np.isclose(x[2], self.bbox[4])

        def boundary_xb(x, on_boundary):
            return on_boundary and (np.isclose(x[0], self.bbox[0]) or 
                                   np.isclose(x[0], self.bbox[1]) or 
                                   np.isclose(x[1], self.bbox[2]) or 
                                   np.isclose(x[1], self.bbox[3]))

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_t0,
            'type': 'ic'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_xb,
            'type': 'dirichlet'
        }])

    def set_diffusivity(self, diffusivity: float):
        """Set the diffusivity parameter for current task.
        
        Args:
            diffusivity: Thermal diffusivity coefficient
        """
        self.current_diffusivity = diffusivity
        # Re-setup PDE with new diffusivity
        self._setup_pde()

    def generate_reference_solution(self, diffusivity: float, 
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution for given diffusivity.
        
        This is a simplified reference solution. In practice, you might want to
        use a high-fidelity solver or analytical solution if available.
        
        Args:
            diffusivity: Diffusivity parameter
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 4) with [x, y, t, u] values
        """
        # Generate random points in domain
        x = np.random.uniform(self.bbox[0], self.bbox[1], n_points)
        y = np.random.uniform(self.bbox[2], self.bbox[3], n_points)
        t = np.random.uniform(self.bbox[4], self.bbox[5], n_points)
        
        # Simple analytical approximation (replace with actual solution if available)
        u = np.exp(-diffusivity * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
        
        return np.column_stack([x, y, t, u])

    def create_task_data(self, diffusivity: float, 
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for a specific diffusivity value.
        
        Args:
            diffusivity: Diffusivity parameter for this task
            n_support: Number of support points
            n_query: Number of query points
            n_collocation: Number of collocation points
            
        Returns:
            Tuple of (support_data, query_data)
        """
        # Generate reference solution
        ref_data = self.generate_reference_solution(diffusivity, n_support + n_query)
        
        # Split into support and query
        support_indices = np.random.choice(len(ref_data), n_support, replace=False)
        query_indices = np.setdiff1d(np.arange(len(ref_data)), support_indices)[:n_query]
        
        # Create support data
        support_inputs = torch.tensor(ref_data[support_indices, :3], dtype=torch.float32)
        support_outputs = torch.tensor(ref_data[support_indices, 3:4], dtype=torch.float32)
        
        # Create query data
        query_inputs = torch.tensor(ref_data[query_indices, :3], dtype=torch.float32)
        query_outputs = torch.tensor(ref_data[query_indices, 3:4], dtype=torch.float32)
        
        # Generate collocation points for physics loss
        collocation_x = np.random.uniform(self.bbox[0], self.bbox[1], n_collocation)
        collocation_y = np.random.uniform(self.bbox[2], self.bbox[3], n_collocation)
        collocation_t = np.random.uniform(self.bbox[4], self.bbox[5], n_collocation)
        collocation_points = torch.tensor(
            np.column_stack([collocation_x, collocation_y, collocation_t]), 
            dtype=torch.float32
        )
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'diffusivity': diffusivity}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'diffusivity': diffusivity}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying diffusivity parameters.
        
        Args:
            n_tasks: Number of training tasks to generate
            
        Returns:
            List of Task objects for training
        """
        tasks = []
        
        for i in range(n_tasks):
            # Sample diffusivity from training distribution
            diffusivity = np.random.uniform(*self.diffusivity_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                diffusivity, 
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='heat_2d_varying_coef',
                parameters={'diffusivity': diffusivity},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'bbox': self.bbox,
                    'A': self.A,
                    'm': self.m,
                    'task_id': i
                }
            )
            
            tasks.append(task)
        
        return tasks

    def generate_val_tasks(self, n_tasks: int) -> List[Task]:
        """Generate validation tasks with varying diffusivity parameters.
        
        Args:
            n_tasks: Number of validation tasks to generate
            
        Returns:
            List of Task objects for validation
        """
        # Use same distribution as training but different random seed
        np.random.seed(42)  # Fixed seed for reproducible validation
        tasks = self.generate_train_tasks(n_tasks)
        np.random.seed()  # Reset seed
        return tasks

    def generate_test_tasks(self, n_tasks: int, 
                           test_diffusivity_range: Optional[Tuple[float, float]] = None) -> List[Task]:
        """Generate test tasks, potentially with different diffusivity range.
        
        Args:
            n_tasks: Number of test tasks to generate
            test_diffusivity_range: Optional different range for test tasks
            
        Returns:
            List of Task objects for testing
        """
        if test_diffusivity_range is None:
            test_diffusivity_range = self.diffusivity_range
        
        tasks = []
        
        # Use fixed seed for reproducible test tasks
        np.random.seed(123)
        
        for i in range(n_tasks):
            # Sample diffusivity from test distribution
            diffusivity = np.random.uniform(*test_diffusivity_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                diffusivity, 
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='heat_2d_varying_coef',
                parameters={'diffusivity': diffusivity},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'bbox': self.bbox,
                    'A': self.A,
                    'm': self.m,
                    'task_id': i,
                    'test_task': True
                }
            )
            
            tasks.append(task)
        
        np.random.seed()  # Reset seed
        return tasks


class ParametricHeat1D(baseclass.BaseTimePDE):
    """1D parametric heat equation for meta-learning.
    
    Simpler 1D version for faster experimentation and debugging.
    """

    def __init__(self, 
                 geom=[-1, 1], 
                 time=[0, 1], 
                 diffusivity_range=(0.01, 0.1),
                 n_support_points=25,
                 n_query_points=500,
                 n_collocation_points=1000):
        """Initialize 1D parametric heat equation.
        
        Args:
            geom: Spatial domain [x_min, x_max]
            time: Time domain [t_min, t_max]
            diffusivity_range: Range of diffusivity values
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.geom_bounds = geom
        self.time_bounds = time
        self.diffusivity_range = diffusivity_range
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
        
        # Current diffusivity (will be set for each task)
        self.current_diffusivity = 0.01
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points()

    def _setup_pde(self):
        """Setup 1D heat equation PDE."""
        def heat_pde_1d(x, u):
            """1D heat equation: ∂u/∂t = κ ∂²u/∂x²"""
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            return u_t - self.current_diffusivity * u_xx

        self.pde = heat_pde_1d
        self.set_pdeloss(num=1)

    def _setup_boundary_conditions(self):
        """Setup boundary and initial conditions."""
        def ic_func(x):
            """Initial condition: u(x, 0) = sin(π*x)"""
            return np.sin(np.pi * x[:, 0:1])

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

    def set_diffusivity(self, diffusivity: float):
        """Set diffusivity parameter."""
        self.current_diffusivity = diffusivity
        self._setup_pde()

    def analytical_solution(self, x: np.ndarray, t: np.ndarray, diffusivity: float) -> np.ndarray:
        """Analytical solution for 1D heat equation with sin initial condition.
        
        Args:
            x: Spatial coordinates
            t: Time coordinates
            diffusivity: Diffusivity parameter
            
        Returns:
            Solution values u(x, t)
        """
        return np.sin(np.pi * x) * np.exp(-diffusivity * np.pi**2 * t)

    def create_task_data(self, diffusivity: float, 
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create task data using analytical solution."""
        # Generate random points
        x_support = np.random.uniform(self.geom_bounds[0], self.geom_bounds[1], n_support)
        t_support = np.random.uniform(self.time_bounds[0], self.time_bounds[1], n_support)
        
        x_query = np.random.uniform(self.geom_bounds[0], self.geom_bounds[1], n_query)
        t_query = np.random.uniform(self.time_bounds[0], self.time_bounds[1], n_query)
        
        # Compute analytical solutions
        u_support = self.analytical_solution(x_support, t_support, diffusivity)
        u_query = self.analytical_solution(x_query, t_query, diffusivity)
        
        # Create tensors
        support_inputs = torch.tensor(np.column_stack([x_support, t_support]), dtype=torch.float32)
        support_outputs = torch.tensor(u_support.reshape(-1, 1), dtype=torch.float32)
        
        query_inputs = torch.tensor(np.column_stack([x_query, t_query]), dtype=torch.float32)
        query_outputs = torch.tensor(u_query.reshape(-1, 1), dtype=torch.float32)
        
        # Collocation points
        x_col = np.random.uniform(self.geom_bounds[0], self.geom_bounds[1], n_collocation)
        t_col = np.random.uniform(self.time_bounds[0], self.time_bounds[1], n_collocation)
        collocation_points = torch.tensor(np.column_stack([x_col, t_col]), dtype=torch.float32)
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'diffusivity': diffusivity}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'diffusivity': diffusivity}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks."""
        tasks = []
        
        for i in range(n_tasks):
            diffusivity = np.random.uniform(*self.diffusivity_range)
            
            support_data, query_data = self.create_task_data(
                diffusivity, 
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            task = Task(
                problem_type='heat_1d',
                parameters={'diffusivity': diffusivity},
                support_data=support_data,
                query_data=query_data,
                metadata={'task_id': i}
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


class HeatTaskDistribution(TaskDistribution):
    """Task distribution for parametric heat equations."""
    
    def __init__(self, problem_variant: str = '2d', **kwargs):
        """Initialize heat task distribution.
        
        Args:
            problem_variant: '1d' or '2d' heat equation variant
            **kwargs: Additional parameters for the specific variant
        """
        if problem_variant == '1d':
            super().__init__('heat_1d', {'diffusivity': (0.01, 0.1)})
            self.pde_class = ParametricHeat1D
        elif problem_variant == '2d':
            super().__init__('heat_2d_varying_coef', {'diffusivity': (0.1, 2.0)})
            self.pde_class = ParametricHeat2D_VaryingCoef
        else:
            raise ValueError(f"Unknown problem variant: {problem_variant}")
        
        self.pde_instance = self.pde_class(**kwargs)
    
    def create_task_batch(self, n_tasks: int) -> List[Task]:
        """Create batch of heat equation tasks."""
        return self.pde_instance.generate_train_tasks(n_tasks)
    
    def create_test_tasks(self, n_tasks: int) -> List[Task]:
        """Create test tasks."""
        return self.pde_instance.generate_test_tasks(n_tasks)