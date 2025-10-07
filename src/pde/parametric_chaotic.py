"""Parametric chaotic PDE tasks for meta-learning."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

import deepxde as dde
from . import baseclass
from ..meta_learning.task import Task, TaskData, TaskDistribution


class ParametricGrayScottEquation(baseclass.BaseTimePDE):
    """Parametric Gray-Scott reaction-diffusion equation with varying reaction kinetics.
    
    Extends GrayScottEquation to generate tasks with different reaction rates
    and diffusion coefficients for few-shot learning scenarios.
    """

    def __init__(self, 
                 bbox=[-1, 1, -1, 1, 0, 200],
                 b_range=(0.02, 0.08),  # Feed rate range
                 d_range=(0.05, 0.15),  # Kill rate range
                 epsilon_u_range=(5e-6, 2e-5),  # Diffusion coefficient for u
                 epsilon_v_range=(2e-6, 1e-5),  # Diffusion coefficient for v
                 n_support_points=200,
                 n_query_points=4000,
                 n_collocation_points=8000):
        """Initialize parametric Gray-Scott equation.
        
        Args:
            bbox: Domain bounding box [x_min, x_max, y_min, y_max, t_min, t_max]
            b_range: Range of feed rate parameter b
            d_range: Range of kill rate parameter d
            epsilon_u_range: Range of diffusion coefficient for species u
            epsilon_v_range: Range of diffusion coefficient for species v
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.bbox = bbox
        self.b_range = b_range
        self.d_range = d_range
        self.epsilon_u_range = epsilon_u_range
        self.epsilon_v_range = epsilon_v_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension (u and v species)
        self.output_dim = 2
        
        # Geometry setup
        self.geom = dde.geometry.Rectangle((self.bbox[0], self.bbox[2]), 
                                          (self.bbox[1], self.bbox[3]))
        timedomain = dde.geometry.TimeDomain(self.bbox[4], self.bbox[5])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        
        # Current parameters (will be set for each task)
        self.current_b = 0.04
        self.current_d = 0.1
        self.current_epsilon = (1e-5, 5e-6)
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points(mul=4)

    def _setup_pde(self):
        """Setup Gray-Scott reaction-diffusion PDE with current parameters."""
        def gray_scott_pde(x, y):
            """Gray-Scott reaction-diffusion system."""
            u, v = y[:, 0:1], y[:, 1:2]

            # Time derivatives
            u_t = dde.grad.jacobian(u, x, i=0, j=2)
            v_t = dde.grad.jacobian(v, x, i=0, j=2)
            
            # Spatial derivatives (Laplacian)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            v_xx = dde.grad.hessian(v, x, i=0, j=0)
            v_yy = dde.grad.hessian(v, x, i=1, j=1)

            # Gray-Scott equations:
            # ∂u/∂t = ε_u ∇²u + b(1-u) - uv²
            # ∂v/∂t = ε_v ∇²v - dv + uv²
            pde_u = (u_t - (self.current_epsilon[0] * (u_xx + u_yy) + 
                           self.current_b * (1 - u) - u * (v**2)))
            pde_v = (v_t - (self.current_epsilon[1] * (v_xx + v_yy) - 
                           self.current_d * v + u * (v**2)))
            
            return [pde_u, pde_v]

        self.pde = gray_scott_pde
        self.set_pdeloss(num=2)

    def _setup_boundary_conditions(self):
        """Setup initial conditions."""
        def boundary_ic(x, on_initial):
            return on_initial and np.isclose(x[2], self.bbox[4])

        def ic_func(x, component):
            """Initial conditions with slight variations."""
            if component == 0:
                # Species u: starts near 1, with localized depletion
                return 1 - np.exp(-80 * ((x[:, 0] + 0.05)**2 + (x[:, 1] + 0.02)**2))
            else:
                # Species v: starts near 0, with localized concentration
                return np.exp(-80 * ((x[:, 0] - 0.05)**2 + (x[:, 1] - 0.02)**2))

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: ic_func(x, component=0)),
            'bc': boundary_ic,
            'type': 'ic'
        }, {
            'component': 1,
            'function': (lambda x: ic_func(x, component=1)),
            'bc': boundary_ic,
            'type': 'ic'
        }])

    def set_parameters(self, b: float, d: float, epsilon_u: float, epsilon_v: float):
        """Set parameters for current task.
        
        Args:
            b: Feed rate parameter
            d: Kill rate parameter
            epsilon_u: Diffusion coefficient for species u
            epsilon_v: Diffusion coefficient for species v
        """
        self.current_b = b
        self.current_d = d
        self.current_epsilon = (epsilon_u, epsilon_v)
        self._setup_pde()

    def generate_reference_solution(self, b: float, d: float, epsilon_u: float, epsilon_v: float,
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution using simplified dynamics.
        
        Args:
            b: Feed rate parameter
            d: Kill rate parameter
            epsilon_u: Diffusion coefficient for species u
            epsilon_v: Diffusion coefficient for species v
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 5) with [x, y, t, u, v] values
        """
        # Generate random points in domain
        x = np.random.uniform(self.bbox[0], self.bbox[1], n_points)
        y = np.random.uniform(self.bbox[2], self.bbox[3], n_points)
        t = np.random.uniform(self.bbox[4], self.bbox[5], n_points)
        
        # Simplified analytical approximation for Gray-Scott dynamics
        # This is a rough approximation - in practice, you'd use numerical integration
        
        # Initial conditions
        u0 = 1 - np.exp(-80 * ((x + 0.05)**2 + (y + 0.02)**2))
        v0 = np.exp(-80 * ((x - 0.05)**2 + (y - 0.02)**2))
        
        # Time evolution (simplified)
        decay_u = np.exp(-b * t / 10)  # Simplified decay
        growth_v = 1 - np.exp(-d * t / 20)  # Simplified growth
        
        # Spatial diffusion effects
        diffusion_u = np.exp(-epsilon_u * 1000 * t * (x**2 + y**2))
        diffusion_v = np.exp(-epsilon_v * 1000 * t * (x**2 + y**2))
        
        u = u0 * decay_u * diffusion_u + 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-t/50)
        v = v0 * growth_v * diffusion_v + 0.05 * np.cos(np.pi * x) * np.cos(np.pi * y) * (1 - np.exp(-t/100))
        
        # Ensure physical bounds
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        
        return np.column_stack([x, y, t, u, v])

    def create_task_data(self, b: float, d: float, epsilon_u: float, epsilon_v: float,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for specific parameters."""
        # Generate reference solution
        ref_data = self.generate_reference_solution(b, d, epsilon_u, epsilon_v, 
                                                   n_support + n_query)
        
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
        x_col = np.random.uniform(self.bbox[0], self.bbox[1], n_collocation)
        y_col = np.random.uniform(self.bbox[2], self.bbox[3], n_collocation)
        t_col = np.random.uniform(self.bbox[4], self.bbox[5], n_collocation)
        collocation_points = torch.tensor(
            np.column_stack([x_col, y_col, t_col]), 
            dtype=torch.float32
        )
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'b': b, 'd': d, 'epsilon_u': epsilon_u, 'epsilon_v': epsilon_v}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'b': b, 'd': d, 'epsilon_u': epsilon_u, 'epsilon_v': epsilon_v}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying reaction kinetics."""
        tasks = []
        
        for i in range(n_tasks):
            # Sample parameters
            b = np.random.uniform(*self.b_range)
            d = np.random.uniform(*self.d_range)
            epsilon_u = np.random.uniform(*self.epsilon_u_range)
            epsilon_v = np.random.uniform(*self.epsilon_v_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                b, d, epsilon_u, epsilon_v,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='gray_scott',
                parameters={
                    'b': b, 'd': d, 
                    'epsilon_u': epsilon_u, 'epsilon_v': epsilon_v
                },
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'bbox': self.bbox,
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


class ParametricKuramotoSivashinskyEquation(baseclass.BaseTimePDE):
    """Parametric Kuramoto-Sivashinsky equation with varying coefficients.
    
    Extends KuramotoSivashinskyEquation to generate tasks with different
    nonlinear and diffusion coefficients for few-shot learning scenarios.
    """

    def __init__(self, 
                 bbox=[0, 2 * np.pi, 0, 1],
                 alpha_range=(50/16, 200/16),  # Nonlinear coefficient range
                 beta_range=(50/(16*16), 200/(16*16)),  # Second-order diffusion range
                 gamma_range=(50/(16**4), 200/(16**4)),  # Fourth-order diffusion range
                 n_support_points=100,
                 n_query_points=2000,
                 n_collocation_points=4000):
        """Initialize parametric Kuramoto-Sivashinsky equation.
        
        Args:
            bbox: Domain bounding box [x_min, x_max, t_min, t_max]
            alpha_range: Range of nonlinear coefficient α
            beta_range: Range of second-order diffusion coefficient β
            gamma_range: Range of fourth-order diffusion coefficient γ
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.bbox = bbox
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.gamma_range = gamma_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension
        self.output_dim = 1
        
        # Geometry setup
        self.geom = dde.geometry.Interval(bbox[0], bbox[1])
        timedomain = dde.geometry.TimeDomain(bbox[2], bbox[3])
        self.geomtime = dde.geometry.GeometryXTime(self.geom, timedomain)
        
        # Current parameters (will be set for each task)
        self.current_alpha = 100 / 16
        self.current_beta = 100 / (16 * 16)
        self.current_gamma = 100 / (16**4)
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points()

    def _setup_pde(self):
        """Setup Kuramoto-Sivashinsky PDE with current parameters."""
        def ks_pde(x, u):
            """Kuramoto-Sivashinsky equation: ∂u/∂t + α u ∂u/∂x + β ∂²u/∂x² + γ ∂⁴u/∂x⁴ = 0"""
            u_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_t = dde.grad.jacobian(u, x, i=0, j=1)
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_xxxx = dde.grad.hessian(u_xx, x, i=0, j=0)

            return (u_t + self.current_alpha * u * u_x + 
                   self.current_beta * u_xx + self.current_gamma * u_xxxx)

        self.pde = ks_pde
        self.set_pdeloss(num=1)

    def _setup_boundary_conditions(self):
        """Setup initial conditions."""
        def ic_func(x):
            """Initial condition with periodic structure."""
            return np.cos(x[:, 0:1]) * (1 + np.sin(x[:, 0:1]))

        self.add_bcs([{
            'component': 0,
            'function': ic_func,
            'bc': (lambda _, on_initial: on_initial),
            'type': 'ic'
        }])

    def set_parameters(self, alpha: float, beta: float, gamma: float):
        """Set parameters for current task.
        
        Args:
            alpha: Nonlinear coefficient
            beta: Second-order diffusion coefficient
            gamma: Fourth-order diffusion coefficient
        """
        self.current_alpha = alpha
        self.current_beta = beta
        self.current_gamma = gamma
        self._setup_pde()

    def generate_reference_solution(self, alpha: float, beta: float, gamma: float,
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution using analytical approximation.
        
        Args:
            alpha: Nonlinear coefficient
            beta: Second-order diffusion coefficient
            gamma: Fourth-order diffusion coefficient
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) with [x, t, u] values
        """
        # Generate random points in domain
        x = np.random.uniform(self.bbox[0], self.bbox[1], n_points)
        t = np.random.uniform(self.bbox[2], self.bbox[3], n_points)
        
        # Simplified analytical approximation
        # This is a rough approximation - the KS equation is chaotic and complex
        
        # Initial condition
        u0 = np.cos(x) * (1 + np.sin(x))
        
        # Time evolution with parameter-dependent dynamics
        # Nonlinear term effect
        nonlinear_effect = np.exp(-alpha * t / 10) * np.sin(2 * x)
        
        # Diffusion effects
        diffusion_effect = np.exp(-beta * t) * np.cos(x)
        hyperdiffusion_effect = np.exp(-gamma * 1000 * t) * np.sin(4 * x)
        
        # Combined solution (simplified)
        u = (u0 * np.exp(-t/2) + 
             0.1 * nonlinear_effect + 
             0.05 * diffusion_effect + 
             0.02 * hyperdiffusion_effect)
        
        return np.column_stack([x, t, u])

    def create_task_data(self, alpha: float, beta: float, gamma: float,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for specific parameters."""
        # Generate reference solution
        ref_data = self.generate_reference_solution(alpha, beta, gamma, 
                                                   n_support + n_query)
        
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
        x_col = np.random.uniform(self.bbox[0], self.bbox[1], n_collocation)
        t_col = np.random.uniform(self.bbox[2], self.bbox[3], n_collocation)
        collocation_points = torch.tensor(np.column_stack([x_col, t_col]), dtype=torch.float32)
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'alpha': alpha, 'beta': beta, 'gamma': gamma}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'alpha': alpha, 'beta': beta, 'gamma': gamma}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying coefficients."""
        tasks = []
        
        for i in range(n_tasks):
            # Sample parameters
            alpha = np.random.uniform(*self.alpha_range)
            beta = np.random.uniform(*self.beta_range)
            gamma = np.random.uniform(*self.gamma_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                alpha, beta, gamma,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='kuramoto_sivashinsky',
                parameters={'alpha': alpha, 'beta': beta, 'gamma': gamma},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'bbox': self.bbox,
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


class ChaoticPDETaskDistribution(TaskDistribution):
    """Task distribution for parametric chaotic PDEs."""
    
    def __init__(self, problem_variant: str = 'gray_scott', **kwargs):
        """Initialize chaotic PDE task distribution.
        
        Args:
            problem_variant: 'gray_scott' or 'kuramoto_sivashinsky'
            **kwargs: Additional parameters for the specific variant
        """
        if problem_variant == 'gray_scott':
            super().__init__('gray_scott', {
                'b': (0.02, 0.08),
                'd': (0.05, 0.15),
                'epsilon_u': (5e-6, 2e-5),
                'epsilon_v': (2e-6, 1e-5)
            })
            self.pde_class = ParametricGrayScottEquation
        elif problem_variant == 'kuramoto_sivashinsky':
            super().__init__('kuramoto_sivashinsky', {
                'alpha': (50/16, 200/16),
                'beta': (50/(16*16), 200/(16*16)),
                'gamma': (50/(16**4), 200/(16**4))
            })
            self.pde_class = ParametricKuramotoSivashinskyEquation
        else:
            raise ValueError(f"Unknown problem variant: {problem_variant}")
        
        self.pde_instance = self.pde_class(**kwargs)
    
    def create_task_batch(self, n_tasks: int) -> List[Task]:
        """Create batch of chaotic PDE tasks."""
        return self.pde_instance.generate_train_tasks(n_tasks)
    
    def create_test_tasks(self, n_tasks: int) -> List[Task]:
        """Create test tasks."""
        return self.pde_instance.generate_test_tasks(n_tasks)
    
    def create_task_distributions(self) -> Dict[str, List[Task]]:
        """Create complete task distributions with specified numbers.
        
        Returns:
            Dictionary with 'train', 'val', 'test' task lists
        """
        return {
            'train': self.pde_instance.generate_train_tasks(100),  # 100 train tasks
            'val': self.pde_instance.generate_val_tasks(20),      # 20 val tasks  
            'test': self.pde_instance.generate_test_tasks(50)     # 50 test tasks
        }