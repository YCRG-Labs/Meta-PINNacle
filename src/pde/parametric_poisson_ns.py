"""Parametric Poisson and Navier-Stokes equation tasks for meta-learning."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

import deepxde as dde
from . import baseclass
from ..meta_learning.task import Task, TaskData, TaskDistribution


class ParametricPoisson2D_Classic(baseclass.BasePDE):
    """Parametric 2D Poisson equation with varying boundary conditions and source terms.
    
    Extends Poisson2D_Classic to generate tasks with different boundary conditions
    and source terms for few-shot learning scenarios.
    """

    def __init__(self, 
                 scale=1,
                 boundary_value_range=(0.5, 2.0),
                 source_strength_range=(0.1, 5.0),
                 n_support_points=100,
                 n_query_points=2000,
                 n_collocation_points=4000):
        """Initialize parametric 2D Poisson equation.
        
        Args:
            scale: Domain scale factor
            boundary_value_range: Range of boundary values
            source_strength_range: Range of source term strengths
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.scale = scale
        self.boundary_value_range = boundary_value_range
        self.source_strength_range = source_strength_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension
        self.output_dim = 1
        
        # Domain setup
        self.bbox = [-scale / 2, scale / 2, -scale / 2, scale / 2]
        self.geom = dde.geometry.Rectangle(xmin=[-scale / 2, -scale / 2], 
                                          xmax=[scale / 2, scale / 2])
        
        # Create holes (circles) in the domain
        circ = np.array([[0.3, 0.3, 0.1], [-0.3, 0.3, 0.1], 
                        [0.3, -0.3, 0.1], [-0.3, -0.3, 0.1]]) * scale
        for c in circ:
            disk = dde.geometry.Disk(c[0:2], c[2])
            self.geom = dde.geometry.CSGDifference(self.geom, disk)
        
        # Current parameters (will be set for each task)
        self.current_boundary_value = 1.0
        self.current_source_strength = 1.0
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points()

    def _setup_pde(self):
        """Setup Poisson equation PDE with current source strength."""
        def poisson_pde(x, u):
            """Poisson equation: ∇²u = -f"""
            u_xx = dde.grad.hessian(u, x, i=0, j=0)
            u_yy = dde.grad.hessian(u, x, i=1, j=1)
            
            # Source term with varying strength
            def f(x):
                return self.current_source_strength * torch.sin(np.pi * x[:, 0:1]) * \
                       torch.sin(np.pi * x[:, 1:2])
            
            return [u_xx + u_yy + f(x)]

        self.pde = poisson_pde
        self.set_pdeloss(num=1)

    def _setup_boundary_conditions(self):
        """Setup boundary conditions with current boundary value."""
        def rec_boundary(x, on_boundary):
            return on_boundary and (
                np.isclose(x[0], self.bbox[0]) or np.isclose(x[0], self.bbox[1]) or 
                np.isclose(x[1], self.bbox[2]) or np.isclose(x[1], self.bbox[3])
            )

        def circ_boundary(x, on_boundary):
            return on_boundary and not rec_boundary(x, on_boundary)

        self.add_bcs([{
            'component': 0,
            'function': (lambda _: self.current_boundary_value),
            'bc': rec_boundary,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': circ_boundary,
            'type': 'dirichlet'
        }])

    def set_parameters(self, boundary_value: float, source_strength: float):
        """Set parameters for current task.
        
        Args:
            boundary_value: Boundary condition value
            source_strength: Source term strength
        """
        self.current_boundary_value = boundary_value
        self.current_source_strength = source_strength
        self._setup_pde()
        self._setup_boundary_conditions()

    def generate_reference_solution(self, boundary_value: float, source_strength: float,
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution using analytical approximation.
        
        Args:
            boundary_value: Boundary condition value
            source_strength: Source term strength
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 3) with [x, y, u] values
        """
        # Generate random points in domain (avoiding holes)
        x_points = []
        y_points = []
        u_points = []
        
        # Simple rejection sampling to avoid holes
        while len(x_points) < n_points:
            x = np.random.uniform(self.bbox[0], self.bbox[1])
            y = np.random.uniform(self.bbox[2], self.bbox[3])
            
            # Check if point is not in any hole
            in_hole = False
            circ = np.array([[0.3, 0.3, 0.1], [-0.3, 0.3, 0.1], 
                            [0.3, -0.3, 0.1], [-0.3, -0.3, 0.1]]) * self.scale
            for c in circ:
                if (x - c[0])**2 + (y - c[1])**2 <= c[2]**2:
                    in_hole = True
                    break
            
            if not in_hole:
                x_points.append(x)
                y_points.append(y)
                
                # Simple analytical approximation
                u = boundary_value * (1 - np.exp(-((x**2 + y**2) / self.scale**2))) + \
                    source_strength * 0.1 * np.sin(np.pi * x) * np.sin(np.pi * y)
                u_points.append(u)
        
        return np.column_stack([x_points, y_points, u_points])

    def create_task_data(self, boundary_value: float, source_strength: float,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for specific parameters."""
        # Generate reference solution
        ref_data = self.generate_reference_solution(boundary_value, source_strength, 
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
        
        # Generate collocation points (avoiding holes)
        collocation_points = []
        while len(collocation_points) < n_collocation:
            x = np.random.uniform(self.bbox[0], self.bbox[1])
            y = np.random.uniform(self.bbox[2], self.bbox[3])
            
            # Check if point is not in any hole
            in_hole = False
            circ = np.array([[0.3, 0.3, 0.1], [-0.3, 0.3, 0.1], 
                            [0.3, -0.3, 0.1], [-0.3, -0.3, 0.1]]) * self.scale
            for c in circ:
                if (x - c[0])**2 + (y - c[1])**2 <= c[2]**2:
                    in_hole = True
                    break
            
            if not in_hole:
                collocation_points.append([x, y])
        
        collocation_points = torch.tensor(collocation_points, dtype=torch.float32)
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'boundary_value': boundary_value, 'source_strength': source_strength}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'boundary_value': boundary_value, 'source_strength': source_strength}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying boundary conditions and source terms."""
        tasks = []
        
        for i in range(n_tasks):
            # Sample parameters
            boundary_value = np.random.uniform(*self.boundary_value_range)
            source_strength = np.random.uniform(*self.source_strength_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                boundary_value, source_strength,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='poisson_2d_classic',
                parameters={'boundary_value': boundary_value, 'source_strength': source_strength},
                support_data=support_data,
                query_data=query_data,
                metadata={
                    'scale': self.scale,
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


class ParametricNS2D_LidDriven(baseclass.BasePDE):
    """Parametric 2D lid-driven cavity flow with varying Reynolds numbers.
    
    Extends NS2D_LidDriven to generate tasks with different Reynolds numbers
    for few-shot learning scenarios.
    """

    def __init__(self, 
                 bbox=[0, 1, 0, 1],
                 reynolds_range=(100, 10000),
                 lid_velocity_range=(1, 8),
                 n_support_points=150,
                 n_query_points=3000,
                 n_collocation_points=6000):
        """Initialize parametric lid-driven cavity flow.
        
        Args:
            bbox: Domain bounding box [x_min, x_max, y_min, y_max]
            reynolds_range: Range of Reynolds numbers Re ∈ [Re_min, Re_max]
            lid_velocity_range: Range of lid velocity parameters
            n_support_points: Number of support points
            n_query_points: Number of query points
            n_collocation_points: Number of collocation points
        """
        super().__init__()
        
        # Store parameters
        self.bbox = bbox
        self.reynolds_range = reynolds_range
        self.lid_velocity_range = lid_velocity_range
        self.n_support_points = n_support_points
        self.n_query_points = n_query_points
        self.n_collocation_points = n_collocation_points
        
        # Output dimension (u, v, p)
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        
        # Geometry setup
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], 
                                          xmax=[bbox[1], bbox[3]])
        
        # Current parameters (will be set for each task)
        self.current_reynolds = 1000
        self.current_lid_velocity = 4
        self.current_nu = 1 / self.current_reynolds  # Viscosity
        
        self._setup_pde()
        self._setup_boundary_conditions()
        
        # Training configuration
        self.training_points()

    def _setup_pde(self):
        """Setup Navier-Stokes equations with current Reynolds number."""
        def ns_pde(x, u):
            """Navier-Stokes equations for incompressible flow."""
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            
            # Velocity gradients
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            # Pressure gradients
            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            # Momentum equations
            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - 
                         self.current_nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - 
                         self.current_nu * (v_vel_xx + v_vel_yy))
            
            # Continuity equation
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

    def _setup_boundary_conditions(self):
        """Setup boundary conditions with current lid velocity."""
        def boundary_top(x, on_boundary):
            return on_boundary and np.isclose(x[1], self.bbox[3])

        def boundary_not_top(x, on_boundary):
            return on_boundary and not np.isclose(x[1], self.bbox[3])

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: self.current_lid_velocity * x[:, 0:1] * (1 - x[:, 0:1])),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 2,
            "points": np.array([[0, 0]]),
            'values': np.array([[0]]),
            'type': 'pointset'
        }])

    def set_parameters(self, reynolds: float, lid_velocity: float):
        """Set parameters for current task.
        
        Args:
            reynolds: Reynolds number
            lid_velocity: Lid velocity parameter
        """
        self.current_reynolds = reynolds
        self.current_lid_velocity = lid_velocity
        self.current_nu = 1 / reynolds
        self._setup_pde()
        self._setup_boundary_conditions()

    def generate_reference_solution(self, reynolds: float, lid_velocity: float,
                                   n_points: int = 10000) -> np.ndarray:
        """Generate reference solution using analytical approximation.
        
        Args:
            reynolds: Reynolds number
            lid_velocity: Lid velocity parameter
            n_points: Number of points to generate
            
        Returns:
            Array of shape (n_points, 5) with [x, y, u, v, p] values
        """
        # Generate random points in domain
        x = np.random.uniform(self.bbox[0], self.bbox[1], n_points)
        y = np.random.uniform(self.bbox[2], self.bbox[3], n_points)
        
        # Simple analytical approximation for lid-driven cavity
        # This is a rough approximation - in practice, you'd use numerical solution
        u = lid_velocity * y * (1 - y) * np.sin(np.pi * x) * np.exp(-y / np.sqrt(reynolds))
        v = -0.1 * lid_velocity * x * (1 - x) * np.cos(np.pi * y) * np.exp(-x / np.sqrt(reynolds))
        p = 0.1 * lid_velocity**2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y) / reynolds
        
        return np.column_stack([x, y, u, v, p])

    def create_task_data(self, reynolds: float, lid_velocity: float,
                        n_support: int, n_query: int, n_collocation: int) -> Tuple[TaskData, TaskData]:
        """Create support and query data for specific parameters."""
        # Generate reference solution
        ref_data = self.generate_reference_solution(reynolds, lid_velocity, 
                                                   n_support + n_query)
        
        # Split into support and query
        support_indices = np.random.choice(len(ref_data), n_support, replace=False)
        query_indices = np.setdiff1d(np.arange(len(ref_data)), support_indices)[:n_query]
        
        # Create support data
        support_inputs = torch.tensor(ref_data[support_indices, :2], dtype=torch.float32)
        support_outputs = torch.tensor(ref_data[support_indices, 2:5], dtype=torch.float32)
        
        # Create query data
        query_inputs = torch.tensor(ref_data[query_indices, :2], dtype=torch.float32)
        query_outputs = torch.tensor(ref_data[query_indices, 2:5], dtype=torch.float32)
        
        # Generate collocation points
        x_col = np.random.uniform(self.bbox[0], self.bbox[1], n_collocation)
        y_col = np.random.uniform(self.bbox[2], self.bbox[3], n_collocation)
        collocation_points = torch.tensor(np.column_stack([x_col, y_col]), dtype=torch.float32)
        
        support_data = TaskData(
            inputs=support_inputs,
            outputs=support_outputs,
            collocation_points=collocation_points,
            metadata={'reynolds': reynolds, 'lid_velocity': lid_velocity}
        )
        
        query_data = TaskData(
            inputs=query_inputs,
            outputs=query_outputs,
            collocation_points=collocation_points,
            metadata={'reynolds': reynolds, 'lid_velocity': lid_velocity}
        )
        
        return support_data, query_data

    def generate_train_tasks(self, n_tasks: int) -> List[Task]:
        """Generate training tasks with varying Reynolds numbers."""
        tasks = []
        
        for i in range(n_tasks):
            # Sample parameters
            reynolds = np.random.uniform(*self.reynolds_range)
            lid_velocity = np.random.uniform(*self.lid_velocity_range)
            
            # Create task data
            support_data, query_data = self.create_task_data(
                reynolds, lid_velocity,
                self.n_support_points, 
                self.n_query_points, 
                self.n_collocation_points
            )
            
            # Create task
            task = Task(
                problem_type='ns_2d_lid_driven',
                parameters={'reynolds': reynolds, 'lid_velocity': lid_velocity},
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


class PoissonTaskDistribution(TaskDistribution):
    """Task distribution for parametric Poisson equations."""
    
    def __init__(self, **kwargs):
        """Initialize Poisson task distribution."""
        super().__init__('poisson_2d_classic', {
            'boundary_value': (0.5, 2.0),
            'source_strength': (0.1, 5.0)
        })
        self.pde_instance = ParametricPoisson2D_Classic(**kwargs)
    
    def create_task_batch(self, n_tasks: int) -> List[Task]:
        """Create batch of Poisson equation tasks."""
        return self.pde_instance.generate_train_tasks(n_tasks)
    
    def create_test_tasks(self, n_tasks: int) -> List[Task]:
        """Create test tasks."""
        return self.pde_instance.generate_test_tasks(n_tasks)


class NavierStokesTaskDistribution(TaskDistribution):
    """Task distribution for parametric Navier-Stokes equations."""
    
    def __init__(self, **kwargs):
        """Initialize Navier-Stokes task distribution."""
        super().__init__('ns_2d_lid_driven', {
            'reynolds': (100, 10000),
            'lid_velocity': (1, 8)
        })
        self.pde_instance = ParametricNS2D_LidDriven(**kwargs)
    
    def create_task_batch(self, n_tasks: int) -> List[Task]:
        """Create batch of Navier-Stokes equation tasks."""
        return self.pde_instance.generate_train_tasks(n_tasks)
    
    def create_test_tasks(self, n_tasks: int) -> List[Task]:
        """Create test tasks."""
        return self.pde_instance.generate_test_tasks(n_tasks)