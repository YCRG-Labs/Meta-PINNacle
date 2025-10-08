"""DeepONet baseline implementation for parametric PDEs.

This module implements DeepONet as a baseline method for comparison with meta-learning PINNs.
DeepONet learns operators by using separate branch and trunk networks to handle
function inputs and coordinate inputs respectively.
"""

import time
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import deepxde as dde
from deepxde.nn import DeepONetCartesianProd

from ..meta_learning.task import Task, TaskData
from ..utils.metrics import compute_l2_relative_error


class DeepONetBaseline:
    """DeepONet baseline for parametric PDE families.
    
    This class implements DeepONet as a baseline method that can be trained on
    parametric PDE families and evaluated in few-shot scenarios using fine-tuning.
    DeepONet uses separate branch and trunk networks to learn operators.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DeepONet baseline with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - branch_layers: List of layer sizes for branch network
                - trunk_layers: List of layer sizes for trunk network
                - activation: Activation function name
                - learning_rate: Learning rate for training
                - batch_size: Batch size for training
                - epochs: Number of training epochs
                - device: Device to run on ('cuda' or 'cpu')
                - num_sensors: Number of sensor points for branch network
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Network architecture configuration
        self.branch_layers = config.get('branch_layers', [100, 128, 128, 128])
        self.trunk_layers = config.get('trunk_layers', [2, 128, 128, 128])
        self.activation = config.get('activation', 'tanh')
        self.num_sensors = config.get('num_sensors', 100)
        
        # Build DeepONet model
        self.model = self._build_deeponet_model()
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 1000)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        
        # Training history
        self.training_history = {}
        
        # Sensor locations (fixed for branch network)
        self.sensor_locations = None
        
    def _build_deeponet_model(self) -> nn.Module:
        """Build DeepONet model using DeepXDE."""
        # Create DeepONet using DeepXDE's implementation
        net = DeepONetCartesianProd(
            layer_sizes_branch=self.branch_layers,
            layer_sizes_trunk=self.trunk_layers,
            activation=self.activation,
            kernel_initializer="Glorot normal"
        )
        
        return net
    
    def _setup_sensor_locations(self, tasks: List[Task]):
        """Setup fixed sensor locations for the branch network.
        
        Args:
            tasks: Training tasks to determine sensor placement
        """
        if self.sensor_locations is not None:
            return
            
        # Use the first task to determine spatial domain
        first_task = tasks[0]
        task_data = first_task.get_task_data()
        coords = task_data.x_physics
        
        # Create uniform grid of sensor locations
        if coords.shape[1] == 1:  # 1D
            x_min, x_max = coords.min(), coords.max()
            self.sensor_locations = np.linspace(x_min, x_max, self.num_sensors).reshape(-1, 1)
        else:  # 2D
            x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
            y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
            
            # Create 2D grid
            n_per_dim = int(np.sqrt(self.num_sensors))
            x_sensors = np.linspace(x_min, x_max, n_per_dim)
            y_sensors = np.linspace(y_min, y_max, n_per_dim)
            X, Y = np.meshgrid(x_sensors, y_sensors)
            self.sensor_locations = np.column_stack([X.ravel(), Y.ravel()])[:self.num_sensors]
    
    def train_on_pde_family(self, pde_family: str, train_tasks: List[Task]) -> Dict[str, Any]:
        """Train DeepONet on a specific PDE family.
        
        Args:
            pde_family: Name of the PDE family (e.g., 'heat', 'burgers')
            train_tasks: List of training tasks for this PDE family
            
        Returns:
            Dictionary containing training results and metrics
        """
        print(f"Training DeepONet on {pde_family} family with {len(train_tasks)} tasks")
        
        # Setup sensor locations
        self._setup_sensor_locations(train_tasks)
        
        # Prepare training data
        train_loader = self._prepare_training_data(train_tasks)
        
        # Training loop
        start_time = time.time()
        losses = []
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_data in train_loader:
                branch_inputs, trunk_inputs, targets = batch_data
                branch_inputs = branch_inputs.to(self.device)
                trunk_inputs = trunk_inputs.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model([branch_inputs, trunk_inputs])
                
                # Compute loss (L2 loss)
                loss = F.mse_loss(predictions, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Print progress
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Store training history
        self.training_history[pde_family] = {
            'losses': losses,
            'training_time': training_time,
            'final_loss': losses[-1] if losses else float('inf')
        }
        
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'final_loss': losses[-1] if losses else float('inf'),
            'losses': losses
        }
    
    def _prepare_training_data(self, tasks: List[Task]) -> DataLoader:
        """Prepare training data from tasks for DeepONet training.
        
        Args:
            tasks: List of training tasks
            
        Returns:
            DataLoader for training
        """
        all_branch_inputs = []
        all_trunk_inputs = []
        all_targets = []
        
        for task in tasks:
            # Get task data
            task_data = task.get_task_data()
            
            # Extract coordinates and parameters
            coords = task_data.x_physics  # Shape: [N, spatial_dim]
            params = task_data.params     # PDE parameters
            
            # Get reference solution
            if hasattr(task_data, 'u_ref') and task_data.u_ref is not None:
                solution = task_data.u_ref
            else:
                # Generate reference solution if not available
                solution = self._generate_reference_solution(task)
            
            # Prepare branch input: function values at sensor locations
            # For parametric PDEs, we use parameter values at sensor locations
            branch_input = self._create_branch_input(params, task)
            
            # Prepare trunk input: evaluation coordinates
            trunk_input = coords
            
            # For each coordinate point, we need the same branch input
            num_points = coords.shape[0]
            branch_inputs_expanded = np.tile(branch_input, (num_points, 1))
            
            all_branch_inputs.append(torch.tensor(branch_inputs_expanded, dtype=torch.float32))
            all_trunk_inputs.append(torch.tensor(trunk_input, dtype=torch.float32))
            all_targets.append(torch.tensor(solution, dtype=torch.float32))
        
        # Concatenate all data
        branch_inputs = torch.cat(all_branch_inputs, dim=0)
        trunk_inputs = torch.cat(all_trunk_inputs, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Create dataset and dataloader
        dataset = TensorDataset(branch_inputs, trunk_inputs, targets)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _create_branch_input(self, params: np.ndarray, task: Task) -> np.ndarray:
        """Create branch network input from PDE parameters.
        
        For parametric PDEs, we create a function representation using parameters.
        This could be parameter values at sensor locations or parameter-based functions.
        
        Args:
            params: PDE parameters
            task: Task object
            
        Returns:
            Branch input array [num_sensors]
        """
        # Simple approach: use parameter value at all sensor locations
        # More sophisticated approaches could use parameter-dependent functions
        if len(params) == 1:
            # Single parameter case
            branch_input = np.full(self.num_sensors, params[0])
        else:
            # Multiple parameters - repeat pattern
            branch_input = np.tile(params, self.num_sensors // len(params) + 1)[:self.num_sensors]
        
        return branch_input
    
    def _generate_reference_solution(self, task: Task) -> np.ndarray:
        """Generate reference solution for a task if not available.
        
        This is a placeholder - in practice, reference solutions should be
        pre-computed using high-fidelity numerical methods.
        """
        # For now, return zeros - this should be replaced with actual reference solutions
        task_data = task.get_task_data()
        return np.zeros((task_data.x_physics.shape[0], 1))
    
    def evaluate_few_shot(self, support_tasks: List[Task], query_tasks: List[Task], 
                         inner_steps: int = 10) -> Dict[str, Any]:
        """Evaluate few-shot performance using fine-tuning approach.
        
        Args:
            support_tasks: Support set tasks for adaptation
            query_tasks: Query set tasks for evaluation
            inner_steps: Number of fine-tuning steps
            
        Returns:
            Dictionary containing evaluation results
        """
        print(f"Evaluating few-shot with {len(support_tasks)} support tasks, {len(query_tasks)} query tasks")
        
        # Create a copy of the model for fine-tuning
        adapted_model = copy.deepcopy(self.model)
        adapted_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.learning_rate * 0.1)
        
        # Fine-tune on support tasks
        start_time = time.time()
        support_loader = self._prepare_training_data(support_tasks)
        
        adapted_model.train()
        for step in range(inner_steps):
            for batch_data in support_loader:
                branch_inputs, trunk_inputs, targets = batch_data
                branch_inputs = branch_inputs.to(self.device)
                trunk_inputs = trunk_inputs.to(self.device)
                targets = targets.to(self.device)
                
                adapted_optimizer.zero_grad()
                predictions = adapted_model([branch_inputs, trunk_inputs])
                loss = F.mse_loss(predictions, targets)
                loss.backward()
                adapted_optimizer.step()
        
        adaptation_time = time.time() - start_time
        
        # Evaluate on query tasks
        adapted_model.eval()
        query_results = []
        
        with torch.no_grad():
            for query_task in query_tasks:
                task_data = query_task.get_task_data()
                
                # Prepare inputs
                coords = task_data.x_physics
                params = task_data.params
                
                # Branch input
                branch_input = self._create_branch_input(params, query_task)
                num_points = coords.shape[0]
                branch_inputs = np.tile(branch_input, (num_points, 1))
                
                # Convert to tensors
                branch_tensor = torch.tensor(branch_inputs, dtype=torch.float32).to(self.device)
                trunk_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
                
                # Get prediction
                prediction = adapted_model([branch_tensor, trunk_tensor]).cpu().numpy()
                
                # Get reference solution
                if hasattr(task_data, 'u_ref') and task_data.u_ref is not None:
                    reference = task_data.u_ref
                else:
                    reference = self._generate_reference_solution(query_task)
                
                # Compute error
                l2_error = compute_l2_relative_error(prediction, reference)
                
                query_results.append({
                    'l2_error': l2_error,
                    'prediction': prediction,
                    'reference': reference
                })
        
        # Compute average metrics
        avg_l2_error = np.mean([result['l2_error'] for result in query_results])
        
        return {
            'avg_l2_error': avg_l2_error,
            'adaptation_time': adaptation_time,
            'query_results': query_results,
            'num_support_tasks': len(support_tasks),
            'num_query_tasks': len(query_tasks)
        }
    
    def predict(self, coordinates: np.ndarray, parameters: np.ndarray) -> np.ndarray:
        """Make predictions for given coordinates and parameters.
        
        Args:
            coordinates: Spatial coordinates [N, spatial_dim]
            parameters: PDE parameters [param_dim]
            
        Returns:
            Predictions [N, 1]
        """
        self.model.eval()
        
        with torch.no_grad():
            # Prepare branch input
            branch_input = self._create_branch_input(parameters, None)
            num_points = coordinates.shape[0]
            branch_inputs = np.tile(branch_input, (num_points, 1))
            
            # Convert to tensors
            branch_tensor = torch.tensor(branch_inputs, dtype=torch.float32).to(self.device)
            trunk_tensor = torch.tensor(coordinates, dtype=torch.float32).to(self.device)
            
            # Get prediction
            prediction = self.model([branch_tensor, trunk_tensor]).cpu().numpy()
            
        return prediction
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'sensor_locations': self.sensor_locations
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
        self.sensor_locations = checkpoint.get('sensor_locations')
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_type': 'DeepONet',
            'branch_layers': self.branch_layers,
            'trunk_layers': self.trunk_layers,
            'activation': self.activation,
            'num_sensors': self.num_sensors,
            'num_parameters': num_params,
            'device': str(self.device)
        }


def create_deeponet_baseline(pde_family: str) -> DeepONetBaseline:
    """Create DeepONet baseline with default configuration for a PDE family.
    
    Args:
        pde_family: Name of the PDE family
        
    Returns:
        Configured DeepONetBaseline instance
    """
    # Default configurations for different PDE families
    default_configs = {
        'heat': {
            'branch_layers': [100, 128, 128, 128],
            'trunk_layers': [2, 128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 1000,
            'num_sensors': 100
        },
        'burgers': {
            'branch_layers': [100, 128, 128, 128],
            'trunk_layers': [2, 128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 1000,
            'num_sensors': 100
        },
        'poisson': {
            'branch_layers': [100, 128, 128, 128],
            'trunk_layers': [2, 128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 800,
            'num_sensors': 100
        },
        'navier_stokes': {
            'branch_layers': [100, 256, 256, 256],
            'trunk_layers': [2, 256, 256, 256],
            'activation': 'tanh',
            'learning_rate': 5e-4,
            'batch_size': 16,
            'epochs': 1500,
            'num_sensors': 150
        },
        'gray_scott': {
            'branch_layers': [100, 256, 256, 256],
            'trunk_layers': [2, 256, 256, 256],
            'activation': 'tanh',
            'learning_rate': 5e-4,
            'batch_size': 16,
            'epochs': 1500,
            'num_sensors': 150
        },
        'kuramoto_sivashinsky': {
            'branch_layers': [100, 256, 256, 256],
            'trunk_layers': [2, 256, 256, 256],
            'activation': 'tanh',
            'learning_rate': 5e-4,
            'batch_size': 16,
            'epochs': 1500,
            'num_sensors': 150
        },
        'darcy': {
            'branch_layers': [100, 128, 128, 128],
            'trunk_layers': [2, 128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'batch_size': 32,
            'epochs': 1000,
            'num_sensors': 100
        }
    }
    
    # Get configuration for the PDE family
    config = default_configs.get(pde_family, default_configs['heat']).copy()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return DeepONetBaseline(config)