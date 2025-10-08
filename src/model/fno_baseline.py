"""Fourier Neural Operator (FNO) baseline implementation for parametric PDEs.

This module implements FNO as a baseline method for comparison with meta-learning PINNs.
FNO is designed to learn solution operators for parametric PDEs by learning mappings
between function spaces using Fourier transforms.
"""

import time
import copy
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from neuraloperator import FNO2d, FNO1d
    NEURALOPERATOR_AVAILABLE = True
except ImportError:
    NEURALOPERATOR_AVAILABLE = False
    print("Warning: neuraloperator library not available. FNO baseline will not work.")

from ..meta_learning.task import Task, TaskData
from ..utils.metrics import compute_l2_relative_error


class FNOBaseline:
    """Fourier Neural Operator baseline for parametric PDE families.
    
    This class implements FNO as a baseline method that can be trained on
    parametric PDE families and evaluated in few-shot scenarios using fine-tuning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize FNO baseline with configuration.
        
        Args:
            config: Configuration dictionary containing:
                - modes1: Number of Fourier modes in first dimension
                - modes2: Number of Fourier modes in second dimension (for 2D)
                - width: Width of the FNO layers
                - layers: Number of FNO layers
                - learning_rate: Learning rate for training
                - batch_size: Batch size for training
                - epochs: Number of training epochs
                - device: Device to run on ('cuda' or 'cpu')
                - dimension: Problem dimension (1 or 2)
        """
        if not NEURALOPERATOR_AVAILABLE:
            raise ImportError("neuraloperator library is required for FNO baseline")
            
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.dimension = config.get('dimension', 2)
        
        # Build FNO model
        self.model = self._build_fno_model()
        self.model.to(self.device)
        
        # Training configuration
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.batch_size = config.get('batch_size', 32)
        self.epochs = config.get('epochs', 1000)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.5)
        
        # Training history
        self.training_history = {}
        
    def _build_fno_model(self) -> nn.Module:
        """Build FNO model based on problem dimension."""
        if self.dimension == 1:
            return FNO1d(
                modes=self.config.get('modes1', 16),
                width=self.config.get('width', 64),
                layers=self.config.get('layers', 4),
                in_channels=self.config.get('in_channels', 2),  # x, parameter
                out_channels=self.config.get('out_channels', 1)  # solution
            )
        elif self.dimension == 2:
            return FNO2d(
                modes1=self.config.get('modes1', 12),
                modes2=self.config.get('modes2', 12),
                width=self.config.get('width', 32),
                layers=self.config.get('layers', 4),
                in_channels=self.config.get('in_channels', 3),  # x, y, parameter
                out_channels=self.config.get('out_channels', 1)  # solution
            )
        else:
            raise ValueError(f"Unsupported dimension: {self.dimension}")
    
    def train_on_pde_family(self, pde_family: str, train_tasks: List[Task]) -> Dict[str, Any]:
        """Train FNO on a specific PDE family.
        
        Args:
            pde_family: Name of the PDE family (e.g., 'heat', 'burgers')
            train_tasks: List of training tasks for this PDE family
            
        Returns:
            Dictionary containing training results and metrics
        """
        print(f"Training FNO on {pde_family} family with {len(train_tasks)} tasks")
        
        # Prepare training data
        train_loader = self._prepare_training_data(train_tasks)
        
        # Training loop
        start_time = time.time()
        losses = []
        
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(batch_inputs)
                
                # Compute loss (L2 loss)
                loss = F.mse_loss(predictions, batch_targets)
                
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
        """Prepare training data from tasks for FNO training.
        
        Args:
            tasks: List of training tasks
            
        Returns:
            DataLoader for training
        """
        all_inputs = []
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
            
            # Prepare input: coordinates + parameter
            if self.dimension == 1:
                # For 1D: [x, param] -> [N, 2]
                param_expanded = np.full((coords.shape[0], 1), params[0])
                input_data = np.concatenate([coords, param_expanded], axis=1)
            else:
                # For 2D: [x, y, param] -> [N, 3]
                param_expanded = np.full((coords.shape[0], 1), params[0])
                input_data = np.concatenate([coords, param_expanded], axis=1)
            
            all_inputs.append(torch.tensor(input_data, dtype=torch.float32))
            all_targets.append(torch.tensor(solution, dtype=torch.float32))
        
        # Stack all data
        inputs = torch.stack(all_inputs)
        targets = torch.stack(all_targets)
        
        # Create dataset and dataloader
        dataset = TensorDataset(inputs, targets)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
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
            for batch_inputs, batch_targets in support_loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                adapted_optimizer.zero_grad()
                predictions = adapted_model(batch_inputs)
                loss = F.mse_loss(predictions, batch_targets)
                loss.backward()
                adapted_optimizer.step()
        
        adaptation_time = time.time() - start_time
        
        # Evaluate on query tasks
        adapted_model.eval()
        query_results = []
        
        with torch.no_grad():
            for query_task in query_tasks:
                task_data = query_task.get_task_data()
                
                # Prepare input
                coords = task_data.x_physics
                params = task_data.params
                
                if self.dimension == 1:
                    param_expanded = np.full((coords.shape[0], 1), params[0])
                    input_data = np.concatenate([coords, param_expanded], axis=1)
                else:
                    param_expanded = np.full((coords.shape[0], 1), params[0])
                    input_data = np.concatenate([coords, param_expanded], axis=1)
                
                input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Get prediction
                prediction = adapted_model(input_tensor).squeeze(0).cpu().numpy()
                
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
            # Prepare input
            if self.dimension == 1:
                param_expanded = np.full((coordinates.shape[0], 1), parameters[0])
                input_data = np.concatenate([coordinates, param_expanded], axis=1)
            else:
                param_expanded = np.full((coordinates.shape[0], 1), parameters[0])
                input_data = np.concatenate([coordinates, param_expanded], axis=1)
            
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            prediction = self.model(input_tensor).squeeze(0).cpu().numpy()
            
        return prediction
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_history = checkpoint.get('training_history', {})
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        num_params = sum(p.numel() for p in self.model.parameters())
        return {
            'model_type': 'FNO',
            'dimension': self.dimension,
            'modes1': self.config.get('modes1'),
            'modes2': self.config.get('modes2') if self.dimension == 2 else None,
            'width': self.config.get('width'),
            'layers': self.config.get('layers'),
            'num_parameters': num_params,
            'device': str(self.device)
        }


def create_fno_baseline(pde_family: str, dimension: int = 2) -> FNOBaseline:
    """Create FNO baseline with default configuration for a PDE family.
    
    Args:
        pde_family: Name of the PDE family
        dimension: Problem dimension (1 or 2)
        
    Returns:
        Configured FNOBaseline instance
    """
    # Default configurations for different PDE families
    default_configs = {
        'heat': {
            'modes1': 12, 'modes2': 12, 'width': 32, 'layers': 4,
            'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 1000
        },
        'burgers': {
            'modes1': 16, 'modes2': 16, 'width': 64, 'layers': 4,
            'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 1000
        },
        'poisson': {
            'modes1': 12, 'modes2': 12, 'width': 32, 'layers': 4,
            'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 800
        },
        'navier_stokes': {
            'modes1': 16, 'modes2': 16, 'width': 64, 'layers': 6,
            'learning_rate': 5e-4, 'batch_size': 16, 'epochs': 1500
        },
        'gray_scott': {
            'modes1': 16, 'modes2': 16, 'width': 64, 'layers': 6,
            'learning_rate': 5e-4, 'batch_size': 16, 'epochs': 1500
        },
        'kuramoto_sivashinsky': {
            'modes1': 20, 'modes2': 20, 'width': 64, 'layers': 6,
            'learning_rate': 5e-4, 'batch_size': 16, 'epochs': 1500
        },
        'darcy': {
            'modes1': 12, 'modes2': 12, 'width': 32, 'layers': 4,
            'learning_rate': 1e-3, 'batch_size': 32, 'epochs': 1000
        }
    }
    
    # Get configuration for the PDE family
    config = default_configs.get(pde_family, default_configs['heat']).copy()
    config['dimension'] = dimension
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Adjust input/output channels based on dimension
    if dimension == 1:
        config['in_channels'] = 2   # x, parameter
        config['out_channels'] = 1  # solution
    else:
        config['in_channels'] = 3   # x, y, parameter
        config['out_channels'] = 1  # solution
    
    return FNOBaseline(config)