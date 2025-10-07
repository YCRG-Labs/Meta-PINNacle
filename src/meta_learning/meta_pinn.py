"""MetaPINN implementation using MAML algorithm that integrates with PINNacle's existing framework."""

import copy
import time
import random
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
import numpy as np
import deepxde as dde
from deepxde.model import Model

from .config import MetaPINNConfig
from .task import Task, TaskData, TaskBatch
from ..model import FNN
from .error_handling import (
    safe_physics_loss_computation,
    safe_gradient_computation,
    safe_adaptation_step,
    validate_task_data,
    with_error_recovery,
    MetaLearningError,
    PhysicsLossError,
    GradientComputationError,
    AdaptationError,
    TaskDataError
)
from .enhanced_logging import (
    get_meta_learning_monitor,
    MetaLearningPhase
)


class MetaPINN:
    """MAML-based meta-learning for Physics-Informed Neural Networks.
    
    This class implements the Model-Agnostic Meta-Learning (MAML) algorithm
    specifically designed for physics-informed neural networks, integrating
    with PINNacle's existing PDE classes and training infrastructure.
    """
    
    def __init__(self, config: MetaPINNConfig):
        """Initialize MetaPINN with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize network architecture using PINNacle's FNN
        self.network = FNN(
            layer_sizes=config.layers,
            activation=config.activation,
            kernel_initializer=config.initializer
        ).to(self.device)
        
        # Meta-learning parameters
        self.meta_lr = config.meta_lr
        self.adapt_lr = config.adapt_lr
        self.adaptation_steps = config.adaptation_steps
        self.meta_batch_size = config.meta_batch_size
        self.first_order = config.first_order
        
        # Physics loss weights (following PINNacle's loss_config pattern)
        self.physics_loss_weight = config.physics_loss_weight
        self.data_loss_weight = config.data_loss_weight
        self.boundary_loss_weight = config.boundary_loss_weight
        self.initial_loss_weight = config.initial_loss_weight
        
        # Initialize meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.meta_lr
        )
        
        # Training state
        self.meta_iteration = 0
        self.training_history = []
        
        # Current PDE problem (set during training)
        self.current_pde = None
        
    def set_pde_problem(self, pde_problem):
        """Set the current PDE problem for physics loss computation."""
        self.current_pde = pde_problem
    
    def forward(self, x: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Forward pass through the network."""
        if params is None:
            return self.network(x)
        else:
            return self._functional_forward(x, params)
    
    def _functional_forward(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Functional forward pass using provided parameters."""
        activation_fn = getattr(torch.nn.functional, self.config.activation)
        current_x = x
        
        # Apply input transform if exists
        if hasattr(self.network, '_input_transform') and self.network._input_transform is not None:
            current_x = self.network._input_transform(current_x)
        
        # Forward through linear layers
        for i, linear_layer in enumerate(self.network.linears):
            weight_key = f'linears.{i}.weight'
            bias_key = f'linears.{i}.bias'
            
            if weight_key in params and bias_key in params:
                current_x = torch.nn.functional.linear(current_x, params[weight_key], params[bias_key])
            else:
                current_x = linear_layer(current_x)
            
            # Apply activation to all layers except the last
            if i < len(self.network.linears) - 1:
                if self.config.activation == 'sin':
                    current_x = torch.sin(current_x)
                elif self.config.activation == 'tanh':
                    current_x = torch.tanh(current_x)
                else:
                    current_x = activation_fn(current_x)
        
        # Apply output transform if exists
        if hasattr(self.network, '_output_transform') and self.network._output_transform is not None:
            current_x = self.network._output_transform(x, current_x)
        
        return current_x
    
    def compute_data_loss(self, task_data: TaskData, params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute data fitting loss (MSE between predictions and targets)."""
        predictions = self.forward(task_data.inputs, params)
        return torch.nn.functional.mse_loss(predictions, task_data.outputs)
    
    @safe_physics_loss_computation
    def compute_physics_loss(self, task_data: TaskData, task: Task, 
                           params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute physics-informed loss using PDE residual."""
        if self.current_pde is None:
            raise PhysicsLossError("PDE problem not set. Call set_pde_problem() first.")
        
        # Validate task data
        if not validate_task_data(task_data):
            raise TaskDataError("Invalid task data for physics loss computation")
        
        # Enable gradient computation for collocation points
        collocation_points = task_data.collocation_points.clone().detach().requires_grad_(True)
        
        # Forward pass to get network predictions
        u_pred = self.forward(collocation_points, params)
        
        # Check for NaN/Inf in predictions
        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            raise PhysicsLossError("NaN/Inf detected in network predictions")
        
        # Compute PDE residual using the current PDE's function
        pde_residual = self.current_pde.pde(collocation_points, u_pred)
        
        # Handle different return types from PDE function
        if isinstance(pde_residual, (list, tuple)):
            physics_loss = sum(torch.mean(residual**2) for residual in pde_residual)
        else:
            physics_loss = torch.mean(pde_residual**2)
        
        # Check for NaN/Inf in physics loss
        if torch.isnan(physics_loss) or torch.isinf(physics_loss):
            raise PhysicsLossError("NaN/Inf detected in physics loss computation")
        
        return physics_loss
    
    def compute_boundary_loss(self, task_data: TaskData, task: Task,
                            params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute boundary condition loss."""
        if task_data.boundary_data is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        boundary_inputs = task_data.boundary_data[:, :-1]
        boundary_targets = task_data.boundary_data[:, -1:]
        
        boundary_pred = self.forward(boundary_inputs, params)
        return torch.nn.functional.mse_loss(boundary_pred, boundary_targets)
    
    def compute_initial_loss(self, task_data: TaskData, task: Task,
                           params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute initial condition loss."""
        if task_data.initial_data is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        initial_inputs = task_data.initial_data[:, :-1]
        initial_targets = task_data.initial_data[:, -1:]
        
        initial_pred = self.forward(initial_inputs, params)
        return torch.nn.functional.mse_loss(initial_pred, initial_targets)
    
    def compute_total_loss(self, task_data: TaskData, task: Task,
                          params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Compute total loss combining data, physics, boundary, and initial losses."""
        data_loss = self.compute_data_loss(task_data, params)
        physics_loss = self.compute_physics_loss(task_data, task, params)
        boundary_loss = self.compute_boundary_loss(task_data, task, params)
        initial_loss = self.compute_initial_loss(task_data, task, params)
        
        total_loss = (
            self.data_loss_weight * data_loss +
            self.physics_loss_weight * physics_loss +
            self.boundary_loss_weight * boundary_loss +
            self.initial_loss_weight * initial_loss
        )
        
        return total_loss
    
    @safe_adaptation_step
    def adapt_to_task(self, task: Task, adaptation_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Adapt model parameters to a specific task using gradient descent."""
        monitor = get_meta_learning_monitor()
        task_id = f"{task.problem_type}_{hash(str(task.parameters))}"
        
        if monitor:
            monitor.log_phase_start(MetaLearningPhase.ADAPTATION, 
                                  {'task_id': task_id, 'adaptation_steps': adaptation_steps})
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        # Validate task data
        if not validate_task_data(task.support_data):
            error = TaskDataError("Invalid support data for adaptation")
            if monitor:
                monitor.log_error(error, f"adapt_to_task_{task_id}")
            raise error
        
        # Clone current parameters for adaptation
        adapted_params = {name: param.clone() for name, param in self.network.named_parameters()}
        
        # Inner loop adaptation
        for step in range(adaptation_steps):
            try:
                # Compute loss on support set
                support_loss = self.compute_total_loss(task.support_data, task, adapted_params)
                
                # Check for NaN/Inf in loss
                if torch.isnan(support_loss) or torch.isinf(support_loss):
                    error = AdaptationError(f"NaN/Inf detected in support loss at step {step}")
                    if monitor:
                        monitor.log_error(error, f"adapt_to_task_{task_id}_step_{step}")
                    raise error
                
                # Compute gradients w.r.t. adapted parameters
                grads = self._safe_gradient_computation(support_loss, adapted_params)
                
                if grads is None:
                    error = AdaptationError(f"Gradient computation failed at step {step}")
                    if monitor:
                        monitor.log_error(error, f"adapt_to_task_{task_id}_step_{step}")
                    raise error
                
                # Compute gradient norm for monitoring
                grad_norm = None
                if grads:
                    grad_norm = torch.norm(torch.stack([torch.norm(g) for g in grads if g is not None]))
                
                # Log adaptation step
                if monitor:
                    monitor.log_adaptation_step(task_id, step, support_loss.item(), 
                                              grad_norm.item() if grad_norm is not None else None)
                
                # Update adapted parameters
                for (name, param), grad in zip(adapted_params.items(), grads):
                    if grad is not None:
                        # Check for NaN/Inf in gradients
                        if torch.isnan(grad).any() or torch.isinf(grad).any():
                            error = AdaptationError(f"NaN/Inf detected in gradients at step {step}")
                            if monitor:
                                monitor.log_error(error, f"adapt_to_task_{task_id}_step_{step}")
                            raise error
                        
                        adapted_params[name] = param - self.adapt_lr * grad
                        
                        # Check for NaN/Inf in updated parameters
                        if torch.isnan(adapted_params[name]).any() or torch.isinf(adapted_params[name]).any():
                            error = AdaptationError(f"NaN/Inf detected in updated parameters at step {step}")
                            if monitor:
                                monitor.log_error(error, f"adapt_to_task_{task_id}_step_{step}")
                            raise error
                            
            except (PhysicsLossError, GradientComputationError) as e:
                # Re-raise as adaptation error with context
                error = AdaptationError(f"Adaptation failed at step {step}: {e}")
                if monitor:
                    monitor.log_error(error, f"adapt_to_task_{task_id}_step_{step}")
                raise error
        
        if monitor:
            monitor.log_phase_end(MetaLearningPhase.ADAPTATION, 
                                {'task_id': task_id, 'final_loss': support_loss.item()})
        
        return adapted_params
    
    @safe_gradient_computation
    def _safe_gradient_computation(self, loss: torch.Tensor, 
                                 params: Dict[str, torch.Tensor]) -> Optional[List[torch.Tensor]]:
        """Safely compute gradients with error handling."""
        try:
            grads = torch.autograd.grad(
                loss, 
                params.values(),
                create_graph=not self.first_order,
                allow_unused=self.config.allow_unused,
                retain_graph=False
            )
            return grads
        except RuntimeError as e:
            if "one of the variables needed for gradient computation has been modified" in str(e):
                # Try with fresh parameter copies
                fresh_params = {name: param.clone().detach().requires_grad_(True) 
                              for name, param in params.items()}
                try:
                    # Recompute loss with fresh parameters
                    fresh_loss = loss  # This might need to be recomputed in practice
                    grads = torch.autograd.grad(
                        fresh_loss,
                        fresh_params.values(),
                        create_graph=not self.first_order,
                        allow_unused=self.config.allow_unused
                    )
                    return grads
                except Exception:
                    return None
            else:
                raise GradientComputationError(f"Gradient computation failed: {e}")
        except Exception as e:
            raise GradientComputationError(f"Unexpected error in gradient computation: {e}")
    
    @with_error_recovery
    def adapt(self, support_data: TaskData, task: Task, k_shots: int, 
              adaptation_steps: Optional[int] = None) -> 'MetaPINN':
        """Adapt to new task with K-shot learning.
        
        This method implements few-shot adaptation for K ∈ {1, 5, 10, 25} samples
        using existing PINNacle optimizers in the inner loop.
        """
        if k_shots not in [1, 5, 10, 25]:
            raise AdaptationError(f"k_shots must be in [1, 5, 10, 25], got {k_shots}")
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        if adaptation_steps not in [1, 5, 10, 25]:
            raise AdaptationError(f"adaptation_steps must be in [1, 5, 10, 25], got {adaptation_steps}")
        
        # Validate support data
        if not validate_task_data(support_data):
            raise TaskDataError("Invalid support data for adaptation")
        
        # Sample K support points if needed
        if len(support_data) > k_shots:
            support_data = support_data.sample(k_shots)
        elif len(support_data) < k_shots:
            print(f"Warning: Only {len(support_data)} samples available, using all for {k_shots}-shot adaptation")
        
        # Create adapted model copy
        try:
            adapted_model = copy.deepcopy(self)
        except Exception as e:
            raise AdaptationError(f"Failed to create adapted model copy: {e}")
        
        # Set up optimizer for inner loop using existing PINNacle patterns
        inner_optimizer = torch.optim.Adam(
            adapted_model.network.parameters(),
            lr=self.adapt_lr
        )
        
        # Inner loop optimization
        adapted_model.network.train()
        for step in range(adaptation_steps):
            try:
                inner_optimizer.zero_grad()
                
                # Compute loss on support set
                support_loss = adapted_model.compute_total_loss(support_data, task)
                
                # Check for NaN/Inf in loss
                if torch.isnan(support_loss) or torch.isinf(support_loss):
                    raise AdaptationError(f"NaN/Inf detected in support loss at step {step}")
                
                # Backward pass and optimization step
                support_loss.backward()
                
                # Check for NaN/Inf in gradients
                for name, param in adapted_model.network.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            raise AdaptationError(f"NaN/Inf detected in gradients for {name} at step {step}")
                
                inner_optimizer.step()
                
                # Check for NaN/Inf in parameters after update
                for name, param in adapted_model.network.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        raise AdaptationError(f"NaN/Inf detected in parameters for {name} after step {step}")
                        
            except (PhysicsLossError, TaskDataError) as e:
                raise AdaptationError(f"Adaptation failed at step {step}: {e}")
        
        return adapted_model
    
    def fast_adapt(self, support_data: TaskData, task: Task, k_shots: int,
                   adaptation_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Fast adaptation using functional gradients (MAML-style).
        
        This is more memory efficient than creating a full model copy.
        """
        if k_shots not in [1, 5, 10, 25]:
            raise ValueError(f"k_shots must be in [1, 5, 10, 25], got {k_shots}")
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        if adaptation_steps not in [1, 5, 10, 25]:
            raise ValueError(f"adaptation_steps must be in [1, 5, 10, 25], got {adaptation_steps}")
        
        # Sample K support points if needed
        if len(support_data) > k_shots:
            support_data = support_data.sample(k_shots)
        
        # Create task with sampled support data
        adapted_task = Task(
            problem_type=task.problem_type,
            parameters=task.parameters,
            support_data=support_data,
            query_data=task.query_data,
            metadata=task.metadata
        )
        
        # Use existing adapt_to_task method
        return self.adapt_to_task(adapted_task, adaptation_steps)
    
    def evaluate_adaptation(self, adapted_params: Dict[str, torch.Tensor], 
                          query_data: TaskData, task: Task) -> Dict[str, float]:
        """Evaluate adapted model on query data."""
        self.network.eval()
        with torch.no_grad():
            # Compute predictions using adapted parameters
            predictions = self.forward(query_data.inputs, adapted_params)
            
            # Compute various losses
            data_loss = torch.nn.functional.mse_loss(predictions, query_data.outputs)
            physics_loss = self.compute_physics_loss(query_data, task, adapted_params)
            total_loss = self.compute_total_loss(query_data, task, adapted_params)
            
            # Compute relative L2 error
            l2_error = torch.norm(predictions - query_data.outputs) / torch.norm(query_data.outputs)
            
        return {
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item(),
            'l2_relative_error': l2_error.item(),
            'mse': data_loss.item()
        }
    
    def few_shot_evaluation(self, test_tasks: List[Task], 
                           k_shots_list: List[int] = [1, 5, 10, 25],
                           adaptation_steps_list: List[int] = [1, 5, 10, 25]) -> Dict[str, Any]:
        """Comprehensive few-shot evaluation across different K and S values."""
        results = {}
        
        for k_shots in k_shots_list:
            for adaptation_steps in adaptation_steps_list:
                key = f"K{k_shots}_S{adaptation_steps}"
                task_results = []
                adaptation_times = []
                
                for task in test_tasks:
                    # Measure adaptation time
                    start_time = time.time()
                    
                    # Fast adaptation
                    adapted_params = self.fast_adapt(
                        task.support_data, task, k_shots, adaptation_steps
                    )
                    
                    adaptation_time = time.time() - start_time
                    adaptation_times.append(adaptation_time)
                    
                    # Evaluate on query set
                    metrics = self.evaluate_adaptation(adapted_params, task.query_data, task)
                    task_results.append(metrics)
                
                # Aggregate results
                results[key] = {
                    'mean_l2_error': np.mean([r['l2_relative_error'] for r in task_results]),
                    'std_l2_error': np.std([r['l2_relative_error'] for r in task_results]),
                    'mean_data_loss': np.mean([r['data_loss'] for r in task_results]),
                    'std_data_loss': np.std([r['data_loss'] for r in task_results]),
                    'mean_physics_loss': np.mean([r['physics_loss'] for r in task_results]),
                    'std_physics_loss': np.std([r['physics_loss'] for r in task_results]),
                    'mean_adaptation_time': np.mean(adaptation_times),
                    'std_adaptation_time': np.std(adaptation_times),
                    'task_results': task_results,
                    'adaptation_times': adaptation_times
                }
        
        return results
    
    def meta_train_step(self, task_batch: TaskBatch) -> Dict[str, float]:
        """Perform single meta-training step using MAML algorithm."""
        monitor = get_meta_learning_monitor()
        
        if monitor:
            monitor.log_phase_start(MetaLearningPhase.META_TRAINING, 
                                  {'batch_size': len(task_batch.tasks), 'iteration': self.meta_iteration})
        
        meta_loss = 0.0
        task_losses = []
        adaptation_times = []
        
        for task in task_batch.tasks:
            try:
                # Inner loop: adapt to task
                start_time = time.time()
                adapted_params = self.adapt_to_task(task)
                adaptation_time = time.time() - start_time
                adaptation_times.append(adaptation_time)
                
                # Outer loop: evaluate on query set
                query_loss = self.compute_total_loss(task.query_data, task, adapted_params)
                meta_loss += query_loss
                task_losses.append(query_loss.item())
                
                # Log memory usage periodically
                if monitor and len(task_losses) % 5 == 0:
                    monitor.log_memory_usage(f"meta_train_step_task_{len(task_losses)}")
                
            except Exception as e:
                if monitor:
                    monitor.log_error(e, f"meta_train_step_task_{len(task_losses)}")
                raise
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / len(task_batch.tasks)
        
        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Apply gradient clipping if specified
        if self.config.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
            if monitor:
                monitor.log_performance_metrics({'gradient_norm': grad_norm.item()}, 'gradient_clipping')
        
        self.meta_optimizer.step()
        
        # Update training state
        self.meta_iteration += 1
        
        # Log meta-training step
        if monitor:
            monitor.log_meta_training_step(self.meta_iteration, meta_loss.item(), 
                                         task_losses, adaptation_times)
            monitor.log_phase_end(MetaLearningPhase.META_TRAINING, 
                                {'meta_loss': meta_loss.item(), 'num_tasks': len(task_batch.tasks)})
        
        return {
            'meta_loss': meta_loss.item(),
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses),
            'meta_iteration': self.meta_iteration,
            'mean_adaptation_time': np.mean(adaptation_times),
            'std_adaptation_time': np.std(adaptation_times)
        }
    
    def meta_train(self, train_tasks: List[Task], val_tasks: Optional[List[Task]] = None,
                   meta_iterations: Optional[int] = None, **train_args) -> Dict[str, Any]:
        """Meta-training using existing trainer.py infrastructure."""
        if meta_iterations is None:
            meta_iterations = self.config.meta_iterations
        
        # Initialize training history
        self.training_history = []
        best_val_loss = float('inf')
        
        print(f"Starting meta-training for {meta_iterations} iterations...")
        print(f"Training on {len(train_tasks)} tasks, validating on {len(val_tasks) if val_tasks else 0} tasks")
        
        start_time = time.time()
        
        for iteration in range(meta_iterations):
            # Sample batch of tasks for meta-training
            task_indices = np.random.choice(len(train_tasks), self.meta_batch_size, replace=False)
            task_batch = TaskBatch([train_tasks[i] for i in task_indices])
            
            # Perform meta-training step
            train_metrics = self.meta_train_step(task_batch)
            
            # Validation step
            if val_tasks and iteration % self.config.validation_frequency == 0:
                val_metrics = self.meta_validate(val_tasks)
                
                # Update best model
                if val_metrics['meta_loss'] < best_val_loss:
                    best_val_loss = val_metrics['meta_loss']
                    # Save best model state (following PINNacle patterns)
                    self.best_state = copy.deepcopy(self.network.state_dict())
                
                # Log progress
                print(f"Iteration {iteration}: train_loss={train_metrics['meta_loss']:.6f}, "
                      f"val_loss={val_metrics['meta_loss']:.6f}")
                
                # Store training history
                self.training_history.append({
                    'iteration': iteration,
                    'train_loss': train_metrics['meta_loss'],
                    'val_loss': val_metrics['meta_loss'],
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
            
            elif iteration % 100 == 0:
                print(f"Iteration {iteration}: train_loss={train_metrics['meta_loss']:.6f}")
        
        total_time = time.time() - start_time
        print(f"Meta-training completed in {total_time:.2f} seconds")
        
        return {
            'training_history': self.training_history,
            'best_val_loss': best_val_loss,
            'total_time': total_time,
            'final_iteration': self.meta_iteration
        }
    
    def meta_validate(self, val_tasks: List[Task]) -> Dict[str, float]:
        """Validate meta-learning model on validation tasks."""
        self.network.eval()
        val_losses = []
        
        with torch.no_grad():
            for task in val_tasks:
                # Adapt to task
                adapted_params = self.adapt_to_task(task)
                
                # Evaluate on query set
                query_loss = self.compute_total_loss(task.query_data, task, adapted_params)
                val_losses.append(query_loss.item())
        
        self.network.train()
        
        return {
            'meta_loss': np.mean(val_losses),
            'std_loss': np.std(val_losses),
            'n_tasks': len(val_tasks)
        }
    
    def save_model(self, filepath: str):
        """Save model state following PINNacle patterns."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.config,
            'meta_iteration': self.meta_iteration,
            'training_history': self.training_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state following PINNacle patterns."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_iteration = checkpoint.get('meta_iteration', 0)
        self.training_history = checkpoint.get('training_history', [])
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary following PINNacle patterns."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            'model_type': 'MetaPINN',
            'network_architecture': self.config.layers,
            'activation': self.config.activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'meta_lr': self.meta_lr,
            'adapt_lr': self.adapt_lr,
            'adaptation_steps': self.adaptation_steps,
            'meta_batch_size': self.meta_batch_size,
            'first_order': self.first_order,
            'meta_iteration': self.meta_iteration,
            'device': str(self.device)
        }