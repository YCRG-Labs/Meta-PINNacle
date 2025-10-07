"""PhysicsInformedMetaLearner: Enhanced meta-learning with adaptive constraint weighting.

This module implements an enhanced version of MetaPINN that includes adaptive
constraint weighting, physics regularization, and improved physics constraint
handling for meta-learning in physics-informed neural networks.
"""

import copy
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .meta_pinn import MetaPINN
from .config import PhysicsInformedMetaLearnerConfig
from .constraint_balancer import (
    create_constraint_balancer, 
    PhysicsRegularizer,
    ConstraintBalancer
)
from .multi_scale_handler import MultiScaleHandler, extend_pde_with_multi_scale
from .uncertainty_estimator import (
    create_uncertainty_estimator,
    UncertaintyGuidedSampler,
    UncertaintyAwareConstraintWeighting,
    UncertaintyAnalyzer
)
from .task import Task, TaskData, TaskBatch


class PhysicsInformedMetaLearner(MetaPINN):
    """Enhanced meta-learning model with adaptive constraint weighting.
    
    This class extends MetaPINN with:
    1. Adaptive constraint weighting using PINNacle's loss_config system
    2. Physics-regularized meta-objectives that preserve PDE structure
    3. Enhanced physics constraint handling for complex multi-physics problems
    """
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize PhysicsInformedMetaLearner.
        
        Args:
            config: Configuration for enhanced meta-learning
        """
        # Initialize base MetaPINN
        super().__init__(config)
        
        # Store enhanced config
        self.enhanced_config = config
        
        # Initialize constraint balancer
        self.constraint_balancer = create_constraint_balancer(config)
        
        # Initialize physics regularizer
        self.physics_regularizer = PhysicsRegularizer(config)
        
        # Initialize multi-scale handler
        self.multi_scale_handler = MultiScaleHandler(config)
        
        # Initialize uncertainty estimation components
        self.uncertainty_estimator = create_uncertainty_estimator(config)
        self.uncertainty_guided_sampler = UncertaintyGuidedSampler(self.uncertainty_estimator)
        self.uncertainty_constraint_weighting = UncertaintyAwareConstraintWeighting(self.uncertainty_estimator)
        self.uncertainty_analyzer = UncertaintyAnalyzer()
        
        # Enhanced loss tracking
        self.constraint_weights_history = []
        self.physics_regularization_history = []
        self.multi_scale_history = []
        self.uncertainty_history = []
        
        # Multi-constraint handling
        self.constraint_types = [
            'data_loss', 'physics_loss', 'boundary_loss', 'initial_loss'
        ]
        
        # Adaptive weighting parameters
        self.adaptive_weighting = config.adaptive_constraint_weighting
        self.constraint_update_frequency = config.constraint_update_frequency
        
        # Physics consistency tracking
        self.physics_consistency_history = []
        
        # Initialize regularizer with current parameters
        if hasattr(self, 'network'):
            self.physics_regularizer.set_initial_params(
                dict(self.network.named_parameters())
            )
    
    def set_pde_problem(self, pde_problem):
        """Set the current PDE problem with multi-scale extension."""
        # Set base PDE problem
        self.current_pde = pde_problem
        
        # Extend PDE with multi-scale capabilities if enabled
        if self.multi_scale_handler.enabled:
            self.current_pde = extend_pde_with_multi_scale(pde_problem, self.enhanced_config)
    
    def compute_enhanced_physics_loss(self, task_data: TaskData, task: Task,
                                    params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute enhanced physics loss with multiple constraint types and multi-scale handling.
        
        Args:
            task_data: Task data containing inputs and targets
            task: Task object with problem parameters
            params: Optional adapted parameters
            
        Returns:
            Dictionary of different physics loss components
        """
        physics_losses = {}
        
        # Multi-scale physics loss computation
        if self.multi_scale_handler.enabled and self.current_pde is not None:
            # Detect problem scales
            scales = self.multi_scale_handler.detect_problem_scales(task_data, self.current_pde)
            
            # Compute multi-scale physics losses
            multi_scale_losses = self.multi_scale_handler.compute_multi_scale_physics_loss(
                self, task_data, task, scales, params
            )
            physics_losses.update(multi_scale_losses)
            
            # Store scale information for analysis
            if len(self.multi_scale_history) < 1000:  # Limit history size
                self.multi_scale_history.append({
                    'iteration': self.meta_iteration,
                    'scales': scales,
                    'scale_losses': {k: v.item() for k, v in multi_scale_losses.items()}
                })
        else:
            # Standard physics loss (PDE residual)
            physics_losses['physics_loss'] = self.compute_physics_loss(task_data, task, params)
        
        # Boundary condition loss
        physics_losses['boundary_loss'] = self.compute_boundary_loss(task_data, task, params)
        
        # Initial condition loss
        physics_losses['initial_loss'] = self.compute_initial_loss(task_data, task, params)
        
        # Additional physics constraints based on PDE type
        if hasattr(self.current_pde, 'compute_continuity_loss'):
            physics_losses['continuity_loss'] = self.current_pde.compute_continuity_loss(
                task_data.collocation_points, self.forward(task_data.collocation_points, params)
            )
        
        if hasattr(self.current_pde, 'compute_momentum_loss'):
            physics_losses['momentum_loss'] = self.current_pde.compute_momentum_loss(
                task_data.collocation_points, self.forward(task_data.collocation_points, params)
            )
        
        return physics_losses
    
    def compute_adaptive_total_loss(self, task_data: TaskData, task: Task,
                                  params: Optional[Dict[str, torch.Tensor]] = None,
                                  iteration: int = 0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss with adaptive constraint weighting.
        
        Args:
            task_data: Task data
            task: Task object
            params: Optional adapted parameters
            iteration: Current training iteration
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Compute individual loss components
        loss_components = {}
        
        # Data loss
        loss_components['data_loss'] = self.compute_data_loss(task_data, params)
        
        # Enhanced physics losses
        physics_losses = self.compute_enhanced_physics_loss(task_data, task, params)
        loss_components.update(physics_losses)
        
        # Compute adaptive weights if enabled
        if self.adaptive_weighting:
            constraint_weights = self.constraint_balancer.compute_weights(
                loss_components, iteration
            )
            
            # Apply uncertainty-aware weighting if enabled
            if self.enhanced_config.uncertainty_estimation:
                uncertainty_weights = self.uncertainty_constraint_weighting.compute_uncertainty_weights(
                    self, task_data, params
                )
                
                # Combine constraint balancer weights with uncertainty weights
                for name in constraint_weights:
                    if name in uncertainty_weights:
                        constraint_weights[name] = constraint_weights[name] * uncertainty_weights[name]
        else:
            # Use static weights from config
            device = list(loss_components.values())[0].device
            constraint_weights = {
                'data_loss': torch.tensor(self.data_loss_weight, device=device),
                'physics_loss': torch.tensor(self.physics_loss_weight, device=device),
                'boundary_loss': torch.tensor(self.boundary_loss_weight, device=device),
                'initial_loss': torch.tensor(self.initial_loss_weight, device=device)
            }
            
            # Apply uncertainty weighting even with static base weights
            if self.enhanced_config.uncertainty_estimation:
                uncertainty_weights = self.uncertainty_constraint_weighting.compute_uncertainty_weights(
                    self, task_data, params
                )
                for name in constraint_weights:
                    if name in uncertainty_weights:
                        constraint_weights[name] = constraint_weights[name] * uncertainty_weights[name]
            
            # Add default weights for additional constraints
            for name in loss_components:
                if name not in constraint_weights:
                    constraint_weights[name] = torch.tensor(1.0, device=device)
        
        # Compute weighted total loss
        total_loss = torch.tensor(0.0, device=list(loss_components.values())[0].device)
        
        for name, loss in loss_components.items():
            weight = constraint_weights.get(name, torch.tensor(1.0, device=loss.device))
            total_loss += weight * loss
        
        # Add physics regularization for meta-parameters
        if params is not None:
            physics_reg_loss = self.physics_regularizer.compute_regularization_loss(params)
            total_loss += physics_reg_loss
            loss_components['physics_regularization'] = physics_reg_loss
        
        # Store constraint weights for analysis
        if iteration % 10 == 0:  # Store every 10 iterations to avoid memory issues
            weight_dict = {name: weight.item() for name, weight in constraint_weights.items()}
            self.constraint_weights_history.append({
                'iteration': iteration,
                'weights': weight_dict,
                'losses': {name: loss.item() for name, loss in loss_components.items()}
            })
        
        return total_loss, loss_components
    
    def compute_total_loss(self, task_data: TaskData, task: Task,
                          params: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """Override base method to use adaptive weighting."""
        total_loss, _ = self.compute_adaptive_total_loss(
            task_data, task, params, self.meta_iteration
        )
        return total_loss
    
    def adapt_to_task(self, task: Task, adaptation_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Enhanced adaptation with physics-aware constraint balancing."""
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        # Clone current parameters for adaptation
        adapted_params = {name: param.clone() for name, param in self.network.named_parameters()}
        
        # Track adaptation progress for constraint balancing
        adaptation_losses = []
        
        # Inner loop adaptation with adaptive constraints and uncertainty estimation
        for step in range(adaptation_steps):
            # Update uncertainty estimator
            if self.enhanced_config.uncertainty_estimation:
                self.uncertainty_estimator.update(self, task.support_data, step)
            
            # Compute adaptive loss
            support_loss, loss_components = self.compute_adaptive_total_loss(
                task.support_data, task, adapted_params, 
                iteration=self.meta_iteration * adaptation_steps + step
            )
            
            adaptation_losses.append(support_loss)
            
            # Log uncertainty information
            if self.enhanced_config.uncertainty_estimation and step % 2 == 0:
                predictions, uncertainties = self.uncertainty_estimator.estimate_uncertainty(
                    self, task.support_data.inputs, adapted_params
                )
                self.uncertainty_analyzer.log_uncertainty(
                    predictions, uncertainties, task.support_data.outputs,
                    self.meta_iteration * adaptation_steps + step
                )
            
            # Compute gradients w.r.t. adapted parameters
            grads = torch.autograd.grad(
                support_loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=self.config.allow_unused
            )
            
            # Update adapted parameters
            for (name, param), grad in zip(adapted_params.items(), grads):
                if grad is not None:
                    adapted_params[name] = param - self.adapt_lr * grad
            
            # Update constraint balancer
            if step % self.constraint_update_frequency == 0:
                self.constraint_balancer.update(loss_components, step)
        
        # Compute physics consistency loss for this adaptation
        if len(adaptation_losses) > 1:
            consistency_loss = self.physics_regularizer.compute_consistency_loss(
                adapted_params, adaptation_losses
            )
            self.physics_consistency_history.append(consistency_loss.item())
        
        return adapted_params
    
    def fast_adapt(self, support_data: TaskData, task: Task, k_shots: int,
                   adaptation_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Fast adaptation using functional gradients with uncertainty-guided sampling.
        
        This is more memory efficient than creating a full model copy.
        """
        if k_shots not in [1, 5, 10, 25]:
            raise ValueError(f"k_shots must be in [1, 5, 10, 25], got {k_shots}")
        
        if adaptation_steps is None:
            adaptation_steps = self.adaptation_steps
        
        if adaptation_steps not in [1, 5, 10, 25]:
            raise ValueError(f"adaptation_steps must be in [1, 5, 10, 25], got {adaptation_steps}")
        
        # Use uncertainty-guided sampling if enabled
        if self.enhanced_config.uncertainty_estimation and len(support_data) > k_shots:
            sampled_inputs, sampled_outputs = self.uncertainty_guided_sampler.sample_support_data(
                self, support_data, k_shots
            )
            
            # Create new support data with sampled points
            sampled_support_data = TaskData(
                inputs=sampled_inputs,
                outputs=sampled_outputs,
                collocation_points=support_data.collocation_points,
                boundary_data=support_data.boundary_data,
                initial_data=support_data.initial_data
            )
        else:
            # Use random sampling or all data if not enough points
            if len(support_data) > k_shots:
                sampled_support_data = support_data.sample(k_shots)
            else:
                sampled_support_data = support_data
        
        # Create task with sampled support data
        adapted_task = Task(
            problem_type=task.problem_type,
            parameters=task.parameters,
            support_data=sampled_support_data,
            query_data=task.query_data,
            metadata=task.metadata
        )
        
        # Use existing adapt_to_task method
        return self.adapt_to_task(adapted_task, adaptation_steps)
    
    def meta_train_step(self, task_batch: TaskBatch) -> Dict[str, float]:
        """Enhanced meta-training step with adaptive constraints."""
        meta_loss = 0.0
        task_losses = []
        all_loss_components = {}
        
        # Initialize component tracking
        for component in self.constraint_types:
            all_loss_components[component] = []
        
        for task in task_batch.tasks:
            # Inner loop: adapt to task with enhanced constraints
            adapted_params = self.adapt_to_task(task)
            
            # Outer loop: evaluate on query set with adaptive weighting
            query_loss, loss_components = self.compute_adaptive_total_loss(
                task.query_data, task, adapted_params, self.meta_iteration
            )
            
            meta_loss += query_loss
            task_losses.append(query_loss.item())
            
            # Track loss components
            for component, loss in loss_components.items():
                if component not in all_loss_components:
                    all_loss_components[component] = []
                all_loss_components[component].append(loss.item())
        
        # Average meta-loss across tasks
        meta_loss = meta_loss / len(task_batch.tasks)
        
        # Meta-gradient update
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        
        # Apply gradient clipping if specified
        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
        
        self.meta_optimizer.step()
        
        # Update constraint balancer at meta-level
        if self.meta_iteration % self.constraint_update_frequency == 0:
            # Aggregate loss components across tasks
            aggregated_losses = {}
            for component, losses in all_loss_components.items():
                if losses:  # Only include components that have values
                    aggregated_losses[component] = torch.tensor(np.mean(losses))
            
            if aggregated_losses:
                self.constraint_balancer.update(aggregated_losses, self.meta_iteration)
        
        # Update training state
        self.meta_iteration += 1
        
        # Prepare return metrics
        metrics = {
            'meta_loss': meta_loss.item(),
            'mean_task_loss': np.mean(task_losses),
            'std_task_loss': np.std(task_losses),
            'meta_iteration': self.meta_iteration
        }
        
        # Add component-wise metrics
        for component, losses in all_loss_components.items():
            if losses:
                metrics[f'mean_{component}'] = np.mean(losses)
                metrics[f'std_{component}'] = np.std(losses)
        
        return metrics
    
    def evaluate_adaptation(self, adapted_params: Dict[str, torch.Tensor],
                          query_data: TaskData, task: Task) -> Dict[str, float]:
        """Enhanced evaluation with detailed constraint analysis."""
        self.network.eval()
        with torch.no_grad():
            # Compute predictions using adapted parameters
            predictions = self.forward(query_data.inputs, adapted_params)
            
            # Compute detailed loss breakdown
            total_loss, loss_components = self.compute_adaptive_total_loss(
                query_data, task, adapted_params, self.meta_iteration
            )
            
            # Compute relative L2 error
            l2_error = torch.norm(predictions - query_data.outputs) / torch.norm(query_data.outputs)
            
            # Compute physics compliance metrics
            physics_losses = self.compute_enhanced_physics_loss(query_data, task, adapted_params)
            physics_compliance = {}
            for name, loss in physics_losses.items():
                physics_compliance[f'{name}_compliance'] = 1.0 / (1.0 + loss.item())
        
        # Prepare detailed metrics
        metrics = {
            'total_loss': total_loss.item(),
            'l2_relative_error': l2_error.item(),
            'mse': loss_components['data_loss'].item()
        }
        
        # Add individual loss components
        for name, loss in loss_components.items():
            metrics[name] = loss.item()
        
        # Add physics compliance metrics
        metrics.update(physics_compliance)
        
        return metrics
    
    def get_constraint_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of constraint weighting behavior."""
        if not self.constraint_weights_history:
            return {'message': 'No constraint weight history available'}
        
        analysis = {
            'weight_evolution': self.constraint_weights_history,
            'physics_consistency': self.physics_consistency_history,
            'final_weights': self.constraint_weights_history[-1]['weights'] if self.constraint_weights_history else {},
            'balancer_type': self.enhanced_config.constraint_balancer_type
        }
        
        # Compute weight statistics
        if len(self.constraint_weights_history) > 10:
            weight_stats = {}
            for constraint in self.constraint_types:
                weights = [entry['weights'].get(constraint, 1.0) 
                          for entry in self.constraint_weights_history]
                weight_stats[constraint] = {
                    'mean': np.mean(weights),
                    'std': np.std(weights),
                    'min': np.min(weights),
                    'max': np.max(weights)
                }
            analysis['weight_statistics'] = weight_stats
        
        return analysis
    
    def get_uncertainty_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of uncertainty estimation behavior."""
        if not self.enhanced_config.uncertainty_estimation:
            return {'message': 'Uncertainty estimation not enabled'}
        
        # Get uncertainty analyzer summary
        uncertainty_summary = self.uncertainty_analyzer.get_uncertainty_summary()
        
        # Add multi-scale handler summary if enabled
        multi_scale_summary = {}
        if self.multi_scale_handler.enabled:
            multi_scale_summary = self.multi_scale_handler.get_multi_scale_summary()
        
        return {
            'uncertainty_enabled': True,
            'uncertainty_method': self.enhanced_config.uncertainty_method,
            'uncertainty_summary': uncertainty_summary,
            'multi_scale_summary': multi_scale_summary,
            'uncertainty_history_length': len(self.uncertainty_history)
        }
    
    def save_model(self, filepath: str):
        """Save enhanced model with constraint balancer state."""
        checkpoint = {
            'network_state_dict': self.network.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'config': self.enhanced_config,
            'meta_iteration': self.meta_iteration,
            'training_history': self.training_history,
            'constraint_weights_history': self.constraint_weights_history,
            'physics_consistency_history': self.physics_consistency_history,
            'multi_scale_history': self.multi_scale_history,
            'uncertainty_history': self.uncertainty_history
        }
        
        # Save constraint balancer state if it has one
        if hasattr(self.constraint_balancer, 'state_dict'):
            checkpoint['constraint_balancer_state'] = self.constraint_balancer.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        """Load enhanced model with constraint balancer state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load base model components
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.meta_iteration = checkpoint.get('meta_iteration', 0)
        self.training_history = checkpoint.get('training_history', [])
        
        # Load enhanced components
        self.constraint_weights_history = checkpoint.get('constraint_weights_history', [])
        self.physics_consistency_history = checkpoint.get('physics_consistency_history', [])
        self.multi_scale_history = checkpoint.get('multi_scale_history', [])
        self.uncertainty_history = checkpoint.get('uncertainty_history', [])
        
        # Load constraint balancer state if available
        if 'constraint_balancer_state' in checkpoint and hasattr(self.constraint_balancer, 'load_state_dict'):
            self.constraint_balancer.load_state_dict(checkpoint['constraint_balancer_state'])
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get enhanced model summary."""
        base_summary = super().get_model_summary()
        
        enhanced_summary = {
            **base_summary,
            'model_type': 'PhysicsInformedMetaLearner',
            'adaptive_constraint_weighting': self.adaptive_weighting,
            'constraint_balancer_type': self.enhanced_config.constraint_balancer_type,
            'physics_regularization_weight': self.enhanced_config.physics_regularization_weight,
            'constraint_update_frequency': self.constraint_update_frequency,
            'n_constraint_updates': len(self.constraint_weights_history),
            'physics_consistency_samples': len(self.physics_consistency_history),
            'multi_scale_enabled': self.multi_scale_handler.enabled,
            'multi_scale_samples': len(self.multi_scale_history),
            'uncertainty_estimation': self.enhanced_config.uncertainty_estimation,
            'uncertainty_method': self.enhanced_config.uncertainty_method if self.enhanced_config.uncertainty_estimation else None,
            'uncertainty_samples': len(self.uncertainty_history)
        }
        
        # Add current constraint weights if available
        if self.constraint_weights_history:
            enhanced_summary['current_constraint_weights'] = self.constraint_weights_history[-1]['weights']
        
        return enhanced_summary