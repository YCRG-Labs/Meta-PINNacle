"""Constraint balancer for adaptive physics constraint weighting in meta-learning.

This module implements adaptive constraint weighting mechanisms that work with
PINNacle's loss_config system to balance multiple physics constraints during
meta-learning training.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

from .config import PhysicsInformedMetaLearnerConfig


class ConstraintBalancer(ABC):
    """Abstract base class for constraint balancing strategies."""
    
    @abstractmethod
    def compute_weights(self, losses: Dict[str, torch.Tensor], 
                       iteration: int = 0) -> Dict[str, torch.Tensor]:
        """Compute adaptive weights for different constraint types.
        
        Args:
            losses: Dictionary mapping constraint names to loss values
            iteration: Current training iteration
            
        Returns:
            Dictionary mapping constraint names to weight tensors
        """
        pass
    
    @abstractmethod
    def update(self, losses: Dict[str, torch.Tensor], iteration: int):
        """Update internal state based on current losses."""
        pass


class StaticConstraintBalancer(ConstraintBalancer):
    """Static constraint balancer with fixed weights."""
    
    def __init__(self, weights: Dict[str, float]):
        """Initialize with fixed weights.
        
        Args:
            weights: Dictionary mapping constraint names to fixed weights
        """
        self.weights = weights
    
    def compute_weights(self, losses: Dict[str, torch.Tensor], 
                       iteration: int = 0) -> Dict[str, torch.Tensor]:
        """Return fixed weights as tensors."""
        weight_tensors = {}
        for name in losses.keys():
            weight = self.weights.get(name, 1.0)
            weight_tensors[name] = torch.tensor(weight, device=list(losses.values())[0].device)
        return weight_tensors
    
    def update(self, losses: Dict[str, torch.Tensor], iteration: int):
        """No update needed for static weights."""
        pass


class DynamicConstraintBalancer(ConstraintBalancer):
    """Dynamic constraint balancer that adapts weights based on loss magnitudes."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize dynamic balancer.
        
        Args:
            config: Configuration containing balancing parameters
        """
        self.config = config
        self.update_frequency = config.constraint_update_frequency
        
        # Moving averages of losses for stability
        self.loss_history = {}
        self.loss_ema = {}  # Exponential moving average
        self.ema_decay = 0.9
        
        # Adaptive parameters
        self.min_weight = 0.01
        self.max_weight = 10.0
        self.balance_factor = 1.0
        
        # Constraint types and their importance
        self.constraint_importance = {
            'data_loss': 1.0,
            'physics_loss': 1.0,
            'boundary_loss': 0.8,
            'initial_loss': 0.8,
            'continuity_loss': 0.6,
            'momentum_loss': 0.6
        }
    
    def compute_weights(self, losses: Dict[str, torch.Tensor], 
                       iteration: int = 0) -> Dict[str, torch.Tensor]:
        """Compute adaptive weights based on loss magnitudes and gradients."""
        weights = {}
        device = list(losses.values())[0].device
        
        # Update loss history
        self._update_loss_history(losses)
        
        if iteration < 10:  # Use equal weights initially
            for name in losses.keys():
                weights[name] = torch.tensor(1.0, device=device)
            return weights
        
        # Compute relative loss magnitudes
        loss_magnitudes = {}
        total_magnitude = 0.0
        
        for name, loss in losses.items():
            # Use EMA for stability
            magnitude = self.loss_ema.get(name, loss.item())
            loss_magnitudes[name] = magnitude
            total_magnitude += magnitude
        
        # Compute adaptive weights using inverse magnitude scaling
        for name, loss in losses.items():
            if total_magnitude > 0:
                # Inverse scaling: larger losses get smaller weights
                relative_magnitude = loss_magnitudes[name] / total_magnitude
                base_weight = 1.0 / (relative_magnitude + 1e-8)
                
                # Apply importance factor
                importance = self.constraint_importance.get(name, 1.0)
                adaptive_weight = base_weight * importance * self.balance_factor
                
                # Clamp weights to reasonable range
                adaptive_weight = np.clip(adaptive_weight, self.min_weight, self.max_weight)
                
                weights[name] = torch.tensor(adaptive_weight, device=device)
            else:
                weights[name] = torch.tensor(1.0, device=device)
        
        return weights
    
    def _update_loss_history(self, losses: Dict[str, torch.Tensor]):
        """Update exponential moving average of losses."""
        for name, loss in losses.items():
            loss_val = loss.item()
            
            if name not in self.loss_ema:
                self.loss_ema[name] = loss_val
            else:
                self.loss_ema[name] = (self.ema_decay * self.loss_ema[name] + 
                                      (1 - self.ema_decay) * loss_val)
            
            # Store in history for analysis
            if name not in self.loss_history:
                self.loss_history[name] = []
            self.loss_history[name].append(loss_val)
            
            # Keep only recent history
            if len(self.loss_history[name]) > 1000:
                self.loss_history[name] = self.loss_history[name][-1000:]
    
    def update(self, losses: Dict[str, torch.Tensor], iteration: int):
        """Update balancer state."""
        if iteration % self.update_frequency == 0:
            # Adjust balance factor based on loss convergence
            self._adjust_balance_factor(iteration)
    
    def _adjust_balance_factor(self, iteration: int):
        """Adjust balance factor based on training progress."""
        # Gradually reduce balance factor to stabilize training
        if iteration > 1000:
            self.balance_factor = max(0.5, self.balance_factor * 0.999)


class LearnedConstraintBalancer(ConstraintBalancer):
    """Learned constraint balancer using a small neural network."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize learned balancer.
        
        Args:
            config: Configuration containing balancing parameters
        """
        self.config = config
        self.device = torch.device(config.device)
        
        # Small network to predict weights
        self.weight_network = nn.Sequential(
            nn.Linear(10, 32),  # Input: loss statistics
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),   # Output: weights for different constraint types
            nn.Softplus()       # Ensure positive weights
        ).to(self.device)
        
        # Optimizer for weight network
        self.weight_optimizer = torch.optim.Adam(
            self.weight_network.parameters(), 
            lr=0.001
        )
        
        # Loss history for feature computation
        self.loss_history = {}
        self.max_history = 100
        
        # Constraint names (fixed order)
        self.constraint_names = [
            'data_loss', 'physics_loss', 'boundary_loss', 
            'initial_loss', 'continuity_loss', 'momentum_loss'
        ]
    
    def compute_weights(self, losses: Dict[str, torch.Tensor], 
                       iteration: int = 0) -> Dict[str, torch.Tensor]:
        """Compute weights using learned network."""
        if iteration < 50:  # Use equal weights initially
            weights = {}
            for name in losses.keys():
                weights[name] = torch.tensor(1.0, device=self.device)
            return weights
        
        # Compute features from loss history
        features = self._compute_features(losses)
        
        # Predict weights
        with torch.no_grad():
            weight_logits = self.weight_network(features)
        
        # Map to constraint names
        weights = {}
        for i, name in enumerate(self.constraint_names):
            if name in losses:
                if i < len(weight_logits):
                    weights[name] = weight_logits[i]
                else:
                    weights[name] = torch.tensor(1.0, device=self.device)
        
        # Add any missing constraints with default weight
        for name in losses.keys():
            if name not in weights:
                weights[name] = torch.tensor(1.0, device=self.device)
        
        return weights
    
    def _compute_features(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute features for weight prediction network."""
        features = []
        
        # Current loss values (normalized)
        current_losses = []
        for name in self.constraint_names:
            if name in losses:
                current_losses.append(losses[name].item())
            else:
                current_losses.append(0.0)
        
        # Normalize current losses
        max_loss = max(current_losses) if max(current_losses) > 0 else 1.0
        normalized_losses = [l / max_loss for l in current_losses]
        features.extend(normalized_losses)
        
        # Loss trends (if history available)
        if len(features) < 10:
            # Pad with zeros if not enough features
            features.extend([0.0] * (10 - len(features)))
        
        return torch.tensor(features[:10], dtype=torch.float32, device=self.device)
    
    def update(self, losses: Dict[str, torch.Tensor], iteration: int):
        """Update learned balancer."""
        # Update loss history
        for name, loss in losses.items():
            if name not in self.loss_history:
                self.loss_history[name] = []
            
            self.loss_history[name].append(loss.item())
            
            # Keep only recent history
            if len(self.loss_history[name]) > self.max_history:
                self.loss_history[name] = self.loss_history[name][-self.max_history:]
        
        # Train weight network periodically
        if iteration % 50 == 0 and iteration > 100:
            self._train_weight_network(losses)
    
    def _train_weight_network(self, current_losses: Dict[str, torch.Tensor]):
        """Train the weight prediction network."""
        # Simple training objective: minimize variance in loss magnitudes
        features = self._compute_features(current_losses)
        predicted_weights = self.weight_network(features)
        
        # Compute weighted losses
        weighted_losses = []
        for i, name in enumerate(self.constraint_names):
            if name in current_losses and i < len(predicted_weights):
                weighted_loss = current_losses[name] * predicted_weights[i]
                weighted_losses.append(weighted_loss)
        
        if weighted_losses:
            # Objective: minimize variance of weighted losses
            weighted_losses_tensor = torch.stack(weighted_losses)
            variance_loss = torch.var(weighted_losses_tensor)
            
            # Add regularization to prevent extreme weights
            weight_reg = torch.mean((predicted_weights - 1.0) ** 2)
            total_loss = variance_loss + 0.1 * weight_reg
            
            # Update weight network
            self.weight_optimizer.zero_grad()
            total_loss.backward()
            self.weight_optimizer.step()


class PhysicsRegularizer:
    """Physics regularization for meta-parameters to preserve PDE structure."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize physics regularizer.
        
        Args:
            config: Configuration containing regularization parameters
        """
        self.config = config
        self.regularization_weight = config.physics_regularization_weight
        self.consistency_weight = config.physics_consistency_weight
        
        # Store initial parameters for structure preservation
        self.initial_params = None
        self.param_structure_weights = {}
    
    def set_initial_params(self, params: Dict[str, torch.Tensor]):
        """Set initial parameters for structure preservation."""
        self.initial_params = {name: param.clone().detach() 
                              for name, param in params.items()}
        
        # Compute structure weights based on parameter importance
        self._compute_structure_weights(params)
    
    def _compute_structure_weights(self, params: Dict[str, torch.Tensor]):
        """Compute weights for different parameter types."""
        for name, param in params.items():
            if 'weight' in name:
                # Higher weight for network weights
                if 'linears.0' in name:  # First layer
                    self.param_structure_weights[name] = 2.0
                elif 'linears.-1' in name or 'linears.{}'.format(len(params)-1) in name:  # Last layer
                    self.param_structure_weights[name] = 1.5
                else:  # Hidden layers
                    self.param_structure_weights[name] = 1.0
            elif 'bias' in name:
                # Lower weight for biases
                self.param_structure_weights[name] = 0.5
            else:
                self.param_structure_weights[name] = 1.0
    
    def compute_regularization_loss(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute physics regularization loss."""
        if self.initial_params is None:
            return torch.tensor(0.0, device=list(params.values())[0].device)
        
        reg_loss = torch.tensor(0.0, device=list(params.values())[0].device)
        
        for name, param in params.items():
            if name in self.initial_params:
                # Structure preservation: penalize large deviations from initial structure
                initial_param = self.initial_params[name]
                param_diff = param - initial_param
                
                # Weighted L2 regularization
                weight = self.param_structure_weights.get(name, 1.0)
                reg_loss += weight * torch.norm(param_diff, p=2)
        
        return self.regularization_weight * reg_loss
    
    def compute_consistency_loss(self, params: Dict[str, torch.Tensor],
                               task_losses: List[torch.Tensor]) -> torch.Tensor:
        """Compute physics consistency loss across tasks."""
        if len(task_losses) < 2:
            return torch.tensor(0.0, device=list(params.values())[0].device)
        
        # Consistency: similar tasks should have similar parameter updates
        consistency_loss = torch.tensor(0.0, device=list(params.values())[0].device)
        
        # Compute variance in task losses as consistency measure
        task_losses_tensor = torch.stack(task_losses)
        loss_variance = torch.var(task_losses_tensor)
        
        # Penalize high variance (inconsistent performance across tasks)
        consistency_loss = self.consistency_weight * loss_variance
        
        return consistency_loss


def create_constraint_balancer(config: PhysicsInformedMetaLearnerConfig) -> ConstraintBalancer:
    """Factory function to create constraint balancer based on configuration.
    
    Args:
        config: Configuration specifying balancer type and parameters
        
    Returns:
        Constraint balancer instance
    """
    balancer_type = config.constraint_balancer_type.lower()
    
    if balancer_type == 'static':
        # Default static weights following PINNacle patterns
        weights = {
            'data_loss': config.data_loss_weight,
            'physics_loss': config.physics_loss_weight,
            'boundary_loss': config.boundary_loss_weight,
            'initial_loss': config.initial_loss_weight
        }
        return StaticConstraintBalancer(weights)
    
    elif balancer_type == 'dynamic':
        return DynamicConstraintBalancer(config)
    
    elif balancer_type == 'learned':
        return LearnedConstraintBalancer(config)
    
    else:
        raise ValueError(f"Unknown constraint balancer type: {balancer_type}")