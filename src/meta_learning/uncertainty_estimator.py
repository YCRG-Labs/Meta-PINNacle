"""Uncertainty estimation components for meta-learning in PINNs.

This module implements uncertainty estimation during the adaptation phase,
uncertainty-guided data sampling for support sets, and uncertainty-aware
constraint weighting mechanisms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

from .config import PhysicsInformedMetaLearnerConfig


class UncertaintyEstimator(ABC):
    """Abstract base class for uncertainty estimation methods."""
    
    @abstractmethod
    def estimate_uncertainty(self, model, inputs: torch.Tensor,
                           params: Optional[Dict[str, torch.Tensor]] = None,
                           n_samples: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty for given inputs.
        
        Args:
            model: Neural network model
            inputs: Input coordinates
            params: Optional model parameters
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        pass
    
    @abstractmethod
    def update(self, model, task_data, adaptation_step: int):
        """Update uncertainty estimator state during adaptation."""
        pass


class EnsembleUncertaintyEstimator(UncertaintyEstimator):
    """Ensemble-based uncertainty estimation using multiple forward passes."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize ensemble uncertainty estimator.
        
        Args:
            config: Configuration containing uncertainty parameters
        """
        self.config = config
        self.n_samples = config.n_uncertainty_samples
        self.ensemble_size = min(10, self.n_samples)  # Limit ensemble size
        
        # Store ensemble parameters during adaptation
        self.ensemble_params = []
        self.adaptation_history = []
        
    def estimate_uncertainty(self, model, inputs: torch.Tensor,
                           params: Optional[Dict[str, torch.Tensor]] = None,
                           n_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using ensemble of adapted parameters."""
        if n_samples is None:
            n_samples = self.n_samples
        
        if not self.ensemble_params:
            # If no ensemble available, use single prediction with zero uncertainty
            if hasattr(model, 'forward'):
                pred = model.forward(inputs, params)
            else:
                pred = model(inputs)
            uncertainty = torch.zeros_like(pred)
            return pred, uncertainty
        
        # Generate predictions from ensemble
        predictions = []
        ensemble_size = min(len(self.ensemble_params), n_samples)
        
        for i in range(ensemble_size):
            ensemble_param = self.ensemble_params[i % len(self.ensemble_params)]
            if hasattr(model, 'forward'):
                pred = model.forward(inputs, ensemble_param)
            else:
                pred = model(inputs)
            predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=0)  # [n_samples, batch_size, output_dim]
        
        # Compute mean and uncertainty (standard deviation)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def update(self, model, task_data, adaptation_step: int):
        """Update ensemble with current adapted parameters."""
        if hasattr(model, 'named_parameters'):
            current_params = {name: param.clone().detach() 
                            for name, param in model.named_parameters()}
            
            # Add to ensemble (with limited size)
            self.ensemble_params.append(current_params)
            if len(self.ensemble_params) > self.ensemble_size:
                self.ensemble_params.pop(0)  # Remove oldest
        
        # Store adaptation history for analysis
        self.adaptation_history.append({
            'step': adaptation_step,
            'ensemble_size': len(self.ensemble_params)
        })
    
    def reset_ensemble(self):
        """Reset ensemble for new task."""
        self.ensemble_params = []
        self.adaptation_history = []


class DropoutUncertaintyEstimator(UncertaintyEstimator):
    """Monte Carlo Dropout-based uncertainty estimation."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize dropout uncertainty estimator.
        
        Args:
            config: Configuration containing uncertainty parameters
        """
        self.config = config
        self.n_samples = config.n_uncertainty_samples
        self.dropout_rate = 0.1
        
        # Track which layers have dropout
        self.dropout_layers = {}
        
    def add_dropout_to_model(self, model):
        """Add dropout layers to model for uncertainty estimation."""
        # Add dropout to linear layers
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'linears' in name:
                # Insert dropout after linear layer
                dropout_name = f"{name}_dropout"
                self.dropout_layers[dropout_name] = nn.Dropout(self.dropout_rate)
    
    def estimate_uncertainty(self, model, inputs: torch.Tensor,
                           params: Optional[Dict[str, torch.Tensor]] = None,
                           n_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using Monte Carlo dropout."""
        if n_samples is None:
            n_samples = self.n_samples
        
        # Enable dropout during inference
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            if hasattr(model, 'forward'):
                pred = model.forward(inputs, params)
            else:
                pred = model(inputs)
            predictions.append(pred)
        
        # Restore evaluation mode
        model.eval()
        
        # Stack predictions and compute statistics
        predictions = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def update(self, model, task_data, adaptation_step: int):
        """Update dropout parameters if needed."""
        # Adaptive dropout rate based on adaptation progress
        if adaptation_step > 5:
            self.dropout_rate = max(0.05, self.dropout_rate * 0.95)


class BayesianUncertaintyEstimator(UncertaintyEstimator):
    """Bayesian uncertainty estimation using variational inference."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize Bayesian uncertainty estimator.
        
        Args:
            config: Configuration containing uncertainty parameters
        """
        self.config = config
        self.n_samples = config.n_uncertainty_samples
        
        # Variational parameters (mean and log variance)
        self.param_means = {}
        self.param_log_vars = {}
        self.initialized = False
        
    def initialize_variational_params(self, model):
        """Initialize variational parameters from model parameters."""
        for name, param in model.named_parameters():
            self.param_means[name] = param.clone().detach()
            self.param_log_vars[name] = torch.full_like(param, -3.0)  # Small initial variance
        self.initialized = True
    
    def sample_parameters(self) -> Dict[str, torch.Tensor]:
        """Sample parameters from variational distribution."""
        sampled_params = {}
        for name in self.param_means:
            mean = self.param_means[name]
            log_var = self.param_log_vars[name]
            std = torch.exp(0.5 * log_var)
            
            # Reparameterization trick
            eps = torch.randn_like(mean)
            sampled_params[name] = mean + std * eps
        
        return sampled_params
    
    def estimate_uncertainty(self, model, inputs: torch.Tensor,
                           params: Optional[Dict[str, torch.Tensor]] = None,
                           n_samples: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate uncertainty using Bayesian sampling."""
        if n_samples is None:
            n_samples = self.n_samples
        
        if not self.initialized:
            self.initialize_variational_params(model)
        
        predictions = []
        for _ in range(n_samples):
            # Sample parameters from variational distribution
            sampled_params = self.sample_parameters()
            
            # Forward pass with sampled parameters
            if hasattr(model, 'forward'):
                pred = model.forward(inputs, sampled_params)
            else:
                pred = model(inputs)
            predictions.append(pred)
        
        # Compute statistics
        predictions = torch.stack(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        uncertainty = torch.std(predictions, dim=0)
        
        return mean_pred, uncertainty
    
    def update(self, model, task_data, adaptation_step: int):
        """Update variational parameters during adaptation."""
        if not self.initialized:
            self.initialize_variational_params(model)
        
        # Update means with current parameters
        for name, param in model.named_parameters():
            if name in self.param_means:
                # Exponential moving average update
                alpha = 0.1
                self.param_means[name] = (1 - alpha) * self.param_means[name] + alpha * param.detach()
                
                # Update variance based on parameter change
                param_change = torch.norm(param.detach() - self.param_means[name])
                self.param_log_vars[name] = torch.clamp(
                    self.param_log_vars[name] + 0.01 * param_change,
                    min=-5.0, max=-1.0
                )


class UncertaintyGuidedSampler:
    """Uncertainty-guided data sampling for support sets."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        """Initialize uncertainty-guided sampler.
        
        Args:
            uncertainty_estimator: Uncertainty estimation method
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.sampling_strategy = 'uncertainty_weighted'  # 'random', 'uncertainty_weighted', 'high_uncertainty'
        
    def sample_support_data(self, model, available_data, k_shots: int,
                          params: Optional[Dict[str, torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample support data based on uncertainty estimates.
        
        Args:
            model: Neural network model
            available_data: Available data points to sample from
            k_shots: Number of shots to sample
            params: Optional model parameters
            
        Returns:
            Tuple of (sampled_inputs, sampled_outputs)
        """
        if len(available_data.inputs) <= k_shots:
            # Return all available data if not enough points
            return available_data.inputs, available_data.outputs
        
        if self.sampling_strategy == 'random':
            # Random sampling baseline
            indices = torch.randperm(len(available_data.inputs))[:k_shots]
            return available_data.inputs[indices], available_data.outputs[indices]
        
        # Estimate uncertainty for all available points
        _, uncertainties = self.uncertainty_estimator.estimate_uncertainty(
            model, available_data.inputs, params
        )
        
        # Aggregate uncertainty across output dimensions
        uncertainty_scores = torch.mean(uncertainties, dim=1)
        
        if self.sampling_strategy == 'uncertainty_weighted':
            # Sample with probability proportional to uncertainty
            probabilities = F.softmax(uncertainty_scores, dim=0)
            indices = torch.multinomial(probabilities, k_shots, replacement=False)
        
        elif self.sampling_strategy == 'high_uncertainty':
            # Sample points with highest uncertainty
            _, indices = torch.topk(uncertainty_scores, k_shots)
        
        else:
            # Default to random sampling
            indices = torch.randperm(len(available_data.inputs))[:k_shots]
        
        return available_data.inputs[indices], available_data.outputs[indices]
    
    def adaptive_sampling(self, model, available_data, current_support_data,
                         adaptation_step: int, params: Optional[Dict[str, torch.Tensor]] = None) -> bool:
        """Determine if additional sampling is needed during adaptation.
        
        Args:
            model: Neural network model
            available_data: Available data points
            current_support_data: Currently used support data
            adaptation_step: Current adaptation step
            params: Optional model parameters
            
        Returns:
            True if additional sampling is recommended
        """
        if adaptation_step < 2:
            return False  # Don't resample too early
        
        # Estimate uncertainty on current support set
        _, support_uncertainties = self.uncertainty_estimator.estimate_uncertainty(
            model, current_support_data.inputs, params
        )
        
        # Check if uncertainty is still high
        mean_uncertainty = torch.mean(support_uncertainties)
        uncertainty_threshold = 0.1  # Configurable threshold
        
        return mean_uncertainty > uncertainty_threshold


class UncertaintyAwareConstraintWeighting:
    """Uncertainty-aware constraint weighting for physics losses."""
    
    def __init__(self, uncertainty_estimator: UncertaintyEstimator):
        """Initialize uncertainty-aware constraint weighting.
        
        Args:
            uncertainty_estimator: Uncertainty estimation method
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.weighting_strategy = 'inverse_uncertainty'  # 'inverse_uncertainty', 'uncertainty_penalty'
        
    def compute_uncertainty_weights(self, model, task_data,
                                  params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute constraint weights based on uncertainty estimates.
        
        Args:
            model: Neural network model
            task_data: Task data
            params: Optional model parameters
            
        Returns:
            Dictionary of uncertainty-based weights
        """
        weights = {}
        
        # Estimate uncertainty on collocation points (physics loss)
        if task_data.collocation_points is not None:
            _, physics_uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                model, task_data.collocation_points, params
            )
            physics_uncertainty_mean = torch.mean(physics_uncertainty)
            
            if self.weighting_strategy == 'inverse_uncertainty':
                # Higher uncertainty -> lower physics weight (focus on data fitting)
                weights['physics_loss'] = 1.0 / (1.0 + physics_uncertainty_mean)
                weights['data_loss'] = 1.0 + physics_uncertainty_mean
            else:  # uncertainty_penalty
                # Higher uncertainty -> higher physics weight (enforce physics more)
                weights['physics_loss'] = 1.0 + physics_uncertainty_mean
                weights['data_loss'] = 1.0 / (1.0 + physics_uncertainty_mean)
        
        # Estimate uncertainty on boundary points
        if task_data.boundary_data is not None:
            boundary_inputs = task_data.boundary_data[:, :-1]
            _, boundary_uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                model, boundary_inputs, params
            )
            boundary_uncertainty_mean = torch.mean(boundary_uncertainty)
            weights['boundary_loss'] = 1.0 + boundary_uncertainty_mean
        
        # Estimate uncertainty on initial condition points
        if task_data.initial_data is not None:
            initial_inputs = task_data.initial_data[:, :-1]
            _, initial_uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                model, initial_inputs, params
            )
            initial_uncertainty_mean = torch.mean(initial_uncertainty)
            weights['initial_loss'] = 1.0 + initial_uncertainty_mean
        
        # Convert to tensors with appropriate device
        device = task_data.inputs.device
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                weights[key] = torch.tensor(value, device=device)
        
        return weights


def create_uncertainty_estimator(config: PhysicsInformedMetaLearnerConfig) -> UncertaintyEstimator:
    """Factory function to create uncertainty estimator based on configuration.
    
    Args:
        config: Configuration specifying uncertainty estimation method
        
    Returns:
        Uncertainty estimator instance
    """
    if not config.uncertainty_estimation:
        # Return dummy estimator that does nothing
        class DummyUncertaintyEstimator(UncertaintyEstimator):
            def estimate_uncertainty(self, model, inputs, params=None, n_samples=10):
                if hasattr(model, 'forward'):
                    pred = model.forward(inputs, params)
                else:
                    pred = model(inputs)
                uncertainty = torch.zeros_like(pred)
                return pred, uncertainty
            
            def update(self, model, task_data, adaptation_step):
                pass
        
        return DummyUncertaintyEstimator()
    
    method = config.uncertainty_method.lower()
    
    if method == 'ensemble':
        return EnsembleUncertaintyEstimator(config)
    elif method == 'dropout':
        return DropoutUncertaintyEstimator(config)
    elif method == 'bayesian':
        return BayesianUncertaintyEstimator(config)
    else:
        raise ValueError(f"Unknown uncertainty estimation method: {method}")


class UncertaintyAnalyzer:
    """Analyzer for uncertainty estimation performance and behavior."""
    
    def __init__(self):
        """Initialize uncertainty analyzer."""
        self.uncertainty_history = []
        self.calibration_data = []
        
    def log_uncertainty(self, predictions: torch.Tensor, uncertainties: torch.Tensor,
                       targets: torch.Tensor, iteration: int):
        """Log uncertainty estimates for analysis.
        
        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            targets: True targets
            iteration: Current iteration
        """
        # Compute prediction errors
        errors = torch.abs(predictions - targets)
        
        # Store for analysis
        self.uncertainty_history.append({
            'iteration': iteration,
            'mean_uncertainty': torch.mean(uncertainties).item(),
            'mean_error': torch.mean(errors).item(),
            'uncertainty_error_correlation': torch.corrcoef(
                torch.stack([uncertainties.flatten(), errors.flatten()])
            )[0, 1].item()
        })
    
    def compute_calibration_metrics(self) -> Dict[str, float]:
        """Compute uncertainty calibration metrics."""
        if not self.uncertainty_history:
            return {}
        
        # Extract data
        uncertainties = [entry['mean_uncertainty'] for entry in self.uncertainty_history]
        errors = [entry['mean_error'] for entry in self.uncertainty_history]
        correlations = [entry['uncertainty_error_correlation'] for entry in self.uncertainty_history]
        
        return {
            'mean_uncertainty': np.mean(uncertainties),
            'mean_error': np.mean(errors),
            'mean_correlation': np.mean([c for c in correlations if not np.isnan(c)]),
            'uncertainty_std': np.std(uncertainties),
            'error_std': np.std(errors)
        }
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get summary of uncertainty estimation behavior."""
        calibration = self.compute_calibration_metrics()
        
        return {
            'n_samples': len(self.uncertainty_history),
            'calibration_metrics': calibration,
            'history': self.uncertainty_history[-10:] if self.uncertainty_history else []  # Last 10 entries
        }