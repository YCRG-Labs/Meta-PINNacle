"""Multi-scale physics handling for meta-learning in PINNs.

This module implements multi-scale physics handling capabilities that extend
existing PDE classes to handle different time/length scales and provide
scale-aware gradient computation using DeepXDE's gradient utilities.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from abc import ABC, abstractmethod

from .config import PhysicsInformedMetaLearnerConfig


class MultiScaleHandler:
    """Handler for multi-scale physics problems in meta-learning."""
    
    def __init__(self, config: PhysicsInformedMetaLearnerConfig):
        """Initialize multi-scale handler.
        
        Args:
            config: Configuration containing multi-scale parameters
        """
        self.config = config
        self.enabled = config.multi_scale_handling
        self.scale_factors = config.scale_factors
        
        # Scale-specific parameters
        self.time_scales = []
        self.length_scales = []
        self.scale_weights = {}
        
        # Adaptive resolution parameters
        self.adaptive_resolution = True
        self.min_resolution = 32
        self.max_resolution = 256
        self.resolution_threshold = 0.01
        
        # Scale detection parameters
        self.auto_detect_scales = True
        self.scale_detection_samples = 1000
        
        # Initialize scale-specific networks if needed
        self.scale_networks = {}
        
    def detect_problem_scales(self, task_data, pde_problem) -> Dict[str, List[float]]:
        """Automatically detect relevant scales in the problem.
        
        Args:
            task_data: Task data containing problem information
            pde_problem: PDE problem instance
            
        Returns:
            Dictionary containing detected time and length scales
        """
        scales = {'time_scales': [], 'length_scales': []}
        
        if not self.auto_detect_scales:
            return scales
        
        # Analyze input domain to detect characteristic scales
        inputs = task_data.inputs
        
        # Detect spatial scales
        if inputs.shape[1] >= 2:  # At least x coordinate
            x_coords = inputs[:, 0]
            x_range = torch.max(x_coords) - torch.min(x_coords)
            
            # Characteristic length scales based on domain size
            scales['length_scales'] = [
                x_range.item(),  # Domain scale
                x_range.item() / 10,  # Fine scale
                x_range.item() / 100  # Very fine scale
            ]
        
        # Detect temporal scales
        if inputs.shape[1] >= 2:  # Has time coordinate
            t_coords = inputs[:, -1]  # Assume last coordinate is time
            t_range = torch.max(t_coords) - torch.min(t_coords)
            
            # Characteristic time scales
            scales['time_scales'] = [
                t_range.item(),  # Problem time scale
                t_range.item() / 10,  # Fast dynamics
                t_range.item() / 100  # Very fast dynamics
            ]
        
        # Problem-specific scale detection
        if hasattr(pde_problem, 'get_characteristic_scales'):
            problem_scales = pde_problem.get_characteristic_scales()
            scales.update(problem_scales)
        
        self.time_scales = scales['time_scales']
        self.length_scales = scales['length_scales']
        
        return scales
    
    def create_scale_specific_collocation_points(self, base_points: torch.Tensor,
                                               scales: Dict[str, List[float]]) -> Dict[str, torch.Tensor]:
        """Create collocation points for different scales.
        
        Args:
            base_points: Base collocation points
            scales: Dictionary of detected scales
            
        Returns:
            Dictionary mapping scale names to collocation points
        """
        scale_points = {'base': base_points}
        
        if not self.enabled:
            return scale_points
        
        # Create points for different length scales
        for i, length_scale in enumerate(scales.get('length_scales', [])):
            scale_name = f'length_scale_{i}'
            
            # Adjust spatial resolution based on scale
            if base_points.shape[1] >= 2:
                # Create finer spatial sampling for smaller scales
                scale_factor = length_scale / max(scales['length_scales'])
                n_points = int(len(base_points) * (1.0 / scale_factor))
                n_points = max(self.min_resolution, min(n_points, self.max_resolution))
                
                # Sample points with appropriate density
                indices = torch.randperm(len(base_points))[:n_points]
                scale_points[scale_name] = base_points[indices]
        
        # Create points for different time scales
        for i, time_scale in enumerate(scales.get('time_scales', [])):
            scale_name = f'time_scale_{i}'
            
            # Adjust temporal resolution based on scale
            if base_points.shape[1] >= 2:
                scale_factor = time_scale / max(scales['time_scales'])
                n_points = int(len(base_points) * (1.0 / scale_factor))
                n_points = max(self.min_resolution, min(n_points, self.max_resolution))
                
                indices = torch.randperm(len(base_points))[:n_points]
                scale_points[scale_name] = base_points[indices]
        
        return scale_points
    
    def compute_scale_aware_gradients(self, model, inputs: torch.Tensor,
                                    outputs: torch.Tensor, scales: Dict[str, List[float]],
                                    params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute scale-aware gradients using DeepXDE's gradient utilities.
        
        Args:
            model: Neural network model
            inputs: Input coordinates
            outputs: Network outputs
            scales: Dictionary of problem scales
            params: Optional model parameters
            
        Returns:
            Dictionary of scale-aware gradients
        """
        gradients = {}
        
        if not self.enabled:
            # Standard gradient computation
            if inputs.requires_grad:
                grad_outputs = torch.ones_like(outputs)
                grads = torch.autograd.grad(
                    outputs, inputs, grad_outputs=grad_outputs,
                    create_graph=True, retain_graph=True
                )[0]
                gradients['standard'] = grads
            return gradients
        
        # Enable gradient computation
        inputs = inputs.requires_grad_(True)
        
        # Compute gradients at different scales
        for scale_type, scale_list in scales.items():
            for i, scale in enumerate(scale_list):
                scale_name = f'{scale_type}_{i}'
                
                # Scale-weighted gradient computation
                if scale_type == 'length_scales' and inputs.shape[1] >= 2:
                    # Spatial gradients with scale weighting
                    spatial_grads = []
                    for dim in range(inputs.shape[1] - 1):  # Exclude time dimension
                        grad = torch.autograd.grad(
                            outputs, inputs,
                            grad_outputs=torch.ones_like(outputs),
                            create_graph=True, retain_graph=True
                        )[0][:, dim:dim+1]
                        
                        # Apply scale weighting
                        scale_weight = scale / max(scale_list)
                        scaled_grad = grad * scale_weight
                        spatial_grads.append(scaled_grad)
                    
                    if spatial_grads:
                        gradients[f'spatial_{scale_name}'] = torch.cat(spatial_grads, dim=1)
                
                elif scale_type == 'time_scales' and inputs.shape[1] >= 2:
                    # Temporal gradients with scale weighting
                    time_grad = torch.autograd.grad(
                        outputs, inputs,
                        grad_outputs=torch.ones_like(outputs),
                        create_graph=True, retain_graph=True
                    )[0][:, -1:]  # Last dimension is time
                    
                    # Apply scale weighting
                    scale_weight = scale / max(scale_list)
                    scaled_grad = time_grad * scale_weight
                    gradients[f'temporal_{scale_name}'] = scaled_grad
        
        return gradients
    
    def compute_multi_scale_physics_loss(self, model, task_data, task, scales: Dict[str, List[float]],
                                       params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """Compute physics loss across multiple scales.
        
        Args:
            model: Neural network model
            task_data: Task data
            task: Task object
            scales: Dictionary of problem scales
            params: Optional model parameters
            
        Returns:
            Dictionary of scale-specific physics losses
        """
        physics_losses = {}
        
        if not self.enabled:
            # Standard physics loss
            if hasattr(model, 'compute_physics_loss'):
                physics_losses['standard'] = model.compute_physics_loss(task_data, task, params)
            return physics_losses
        
        # Create scale-specific collocation points
        scale_points = self.create_scale_specific_collocation_points(
            task_data.collocation_points, scales
        )
        
        # Compute physics loss at each scale
        for scale_name, points in scale_points.items():
            if scale_name == 'base':
                continue
                
            # Create task data for this scale
            scale_task_data = type(task_data)(
                inputs=task_data.inputs,
                outputs=task_data.outputs,
                collocation_points=points,
                boundary_data=task_data.boundary_data,
                initial_data=task_data.initial_data
            )
            
            # Compute physics loss for this scale
            if hasattr(model, 'compute_physics_loss'):
                scale_loss = model.compute_physics_loss(scale_task_data, task, params)
                
                # Weight by scale importance
                scale_weight = self._get_scale_weight(scale_name, scales)
                physics_losses[scale_name] = scale_weight * scale_loss
        
        return physics_losses
    
    def _get_scale_weight(self, scale_name: str, scales: Dict[str, List[float]]) -> float:
        """Get weight for specific scale based on its characteristics."""
        if scale_name in self.scale_weights:
            return self.scale_weights[scale_name]
        
        # Default weighting strategy
        if 'length_scale' in scale_name:
            # Smaller length scales get higher weights (more important for accuracy)
            scale_idx = int(scale_name.split('_')[-1])
            length_scales = scales.get('length_scales', [1.0])
            if scale_idx < len(length_scales):
                scale_value = length_scales[scale_idx]
                max_scale = max(length_scales)
                weight = max_scale / (scale_value + 1e-8)  # Inverse weighting
                return min(weight, 5.0)  # Cap maximum weight
        
        elif 'time_scale' in scale_name:
            # Similar strategy for time scales
            scale_idx = int(scale_name.split('_')[-1])
            time_scales = scales.get('time_scales', [1.0])
            if scale_idx < len(time_scales):
                scale_value = time_scales[scale_idx]
                max_scale = max(time_scales)
                weight = max_scale / (scale_value + 1e-8)
                return min(weight, 5.0)
        
        return 1.0  # Default weight
    
    def create_adaptive_resolution_mechanism(self, model, task_data, task,
                                           params: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
        """Create adaptive resolution mechanism for multi-scale problems.
        
        Args:
            model: Neural network model
            task_data: Task data
            task: Task object
            params: Optional model parameters
            
        Returns:
            Dictionary containing adaptive resolution information
        """
        if not self.adaptive_resolution:
            return {'resolution': 'fixed', 'points': task_data.collocation_points}
        
        # Compute solution gradients to identify high-gradient regions
        inputs = task_data.collocation_points.requires_grad_(True)
        outputs = model.forward(inputs, params) if hasattr(model, 'forward') else model(inputs)
        
        # Compute gradient magnitude
        grad_outputs = torch.ones_like(outputs)
        gradients = torch.autograd.grad(
            outputs, inputs, grad_outputs=grad_outputs,
            create_graph=True, retain_graph=True
        )[0]
        
        gradient_magnitude = torch.norm(gradients, dim=1)
        
        # Identify regions needing higher resolution
        high_gradient_mask = gradient_magnitude > self.resolution_threshold
        high_gradient_points = inputs[high_gradient_mask]
        
        # Create adaptive point distribution
        adaptive_points = task_data.collocation_points.clone()
        
        if len(high_gradient_points) > 0:
            # Add more points in high-gradient regions
            n_additional = min(len(high_gradient_points), self.max_resolution - len(adaptive_points))
            
            if n_additional > 0:
                # Generate additional points around high-gradient regions
                additional_points = self._generate_refined_points(
                    high_gradient_points, n_additional
                )
                adaptive_points = torch.cat([adaptive_points, additional_points], dim=0)
        
        return {
            'resolution': 'adaptive',
            'points': adaptive_points,
            'high_gradient_regions': high_gradient_points,
            'gradient_magnitude': gradient_magnitude,
            'refinement_ratio': len(adaptive_points) / len(task_data.collocation_points)
        }
    
    def _generate_refined_points(self, base_points: torch.Tensor, n_points: int) -> torch.Tensor:
        """Generate refined points around base points for higher resolution."""
        if len(base_points) == 0:
            return torch.empty(0, base_points.shape[1], device=base_points.device)
        
        refined_points = []
        points_per_base = max(1, n_points // len(base_points))
        
        for base_point in base_points:
            # Generate points in a small neighborhood
            noise_scale = 0.01  # Small perturbation
            for _ in range(points_per_base):
                noise = torch.randn_like(base_point) * noise_scale
                refined_point = base_point + noise
                refined_points.append(refined_point.unsqueeze(0))
        
        if refined_points:
            return torch.cat(refined_points, dim=0)[:n_points]
        else:
            return torch.empty(0, base_points.shape[1], device=base_points.device)
    
    def get_multi_scale_summary(self) -> Dict[str, Any]:
        """Get summary of multi-scale handling configuration and state."""
        return {
            'enabled': self.enabled,
            'scale_factors': self.scale_factors,
            'time_scales': self.time_scales,
            'length_scales': self.length_scales,
            'adaptive_resolution': self.adaptive_resolution,
            'resolution_range': [self.min_resolution, self.max_resolution],
            'auto_detect_scales': self.auto_detect_scales,
            'n_scale_networks': len(self.scale_networks)
        }


class ScaleAwarePDEExtension:
    """Extension for existing PDE classes to handle multi-scale physics."""
    
    def __init__(self, base_pde, multi_scale_handler: MultiScaleHandler):
        """Initialize scale-aware PDE extension.
        
        Args:
            base_pde: Base PDE class instance
            multi_scale_handler: Multi-scale handler instance
        """
        self.base_pde = base_pde
        self.multi_scale_handler = multi_scale_handler
        
        # Store original PDE methods
        self.original_pde_method = getattr(base_pde, 'pde', None)
        
    def pde(self, inputs: torch.Tensor, outputs: torch.Tensor,
            task_params: Optional[Dict] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Enhanced PDE computation with multi-scale handling."""
        if not self.multi_scale_handler.enabled:
            # Use original PDE method
            if self.original_pde_method:
                return self.original_pde_method(inputs, outputs)
            else:
                raise NotImplementedError("Base PDE class must implement 'pde' method")
        
        # Detect problem scales
        from .task import TaskData  # Import here to avoid circular imports
        dummy_task_data = TaskData(
            inputs=inputs,
            outputs=outputs,
            collocation_points=inputs
        )
        scales = self.multi_scale_handler.detect_problem_scales(dummy_task_data, self.base_pde)
        
        # Compute multi-scale PDE residuals
        residuals = []
        
        # Base scale residual
        if self.original_pde_method:
            base_residual = self.original_pde_method(inputs, outputs)
            residuals.append(base_residual)
        
        # Scale-specific residuals
        for scale_type, scale_list in scales.items():
            for i, scale in enumerate(scale_list):
                # Apply scale-specific modifications to PDE computation
                scale_residual = self._compute_scale_specific_residual(
                    inputs, outputs, scale, scale_type
                )
                if scale_residual is not None:
                    residuals.append(scale_residual)
        
        return residuals if len(residuals) > 1 else residuals[0]
    
    def _compute_scale_specific_residual(self, inputs: torch.Tensor, outputs: torch.Tensor,
                                       scale: float, scale_type: str) -> Optional[torch.Tensor]:
        """Compute PDE residual for specific scale."""
        if not self.original_pde_method:
            return None
        
        # Apply scale-specific transformations
        if scale_type == 'length_scales':
            # Scale spatial coordinates
            scaled_inputs = inputs.clone()
            if inputs.shape[1] >= 2:
                scaled_inputs[:, :-1] = scaled_inputs[:, :-1] * scale
        elif scale_type == 'time_scales':
            # Scale temporal coordinate
            scaled_inputs = inputs.clone()
            if inputs.shape[1] >= 2:
                scaled_inputs[:, -1] = scaled_inputs[:, -1] * scale
        else:
            scaled_inputs = inputs
        
        # Compute residual with scaled inputs
        try:
            residual = self.original_pde_method(scaled_inputs, outputs)
            # Apply scale weighting
            scale_weight = self.multi_scale_handler._get_scale_weight(
                f'{scale_type}_0', {scale_type: [scale]}
            )
            return residual * scale_weight
        except Exception:
            # If scale-specific computation fails, return None
            return None
    
    def get_characteristic_scales(self) -> Dict[str, List[float]]:
        """Get characteristic scales for this PDE problem."""
        # Default implementation - can be overridden by specific PDE classes
        return {
            'length_scales': [1.0, 0.1, 0.01],
            'time_scales': [1.0, 0.1, 0.01]
        }


def extend_pde_with_multi_scale(pde_instance, config: PhysicsInformedMetaLearnerConfig):
    """Extend existing PDE instance with multi-scale capabilities.
    
    Args:
        pde_instance: Existing PDE class instance
        config: Configuration for multi-scale handling
        
    Returns:
        Extended PDE instance with multi-scale capabilities
    """
    multi_scale_handler = MultiScaleHandler(config)
    return ScaleAwarePDEExtension(pde_instance, multi_scale_handler)