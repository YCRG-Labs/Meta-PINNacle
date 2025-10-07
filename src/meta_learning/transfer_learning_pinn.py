"""TransferLearningPINN implementation using existing PINNacle infrastructure.

This module implements transfer learning approaches for physics-informed neural networks,
leveraging PINNacle's existing trainer.py parallel capabilities and loss weighting system.
"""

import copy
import time
import random
from typing import Dict, List, Optional, Tuple, Any, Union
import torch
import torch.nn as nn
import numpy as np
import deepxde as dde
from deepxde.model import Model

from .config import TransferLearningPINNConfig
from .task import Task, TaskData, TaskBatch
from ..model import FNN


class TransferLearningPINN:
    """Transfer learning baseline for Physics-Informed Neural Networks.
    
    This class implements multi-task pre-training and fine-tuning strategies
    using PINNacle's existing training infrastructure and parallel capabilities.
    """
    
    def __init__(self, config: TransferLearningPINNConfig):
        """Initialize TransferLearningPINN with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize network architecture using PINNacle's FNN
        self.network = FNN(
            layer_sizes=config.layers,
            activation=config.activation,
            kernel_initializer=config.initializer
        ).to(self.device)
        
        # Transfer learning parameters
        self.pretrain_tasks = config.pretrain_tasks
        self.pretrain_epochs_per_task = config.pretrain_epochs_per_task
        self.fine_tune_strategy = config.fine_tune_strategy
        self.fine_tune_epochs = config.fine_tune_epochs
        self.fine_tune_lr = config.fine_tune_lr
        
        # Physics loss weights (following PINNacle's loss_config pattern)
        self.physics_loss_weight = config.physics_loss_weight
        self.data_loss_weight = config.data_loss_weight
        self.boundary_loss_weight = config.boundary_loss_weight
        self.initial_loss_weight = config.initial_loss_weight
        
        # Initialize optimizers
        self.pretrain_optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=config.meta_lr
        )
        self.fine_tune_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.fine_tune_lr
        )
        
        # Training state
        self.pretrain_iteration = 0
        self.fine_tune_iteration = 0
        self.training_history = []
        self.pretrained = False
        
        # Current PDE problem (set during training)
        self.current_pde = None
        
        # Source distribution cache for efficient transfer
        self.source_distribution_cache = {}
        self.cache_size = config.source_distribution_cache_size
        
        # Domain adaptation parameters
        self.domain_adaptation = config.domain_adaptation
        self.domain_adaptation_weight = config.domain_adaptation_weight
        
        # Layer freezing schedule for gradual unfreezing
        self.unfreeze_schedule = config.unfreeze_schedule
        self.frozen_layers = set()
        
    def set_pde_problem(self, pde_problem):
        """Set the current PDE problem for physics loss computation."""
        self.current_pde = pde_problem
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)
    
    def compute_data_loss(self, task_data: TaskData) -> torch.Tensor:
        """Compute data fitting loss (MSE between predictions and targets)."""
        predictions = self.forward(task_data.inputs)
        return torch.nn.functional.mse_loss(predictions, task_data.outputs)
    
    def compute_physics_loss(self, task_data: TaskData, task: Task) -> torch.Tensor:
        """Compute physics-informed loss using PDE residual."""
        if self.current_pde is None:
            raise ValueError("PDE problem not set. Call set_pde_problem() first.")
        
        # Enable gradient computation for collocation points
        collocation_points = task_data.collocation_points.clone().detach().requires_grad_(True)
        
        # Forward pass to get network predictions
        u_pred = self.forward(collocation_points)
        
        # Compute PDE residual using the current PDE's function
        pde_residual = self.current_pde.pde(collocation_points, u_pred)
        
        # Handle different return types from PDE function
        if isinstance(pde_residual, (list, tuple)):
            physics_loss = sum(torch.mean(residual**2) for residual in pde_residual)
        else:
            physics_loss = torch.mean(pde_residual**2)
        
        return physics_loss
    
    def compute_boundary_loss(self, task_data: TaskData, task: Task) -> torch.Tensor:
        """Compute boundary condition loss."""
        if task_data.boundary_data is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        boundary_inputs = task_data.boundary_data[:, :-1]
        boundary_targets = task_data.boundary_data[:, -1:]
        
        boundary_pred = self.forward(boundary_inputs)
        return torch.nn.functional.mse_loss(boundary_pred, boundary_targets)
    
    def compute_initial_loss(self, task_data: TaskData, task: Task) -> torch.Tensor:
        """Compute initial condition loss."""
        if task_data.initial_data is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        initial_inputs = task_data.initial_data[:, :-1]
        initial_targets = task_data.initial_data[:, -1:]
        
        initial_pred = self.forward(initial_inputs)
        return torch.nn.functional.mse_loss(initial_pred, initial_targets)
    
    def compute_domain_adaptation_loss(self, source_data: TaskData, target_data: TaskData) -> torch.Tensor:
        """Compute domain adaptation loss using conditional shift penalization."""
        if not self.domain_adaptation:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Extract features from both domains
        source_features = self._extract_features(source_data.inputs)
        target_features = self._extract_features(target_data.inputs)
        
        # Compute Maximum Mean Discrepancy (MMD) for domain adaptation
        mmd_loss = self._compute_mmd_loss(source_features, target_features)
        
        return mmd_loss
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from the network for domain adaptation."""
        # Forward through all layers except the last one
        current_x = x
        
        # Apply input transform if exists
        if hasattr(self.network, '_input_transform') and self.network._input_transform is not None:
            current_x = self.network._input_transform(current_x)
        
        # Forward through all but last layer
        for i, linear_layer in enumerate(self.network.linears[:-1]):
            current_x = linear_layer(current_x)
            
            # Apply activation
            if self.config.activation == 'sin':
                current_x = torch.sin(current_x)
            elif self.config.activation == 'tanh':
                current_x = torch.tanh(current_x)
            else:
                activation_fn = getattr(torch.nn.functional, self.config.activation)
                current_x = activation_fn(current_x)
        
        return current_x
    
    def _compute_mmd_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss for domain adaptation."""
        def gaussian_kernel(x, y, sigma=1.0):
            """Gaussian RBF kernel."""
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist**2 / (2 * sigma**2))
        
        # Compute kernel matrices
        k_ss = gaussian_kernel(source_features, source_features)
        k_tt = gaussian_kernel(target_features, target_features)
        k_st = gaussian_kernel(source_features, target_features)
        
        # Compute MMD
        mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
        
        return mmd
    
    def compute_multi_task_loss(self, task_batch: TaskBatch) -> torch.Tensor:
        """Compute multi-task loss leveraging PINNacle's loss weighting system."""
        total_loss = 0.0
        task_weights = self._compute_task_weights(task_batch)
        
        for i, task in enumerate(task_batch.tasks):
            # Compute individual task loss
            task_loss = self.compute_total_loss(task.support_data, task)
            
            # Weight by task importance
            weighted_loss = task_weights[i] * task_loss
            total_loss += weighted_loss
        
        return total_loss / len(task_batch.tasks)
    
    def _compute_task_weights(self, task_batch: TaskBatch) -> torch.Tensor:
        """Compute task weights for multi-task learning."""
        # Simple uniform weighting for now
        # Can be extended with more sophisticated weighting schemes
        return torch.ones(len(task_batch.tasks), device=self.device)
    
    def compute_total_loss(self, task_data: TaskData, task: Task) -> torch.Tensor:
        """Compute total loss combining data, physics, boundary, and initial losses."""
        data_loss = self.compute_data_loss(task_data)
        physics_loss = self.compute_physics_loss(task_data, task)
        boundary_loss = self.compute_boundary_loss(task_data, task)
        initial_loss = self.compute_initial_loss(task_data, task)
        
        total_loss = (
            self.data_loss_weight * data_loss +
            self.physics_loss_weight * physics_loss +
            self.boundary_loss_weight * boundary_loss +
            self.initial_loss_weight * initial_loss
        )
        
        return total_loss
    
    def multi_task_pretrain(self, train_tasks: List[Task], val_tasks: Optional[List[Task]] = None,
                           **train_args) -> Dict[str, Any]:
        """Multi-task pre-training using existing trainer.py parallel capabilities.
        
        This method implements shared representation learning across parametric PDE tasks
        using PINNacle's existing parallel training infrastructure.
        """
        print(f"Starting multi-task pre-training on {len(train_tasks)} tasks...")
        
        # Initialize training history
        self.training_history = []
        best_val_loss = float('inf')
        
        start_time = time.time()
        
        # Multi-task training loop
        for epoch in range(self.pretrain_epochs_per_task):
            # Sample batch of tasks for multi-task training
            task_indices = np.random.choice(len(train_tasks), min(32, len(train_tasks)), replace=False)
            task_batch = TaskBatch([train_tasks[i] for i in task_indices])
            
            # Compute multi-task loss
            self.pretrain_optimizer.zero_grad()
            multi_task_loss = self.compute_multi_task_loss(task_batch)
            multi_task_loss.backward()
            
            # Apply gradient clipping if specified
            if hasattr(self.config, 'grad_clip_norm') and self.config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.grad_clip_norm)
            
            self.pretrain_optimizer.step()
            self.pretrain_iteration += 1
            
            # Validation step
            if val_tasks and epoch % 100 == 0:
                val_loss = self._validate_pretrain(val_tasks)
                
                # Update best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.best_pretrain_state = copy.deepcopy(self.network.state_dict())
                
                # Log progress
                print(f"Epoch {epoch}: train_loss={multi_task_loss.item():.6f}, val_loss={val_loss:.6f}")
                
                # Store training history
                self.training_history.append({
                    'epoch': epoch,
                    'train_loss': multi_task_loss.item(),
                    'val_loss': val_loss
                })
            
            elif epoch % 100 == 0:
                print(f"Epoch {epoch}: train_loss={multi_task_loss.item():.6f}")
        
        # Cache source distribution for efficient transfer
        self._cache_source_distribution(train_tasks)
        
        self.pretrained = True
        total_time = time.time() - start_time
        print(f"Multi-task pre-training completed in {total_time:.2f} seconds")
        
        return {
            'training_history': self.training_history,
            'best_val_loss': best_val_loss,
            'total_time': total_time,
            'pretrain_iterations': self.pretrain_iteration
        }
    
    def _validate_pretrain(self, val_tasks: List[Task]) -> float:
        """Validate pre-training on validation tasks."""
        self.network.eval()
        val_losses = []
        
        with torch.no_grad():
            for task in val_tasks:
                task_loss = self.compute_total_loss(task.support_data, task)
                val_losses.append(task_loss.item())
        
        self.network.train()
        return np.mean(val_losses)
    
    def _cache_source_distribution(self, source_tasks: List[Task]):
        """Cache source distribution for efficient transfer."""
        print("Caching source distribution...")
        
        # Sample representative tasks for caching
        cache_tasks = random.sample(source_tasks, min(self.cache_size, len(source_tasks)))
        
        self.network.eval()
        with torch.no_grad():
            for i, task in enumerate(cache_tasks):
                # Extract features and store task information
                features = self._extract_features(task.support_data.inputs)
                
                self.source_distribution_cache[i] = {
                    'features': features.cpu(),
                    'task_params': task.parameters,
                    'problem_type': task.problem_type
                }
        
        self.network.train()
        print(f"Cached {len(self.source_distribution_cache)} source tasks")
    
    def fine_tune(self, target_task: Task, k_shots: int, strategy: Optional[str] = None,
                  epochs: Optional[int] = None) -> 'TransferLearningPINN':
        """Fine-tune pre-trained model on target task with different strategies.
        
        Implements full, feature extraction, and gradual unfreezing strategies
        with layer-wise fine-tuning using existing network architectures.
        """
        if not self.pretrained:
            raise ValueError("Model must be pre-trained before fine-tuning. Call multi_task_pretrain() first.")
        
        if k_shots not in [1, 5, 10, 25]:
            raise ValueError(f"k_shots must be in [1, 5, 10, 25], got {k_shots}")
        
        if strategy is None:
            strategy = self.fine_tune_strategy
        
        if epochs is None:
            epochs = self.fine_tune_epochs
        
        # Sample K support points
        support_data = target_task.support_data
        if len(support_data) > k_shots:
            support_data = support_data.sample(k_shots)
        elif len(support_data) < k_shots:
            print(f"Warning: Only {len(support_data)} samples available, using all for {k_shots}-shot fine-tuning")
        
        # Create fine-tuned model copy
        fine_tuned_model = copy.deepcopy(self)
        
        # Apply fine-tuning strategy
        if strategy == 'full':
            fine_tuned_model._full_fine_tuning(target_task, support_data, epochs)
        elif strategy == 'feature_extraction':
            fine_tuned_model._feature_extraction_fine_tuning(target_task, support_data, epochs)
        elif strategy == 'gradual_unfreezing':
            fine_tuned_model._gradual_unfreezing_fine_tuning(target_task, support_data, epochs)
        else:
            raise ValueError(f"Unknown fine-tuning strategy: {strategy}")
        
        return fine_tuned_model
    
    def _full_fine_tuning(self, target_task: Task, support_data: TaskData, epochs: int):
        """Full fine-tuning strategy - all parameters are trainable."""
        print(f"Starting full fine-tuning for {epochs} epochs...")
        
        # Ensure all parameters are trainable
        for param in self.network.parameters():
            param.requires_grad = True
        
        # Reset optimizer for fine-tuning
        self.fine_tune_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.fine_tune_lr
        )
        
        # Fine-tuning loop
        for epoch in range(epochs):
            self.fine_tune_optimizer.zero_grad()
            
            # Compute loss on support data
            loss = self.compute_total_loss(support_data, target_task)
            
            # Add domain adaptation loss if enabled
            if self.domain_adaptation and hasattr(self, 'source_distribution_cache'):
                domain_loss = self._compute_domain_adaptation_loss_cached(support_data, target_task)
                loss += self.domain_adaptation_weight * domain_loss
            
            loss.backward()
            self.fine_tune_optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Fine-tuning epoch {epoch}: loss={loss.item():.6f}")
        
        print("Full fine-tuning completed")
    
    def _feature_extraction_fine_tuning(self, target_task: Task, support_data: TaskData, epochs: int):
        """Feature extraction strategy - only final layer is trainable."""
        print(f"Starting feature extraction fine-tuning for {epochs} epochs...")
        
        # Freeze all layers except the last one
        for i, layer in enumerate(self.network.linears):
            if i < len(self.network.linears) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Create optimizer for only trainable parameters
        trainable_params = [p for p in self.network.parameters() if p.requires_grad]
        self.fine_tune_optimizer = torch.optim.Adam(trainable_params, lr=self.fine_tune_lr)
        
        # Fine-tuning loop
        for epoch in range(epochs):
            self.fine_tune_optimizer.zero_grad()
            
            # Compute loss on support data
            loss = self.compute_total_loss(support_data, target_task)
            loss.backward()
            self.fine_tune_optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Feature extraction epoch {epoch}: loss={loss.item():.6f}")
        
        print("Feature extraction fine-tuning completed")
    
    def _gradual_unfreezing_fine_tuning(self, target_task: Task, support_data: TaskData, epochs: int):
        """Gradual unfreezing strategy - progressively unfreeze layers."""
        print(f"Starting gradual unfreezing fine-tuning for {epochs} epochs...")
        
        # Initially freeze all layers except the last one
        for i, layer in enumerate(self.network.linears):
            if i < len(self.network.linears) - 1:
                for param in layer.parameters():
                    param.requires_grad = False
                self.frozen_layers.add(i)
            else:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Fine-tuning loop with gradual unfreezing
        for epoch in range(epochs):
            # Check if we should unfreeze a layer
            if epoch in self.unfreeze_schedule:
                self._unfreeze_next_layer()
            
            # Update optimizer with current trainable parameters
            trainable_params = [p for p in self.network.parameters() if p.requires_grad]
            self.fine_tune_optimizer = torch.optim.Adam(trainable_params, lr=self.fine_tune_lr)
            
            self.fine_tune_optimizer.zero_grad()
            
            # Compute loss on support data
            loss = self.compute_total_loss(support_data, target_task)
            loss.backward()
            self.fine_tune_optimizer.step()
            
            if epoch % 20 == 0:
                print(f"Gradual unfreezing epoch {epoch}: loss={loss.item():.6f}, "
                      f"frozen_layers={len(self.frozen_layers)}")
        
        print("Gradual unfreezing fine-tuning completed")
    
    def _unfreeze_next_layer(self):
        """Unfreeze the next layer in the gradual unfreezing strategy."""
        if not self.frozen_layers:
            return
        
        # Unfreeze the deepest frozen layer (closest to output)
        layer_to_unfreeze = max(self.frozen_layers)
        
        for param in self.network.linears[layer_to_unfreeze].parameters():
            param.requires_grad = True
        
        self.frozen_layers.remove(layer_to_unfreeze)
        print(f"Unfroze layer {layer_to_unfreeze}")
    
    def adaptive_fine_tuning_strategy_selection(self, target_task: Task, 
                                               source_tasks: List[Task]) -> str:
        """Select fine-tuning strategy based on task similarity.
        
        This method implements adaptive strategy selection by analyzing
        the similarity between target and source tasks.
        """
        if not hasattr(self, 'source_distribution_cache') or not self.source_distribution_cache:
            print("No source distribution cache available, defaulting to full fine-tuning")
            return 'full'
        
        # Compute task similarity
        similarity_score = self._compute_task_similarity(target_task, source_tasks)
        
        # Select strategy based on similarity
        if similarity_score > 0.8:
            # High similarity - feature extraction should work well
            strategy = 'feature_extraction'
            print(f"High task similarity ({similarity_score:.3f}) - using feature extraction")
        elif similarity_score > 0.5:
            # Medium similarity - gradual unfreezing for balanced adaptation
            strategy = 'gradual_unfreezing'
            print(f"Medium task similarity ({similarity_score:.3f}) - using gradual unfreezing")
        else:
            # Low similarity - full fine-tuning needed
            strategy = 'full'
            print(f"Low task similarity ({similarity_score:.3f}) - using full fine-tuning")
        
        return strategy
    
    def _compute_task_similarity(self, target_task: Task, source_tasks: List[Task]) -> float:
        """Compute similarity between target task and source task distribution."""
        # Extract target task features
        self.network.eval()
        with torch.no_grad():
            target_features = self._extract_features(target_task.support_data.inputs)
        
        # Compute similarity with cached source features
        similarities = []
        for cached_task in self.source_distribution_cache.values():
            source_features = cached_task['features'].to(self.device)
            
            # Compute cosine similarity between feature distributions
            target_mean = target_features.mean(dim=0)
            source_mean = source_features.mean(dim=0)
            
            similarity = torch.nn.functional.cosine_similarity(
                target_mean.unsqueeze(0), 
                source_mean.unsqueeze(0)
            ).item()
            
            similarities.append(similarity)
        
        self.network.train()
        
        # Return maximum similarity (most similar source task)
        return max(similarities) if similarities else 0.0
    
    def _compute_domain_adaptation_loss_cached(self, target_data: TaskData, target_task: Task) -> torch.Tensor:
        """Compute domain adaptation loss using cached source distribution."""
        if not self.source_distribution_cache:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Find most similar source task
        max_similarity = -1
        best_source_features = None
        
        target_features = self._extract_features(target_data.inputs)
        
        for cached_task in self.source_distribution_cache.values():
            source_features = cached_task['features'].to(self.device)
            
            # Compute similarity
            target_mean = target_features.mean(dim=0)
            source_mean = source_features.mean(dim=0)
            
            similarity = torch.nn.functional.cosine_similarity(
                target_mean.unsqueeze(0), 
                source_mean.unsqueeze(0)
            ).item()
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_source_features = source_features
        
        if best_source_features is not None:
            return self._compute_mmd_loss(best_source_features, target_features)
        else:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
    
    def evaluate(self, test_data: TaskData, task: Task) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        self.network.eval()
        with torch.no_grad():
            # Compute predictions
            predictions = self.forward(test_data.inputs)
            
            # Compute various losses
            data_loss = torch.nn.functional.mse_loss(predictions, test_data.outputs)
            physics_loss = self.compute_physics_loss(test_data, task)
            total_loss = self.compute_total_loss(test_data, task)
            
            # Compute relative L2 error
            l2_error = torch.norm(predictions - test_data.outputs) / torch.norm(test_data.outputs)
            
        self.network.train()
        
        return {
            'data_loss': data_loss.item(),
            'physics_loss': physics_loss.item(),
            'total_loss': total_loss.item(),
            'l2_relative_error': l2_error.item(),
            'mse': data_loss.item()
        }
    
    def few_shot_evaluation(self, test_tasks: List[Task], 
                           k_shots_list: List[int] = [1, 5, 10, 25],
                           strategies: List[str] = ['full', 'feature_extraction', 'gradual_unfreezing']) -> Dict[str, Any]:
        """Comprehensive few-shot evaluation with different fine-tuning strategies."""
        results = {}
        
        for strategy in strategies:
            for k_shots in k_shots_list:
                key = f"{strategy}_K{k_shots}"
                task_results = []
                fine_tuning_times = []
                
                for task in test_tasks:
                    # Measure fine-tuning time
                    start_time = time.time()
                    
                    # Fine-tune model
                    fine_tuned_model = self.fine_tune(task, k_shots, strategy)
                    
                    fine_tuning_time = time.time() - start_time
                    fine_tuning_times.append(fine_tuning_time)
                    
                    # Evaluate on query set
                    metrics = fine_tuned_model.evaluate(task.query_data, task)
                    task_results.append(metrics)
                
                # Aggregate results
                results[key] = {
                    'mean_l2_error': np.mean([r['l2_relative_error'] for r in task_results]),
                    'std_l2_error': np.std([r['l2_relative_error'] for r in task_results]),
                    'mean_data_loss': np.mean([r['data_loss'] for r in task_results]),
                    'std_data_loss': np.std([r['data_loss'] for r in task_results]),
                    'mean_physics_loss': np.mean([r['physics_loss'] for r in task_results]),
                    'std_physics_loss': np.std([r['physics_loss'] for r in task_results]),
                    'mean_fine_tuning_time': np.mean(fine_tuning_times),
                    'std_fine_tuning_time': np.std(fine_tuning_times),
                    'task_results': task_results,
                    'fine_tuning_times': fine_tuning_times
                }
        
        return results
    
    def save_model(self, filepath: str):
        """Save model state following PINNacle patterns."""
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'pretrain_optimizer_state_dict': self.pretrain_optimizer.state_dict(),
            'fine_tune_optimizer_state_dict': self.fine_tune_optimizer.state_dict(),
            'config': self.config,
            'pretrain_iteration': self.pretrain_iteration,
            'fine_tune_iteration': self.fine_tune_iteration,
            'training_history': self.training_history,
            'pretrained': self.pretrained,
            'source_distribution_cache': self.source_distribution_cache,
            'frozen_layers': self.frozen_layers
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state following PINNacle patterns."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.pretrain_optimizer.load_state_dict(checkpoint['pretrain_optimizer_state_dict'])
        self.fine_tune_optimizer.load_state_dict(checkpoint['fine_tune_optimizer_state_dict'])
        self.pretrain_iteration = checkpoint.get('pretrain_iteration', 0)
        self.fine_tune_iteration = checkpoint.get('fine_tune_iteration', 0)
        self.training_history = checkpoint.get('training_history', [])
        self.pretrained = checkpoint.get('pretrained', False)
        self.source_distribution_cache = checkpoint.get('source_distribution_cache', {})
        self.frozen_layers = checkpoint.get('frozen_layers', set())
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model summary following PINNacle patterns."""
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        return {
            'model_type': 'TransferLearningPINN',
            'network_architecture': self.config.layers,
            'activation': self.config.activation,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'pretrained': self.pretrained,
            'fine_tune_strategy': self.fine_tune_strategy,
            'fine_tune_lr': self.fine_tune_lr,
            'domain_adaptation': self.domain_adaptation,
            'pretrain_iteration': self.pretrain_iteration,
            'fine_tune_iteration': self.fine_tune_iteration,
            'cached_source_tasks': len(self.source_distribution_cache),
            'frozen_layers': len(self.frozen_layers),
            'device': str(self.device)
        }


class PhysicsAwareDomainAdapter:
    """Physics-aware domain adaptation techniques for transfer learning.
    
    This class implements specialized domain adaptation methods that consider
    the physics structure of PDE problems for better transfer performance.
    """
    
    def __init__(self, config: TransferLearningPINNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Domain adaptation parameters
        self.adaptation_weight = config.domain_adaptation_weight
        self.cache_size = config.source_distribution_cache_size
        
        # Physics-aware adaptation components
        self.physics_discriminator = None
        self.parameter_encoder = None
        
    def setup_physics_discriminator(self, input_dim: int, hidden_dim: int = 64):
        """Set up physics discriminator for adversarial domain adaptation."""
        self.physics_discriminator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def setup_parameter_encoder(self, param_dim: int, hidden_dim: int = 32):
        """Set up parameter encoder for physics parameter adaptation."""
        self.parameter_encoder = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, param_dim)
        ).to(self.device)
    
    def compute_physics_aware_domain_loss(self, source_features: torch.Tensor, 
                                        target_features: torch.Tensor,
                                        source_params: Dict[str, float],
                                        target_params: Dict[str, float]) -> torch.Tensor:
        """Compute physics-aware domain adaptation loss."""
        # Standard MMD loss
        mmd_loss = self._compute_mmd_loss(source_features, target_features)
        
        # Physics parameter alignment loss
        param_loss = self._compute_parameter_alignment_loss(source_params, target_params)
        
        # Physics structure preservation loss
        structure_loss = self._compute_physics_structure_loss(source_features, target_features)
        
        # Combine losses
        total_loss = mmd_loss + 0.1 * param_loss + 0.05 * structure_loss
        
        return total_loss
    
    def _compute_mmd_loss(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy loss."""
        def gaussian_kernel(x, y, sigma=1.0):
            dist = torch.cdist(x, y, p=2)
            return torch.exp(-dist**2 / (2 * sigma**2))
        
        k_ss = gaussian_kernel(source_features, source_features)
        k_tt = gaussian_kernel(target_features, target_features)
        k_st = gaussian_kernel(source_features, target_features)
        
        mmd = k_ss.mean() + k_tt.mean() - 2 * k_st.mean()
        return mmd
    
    def _compute_parameter_alignment_loss(self, source_params: Dict[str, float], 
                                        target_params: Dict[str, float]) -> torch.Tensor:
        """Compute loss to align physics parameters between domains."""
        param_loss = 0.0
        
        # Compare common physics parameters
        common_params = set(source_params.keys()) & set(target_params.keys())
        
        for param_name in common_params:
            source_val = torch.tensor(source_params[param_name], device=self.device)
            target_val = torch.tensor(target_params[param_name], device=self.device)
            
            # Normalize parameters to [0, 1] range for comparison
            if param_name in ['diffusivity', 'viscosity', 'conductivity']:
                # Log-scale normalization for physical parameters
                source_norm = torch.log(source_val + 1e-8)
                target_norm = torch.log(target_val + 1e-8)
            else:
                source_norm = source_val
                target_norm = target_val
            
            param_loss += torch.abs(source_norm - target_norm)
        
        return param_loss
    
    def _compute_physics_structure_loss(self, source_features: torch.Tensor, 
                                      target_features: torch.Tensor) -> torch.Tensor:
        """Compute loss to preserve physics structure across domains."""
        # Compute feature covariance matrices
        source_cov = torch.cov(source_features.T)
        target_cov = torch.cov(target_features.T)
        
        # Frobenius norm of covariance difference
        structure_loss = torch.norm(source_cov - target_cov, p='fro')
        
        return structure_loss
    
    def conditional_shift_penalization(self, source_data: TaskData, target_data: TaskData,
                                     source_task: Task, target_task: Task) -> torch.Tensor:
        """Implement conditional shift penalization for physics domains.
        
        This method penalizes distribution shifts that are not explained by
        changes in physics parameters, helping preserve physics structure.
        """
        # Extract features from both domains
        source_features = self._extract_features(source_data.inputs)
        target_features = self._extract_features(target_data.inputs)
        
        # Compute expected shift based on physics parameter changes
        expected_shift = self._compute_expected_physics_shift(
            source_task.parameters, target_task.parameters
        )
        
        # Compute actual feature shift
        actual_shift = torch.mean(target_features, dim=0) - torch.mean(source_features, dim=0)
        
        # Penalize unexpected shifts
        shift_penalty = torch.norm(actual_shift - expected_shift, p=2)
        
        return shift_penalty
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features using the main network (placeholder - would use actual network)."""
        # This would be implemented by the main TransferLearningPINN class
        # For now, return input as features
        return x
    
    def _compute_expected_physics_shift(self, source_params: Dict[str, float], 
                                      target_params: Dict[str, float]) -> torch.Tensor:
        """Compute expected feature shift based on physics parameter changes."""
        # Simple linear model for expected shift
        # In practice, this could be learned or based on physics knowledge
        
        shift_magnitude = 0.0
        for param_name in source_params:
            if param_name in target_params:
                relative_change = abs(target_params[param_name] - source_params[param_name]) / (source_params[param_name] + 1e-8)
                shift_magnitude += relative_change
        
        # Return uniform shift vector (simplified)
        feature_dim = 64  # Assuming 64-dimensional features
        expected_shift = torch.ones(feature_dim, device=self.device) * shift_magnitude * 0.1
        
        return expected_shift


# Extend TransferLearningPINN with domain adaptation capabilities
class TransferLearningPINNWithDomainAdaptation(TransferLearningPINN):
    """Extended TransferLearningPINN with advanced domain adaptation capabilities."""
    
    def __init__(self, config: TransferLearningPINNConfig):
        super().__init__(config)
        
        # Initialize domain adapter
        self.domain_adapter = PhysicsAwareDomainAdapter(config)
        
        # Set up domain adaptation components
        if config.domain_adaptation:
            # Will be initialized after network is created
            self.domain_adapter.setup_physics_discriminator(64)  # Assuming 64-dim features
            self.domain_adapter.setup_parameter_encoder(len(config.layers))
    
    def compute_domain_adaptation_loss(self, source_data: TaskData, target_data: TaskData,
                                     source_task: Task, target_task: Task) -> torch.Tensor:
        """Enhanced domain adaptation loss with physics awareness."""
        if not self.domain_adaptation:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Extract features from both domains
        source_features = self._extract_features(source_data.inputs)
        target_features = self._extract_features(target_data.inputs)
        
        # Compute physics-aware domain loss
        domain_loss = self.domain_adapter.compute_physics_aware_domain_loss(
            source_features, target_features,
            source_task.parameters, target_task.parameters
        )
        
        # Add conditional shift penalization
        shift_penalty = self.domain_adapter.conditional_shift_penalization(
            source_data, target_data, source_task, target_task
        )
        
        return domain_loss + 0.1 * shift_penalty
    
    def _extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from the network for domain adaptation."""
        # Forward through all layers except the last one
        current_x = x
        
        # Apply input transform if exists
        if hasattr(self.network, '_input_transform') and self.network._input_transform is not None:
            current_x = self.network._input_transform(current_x)
        
        # Forward through all but last layer
        for i, linear_layer in enumerate(self.network.linears[:-1]):
            current_x = linear_layer(current_x)
            
            # Apply activation
            if self.config.activation == 'sin':
                current_x = torch.sin(current_x)
            elif self.config.activation == 'tanh':
                current_x = torch.tanh(current_x)
            else:
                activation_fn = getattr(torch.nn.functional, self.config.activation)
                current_x = activation_fn(current_x)
        
        return current_x
    
    def enhanced_source_distribution_caching(self, source_tasks: List[Task]):
        """Enhanced source distribution caching with physics parameter clustering."""
        print("Enhanced caching of source distribution with physics clustering...")
        
        # Group tasks by physics parameters for better caching
        param_clusters = self._cluster_tasks_by_physics_params(source_tasks)
        
        self.network.eval()
        cache_index = 0
        
        with torch.no_grad():
            for cluster_id, cluster_tasks in param_clusters.items():
                # Sample representative tasks from each cluster
                n_samples = min(self.cache_size // len(param_clusters), len(cluster_tasks))
                sampled_tasks = random.sample(cluster_tasks, n_samples)
                
                for task in sampled_tasks:
                    # Extract features and store comprehensive task information
                    features = self._extract_features(task.support_data.inputs)
                    
                    self.source_distribution_cache[cache_index] = {
                        'features': features.cpu(),
                        'task_params': task.parameters,
                        'problem_type': task.problem_type,
                        'cluster_id': cluster_id,
                        'physics_signature': self._compute_physics_signature(task)
                    }
                    cache_index += 1
        
        self.network.train()
        print(f"Enhanced caching completed: {len(self.source_distribution_cache)} tasks across {len(param_clusters)} clusters")
    
    def _cluster_tasks_by_physics_params(self, tasks: List[Task]) -> Dict[int, List[Task]]:
        """Cluster tasks based on physics parameters for better caching."""
        from sklearn.cluster import KMeans
        
        # Extract physics parameters as feature vectors
        param_vectors = []
        for task in tasks:
            param_vector = []
            for key in sorted(task.parameters.keys()):
                param_vector.append(task.parameters[key])
            param_vectors.append(param_vector)
        
        param_vectors = np.array(param_vectors)
        
        # Cluster tasks (use 5 clusters by default)
        n_clusters = min(5, len(tasks))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(param_vectors)
        
        # Group tasks by cluster
        clusters = {}
        for i, task in enumerate(tasks):
            cluster_id = cluster_labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(task)
        
        return clusters
    
    def _compute_physics_signature(self, task: Task) -> torch.Tensor:
        """Compute a physics signature for the task based on its parameters."""
        # Create a signature vector from physics parameters
        signature_components = []
        
        # Add normalized parameter values
        for key in sorted(task.parameters.keys()):
            value = task.parameters[key]
            if key in ['diffusivity', 'viscosity', 'conductivity']:
                # Log-normalize physical parameters
                signature_components.append(np.log(value + 1e-8))
            else:
                signature_components.append(value)
        
        # Add problem type encoding
        problem_type_encoding = hash(task.problem_type) % 1000 / 1000.0
        signature_components.append(problem_type_encoding)
        
        return torch.tensor(signature_components, dtype=torch.float32)
    
    def physics_guided_fine_tuning(self, target_task: Task, k_shots: int) -> 'TransferLearningPINNWithDomainAdaptation':
        """Physics-guided fine-tuning that leverages domain adaptation."""
        # Find most similar source tasks based on physics parameters
        similar_source_tasks = self._find_similar_source_tasks(target_task, top_k=5)
        
        # Use adaptive strategy selection with physics awareness
        strategy = self._physics_aware_strategy_selection(target_task, similar_source_tasks)
        
        # Perform fine-tuning with domain adaptation
        fine_tuned_model = self.fine_tune(target_task, k_shots, strategy)
        
        return fine_tuned_model
    
    def _find_similar_source_tasks(self, target_task: Task, top_k: int = 5) -> List[Dict]:
        """Find most similar source tasks based on physics parameters."""
        if not self.source_distribution_cache:
            return []
        
        target_signature = self._compute_physics_signature(target_task)
        similarities = []
        
        for cache_id, cached_task in self.source_distribution_cache.items():
            cached_signature = cached_task['physics_signature']
            
            # Compute cosine similarity between physics signatures
            similarity = torch.nn.functional.cosine_similarity(
                target_signature.unsqueeze(0),
                cached_signature.unsqueeze(0)
            ).item()
            
            similarities.append((cache_id, similarity, cached_task))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [cached_task for _, _, cached_task in similarities[:top_k]]
    
    def _physics_aware_strategy_selection(self, target_task: Task, similar_tasks: List[Dict]) -> str:
        """Select fine-tuning strategy based on physics similarity."""
        if not similar_tasks:
            return 'full'
        
        # Compute average physics similarity
        target_signature = self._compute_physics_signature(target_task)
        similarities = []
        
        for similar_task in similar_tasks:
            similarity = torch.nn.functional.cosine_similarity(
                target_signature.unsqueeze(0),
                similar_task['physics_signature'].unsqueeze(0)
            ).item()
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities)
        
        # Strategy selection based on physics similarity
        if avg_similarity > 0.9:
            return 'feature_extraction'  # Very similar physics
        elif avg_similarity > 0.7:
            return 'gradual_unfreezing'  # Moderately similar physics
        else:
            return 'full'  # Different physics, need full adaptation