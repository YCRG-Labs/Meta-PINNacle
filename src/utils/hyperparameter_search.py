"""Hyperparameter search implementation for baseline methods.

This module provides comprehensive hyperparameter search functionality for baseline methods
including Standard PINN, FNO, and DeepONet to ensure fair comparison with meta-learning methods.
"""

import os
import json
import time
import itertools
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from pathlib import Path

from ..model.fno_baseline import FNOBaseline, create_fno_baseline
from ..model.deeponet_baseline import DeepONetBaseline, create_deeponet_baseline
from ..meta_learning.task import Task
from ..utils.metrics import compute_l2_relative_error


@dataclass
class HyperparameterConfig:
    """Configuration for hyperparameter search."""
    learning_rates: List[float]
    batch_sizes: List[int]
    architectures: List[Dict[str, Any]]
    loss_weights: List[Dict[str, float]]
    epochs: int = 1000
    validation_split: float = 0.2
    num_random_trials: int = 20
    early_stopping_patience: int = 50
    

@dataclass
class SearchResult:
    """Result from a single hyperparameter configuration trial."""
    config: Dict[str, Any]
    validation_loss: float
    training_loss: float
    training_time: float
    convergence_epoch: int
    l2_error: float
    

class BaselineHyperparameterSearch:
    """Comprehensive hyperparameter search for baseline methods.
    
    This class implements grid search and random search over hyperparameters
    for baseline methods to ensure fair comparison with meta-learning approaches.
    """
    
    def __init__(self, results_dir: str = "hyperparameter_search_results"):
        """Initialize hyperparameter search.
        
        Args:
            results_dir: Directory to save search results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Define search spaces for different baseline methods
        self.search_spaces = self._define_search_spaces()
        
        # Results storage
        self.search_results = {}
        
    def _define_search_spaces(self) -> Dict[str, HyperparameterConfig]:
        """Define hyperparameter search spaces for each baseline method."""
        
        # Common learning rates and batch sizes
        learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
        batch_sizes = [256, 512, 1024, 2048]
        
        # FNO search space
        fno_architectures = [
            {'width': 32, 'layers': 3, 'modes1': 8, 'modes2': 8},
            {'width': 64, 'layers': 5, 'modes1': 12, 'modes2': 12},
            {'width': 128, 'layers': 8, 'modes1': 16, 'modes2': 16}
        ]
        
        # DeepONet search space
        deeponet_architectures = [
            {
                'branch_layers': [100, 64, 64, 64],
                'trunk_layers': [2, 64, 64, 64],
                'num_sensors': 50
            },
            {
                'branch_layers': [100, 128, 128, 128],
                'trunk_layers': [2, 128, 128, 128],
                'num_sensors': 100
            },
            {
                'branch_layers': [100, 256, 256, 256],
                'trunk_layers': [2, 256, 256, 256],
                'num_sensors': 150
            }
        ]
        
        # Standard PINN architectures (for comparison)
        pinn_architectures = [
            {'layers': [2, 32, 32, 32, 1], 'activation': 'tanh'},
            {'layers': [2, 64, 64, 64, 64, 64, 1], 'activation': 'tanh'},
            {'layers': [2, 128, 128, 128, 128, 128, 128, 128, 128, 1], 'activation': 'tanh'}
        ]
        
        # Loss weight combinations for physics-informed methods
        loss_weights = [
            {'physics_weight': 1.0, 'data_weight': 1.0},
            {'physics_weight': 0.1, 'data_weight': 1.0},
            {'physics_weight': 10.0, 'data_weight': 1.0},
            {'physics_weight': 1.0, 'data_weight': 0.1},
            {'physics_weight': 1.0, 'data_weight': 10.0}
        ]
        
        return {
            'fno': HyperparameterConfig(
                learning_rates=learning_rates,
                batch_sizes=batch_sizes,
                architectures=fno_architectures,
                loss_weights=[{}],  # FNO doesn't use physics loss weights
                epochs=1000
            ),
            'deeponet': HyperparameterConfig(
                learning_rates=learning_rates,
                batch_sizes=batch_sizes,
                architectures=deeponet_architectures,
                loss_weights=[{}],  # DeepONet doesn't use physics loss weights
                epochs=1000
            ),
            'standard_pinn': HyperparameterConfig(
                learning_rates=learning_rates,
                batch_sizes=batch_sizes,
                architectures=pinn_architectures,
                loss_weights=loss_weights,
                epochs=1000
            )
        }
    
    def search_baseline_hyperparameters(self, 
                                      method_name: str,
                                      pde_family: str,
                                      train_tasks: List[Task],
                                      validation_tasks: List[Task]) -> Dict[str, Any]:
        """Perform hyperparameter search for a baseline method.
        
        Args:
            method_name: Name of the baseline method ('fno', 'deeponet', 'standard_pinn')
            pde_family: Name of the PDE family
            train_tasks: Training tasks
            validation_tasks: Validation tasks for hyperparameter selection
            
        Returns:
            Dictionary containing best configuration and search results
        """
        print(f"Starting hyperparameter search for {method_name} on {pde_family}")
        
        if method_name not in self.search_spaces:
            raise ValueError(f"Unknown method: {method_name}")
        
        search_config = self.search_spaces[method_name]
        
        # Generate all hyperparameter combinations
        all_configs = self._generate_hyperparameter_combinations(search_config)
        
        # Randomly sample configurations if too many
        if len(all_configs) > search_config.num_random_trials:
            selected_indices = np.random.choice(
                len(all_configs), 
                size=search_config.num_random_trials, 
                replace=False
            )
            configs_to_test = [all_configs[i] for i in selected_indices]
        else:
            configs_to_test = all_configs
        
        print(f"Testing {len(configs_to_test)} hyperparameter configurations")
        
        # Run hyperparameter search
        search_results = []
        best_result = None
        best_validation_loss = float('inf')
        
        for i, config in enumerate(configs_to_test):
            print(f"Testing configuration {i+1}/{len(configs_to_test)}")
            
            try:
                result = self._evaluate_hyperparameter_config(
                    method_name, pde_family, config, train_tasks, validation_tasks
                )
                search_results.append(result)
                
                # Track best configuration
                if result.validation_loss < best_validation_loss:
                    best_validation_loss = result.validation_loss
                    best_result = result
                    
            except Exception as e:
                print(f"Configuration {i+1} failed: {e}")
                continue
        
        # Save search results
        self._save_search_results(method_name, pde_family, search_results)
        
        # Generate search analysis
        analysis = self._analyze_search_results(search_results)
        
        return {
            'best_config': best_result.config if best_result else None,
            'best_validation_loss': best_validation_loss,
            'best_l2_error': best_result.l2_error if best_result else None,
            'search_results': search_results,
            'analysis': analysis,
            'num_configs_tested': len(search_results)
        }
    
    def _generate_hyperparameter_combinations(self, 
                                            search_config: HyperparameterConfig) -> List[Dict[str, Any]]:
        """Generate all combinations of hyperparameters."""
        combinations = []
        
        for lr in search_config.learning_rates:
            for batch_size in search_config.batch_sizes:
                for arch in search_config.architectures:
                    for loss_weights in search_config.loss_weights:
                        config = {
                            'learning_rate': lr,
                            'batch_size': batch_size,
                            'epochs': search_config.epochs,
                            **arch,
                            **loss_weights
                        }
                        combinations.append(config)
        
        return combinations
    
    def _evaluate_hyperparameter_config(self,
                                       method_name: str,
                                       pde_family: str,
                                       config: Dict[str, Any],
                                       train_tasks: List[Task],
                                       validation_tasks: List[Task]) -> SearchResult:
        """Evaluate a single hyperparameter configuration.
        
        Args:
            method_name: Name of the baseline method
            pde_family: Name of the PDE family
            config: Hyperparameter configuration
            train_tasks: Training tasks
            validation_tasks: Validation tasks
            
        Returns:
            SearchResult containing evaluation metrics
        """
        start_time = time.time()
        
        # Create model with configuration
        if method_name == 'fno':
            model = self._create_fno_with_config(config)
        elif method_name == 'deeponet':
            model = self._create_deeponet_with_config(config)
        elif method_name == 'standard_pinn':
            model = self._create_standard_pinn_with_config(config)
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        # Train model
        training_result = model.train_on_pde_family(pde_family, train_tasks)
        training_time = time.time() - start_time
        
        # Evaluate on validation tasks
        validation_loss, l2_error = self._evaluate_on_validation_tasks(model, validation_tasks)
        
        return SearchResult(
            config=config,
            validation_loss=validation_loss,
            training_loss=training_result.get('final_loss', float('inf')),
            training_time=training_time,
            convergence_epoch=len(training_result.get('losses', [])),
            l2_error=l2_error
        )
    
    def _create_fno_with_config(self, config: Dict[str, Any]) -> FNOBaseline:
        """Create FNO model with specific configuration."""
        fno_config = {
            'modes1': config.get('modes1', 12),
            'modes2': config.get('modes2', 12),
            'width': config.get('width', 32),
            'layers': config.get('layers', 4),
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        return FNOBaseline(fno_config)
    
    def _create_deeponet_with_config(self, config: Dict[str, Any]) -> DeepONetBaseline:
        """Create DeepONet model with specific configuration."""
        deeponet_config = {
            'branch_layers': config.get('branch_layers', [100, 128, 128, 128]),
            'trunk_layers': config.get('trunk_layers', [2, 128, 128, 128]),
            'num_sensors': config.get('num_sensors', 100),
            'activation': 'tanh',
            'learning_rate': config['learning_rate'],
            'batch_size': config['batch_size'],
            'epochs': config['epochs'],
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        return DeepONetBaseline(deeponet_config)
    
    def _create_standard_pinn_with_config(self, config: Dict[str, Any]):
        """Create Standard PINN model with specific configuration.
        
        Note: This is a placeholder - actual Standard PINN implementation
        would need to be imported from the appropriate module.
        """
        # This would need to be implemented based on the actual Standard PINN class
        # For now, return None as placeholder
        return None
    
    def _evaluate_on_validation_tasks(self, model, validation_tasks: List[Task]) -> Tuple[float, float]:
        """Evaluate model on validation tasks.
        
        Args:
            model: Trained model
            validation_tasks: Validation tasks
            
        Returns:
            Tuple of (validation_loss, average_l2_error)
        """
        if not validation_tasks:
            return float('inf'), float('inf')
        
        total_loss = 0.0
        total_l2_error = 0.0
        num_tasks = len(validation_tasks)
        
        for task in validation_tasks:
            task_data = task.get_task_data()
            
            # Get model prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(task_data.x_physics, task_data.params)
            else:
                # Fallback for models without predict method
                continue
            
            # Get reference solution
            if hasattr(task_data, 'u_ref') and task_data.u_ref is not None:
                reference = task_data.u_ref
            else:
                continue
            
            # Compute metrics
            mse_loss = np.mean((prediction - reference) ** 2)
            l2_error = compute_l2_relative_error(prediction, reference)
            
            total_loss += mse_loss
            total_l2_error += l2_error
        
        avg_loss = total_loss / num_tasks if num_tasks > 0 else float('inf')
        avg_l2_error = total_l2_error / num_tasks if num_tasks > 0 else float('inf')
        
        return avg_loss, avg_l2_error
    
    def _save_search_results(self, method_name: str, pde_family: str, results: List[SearchResult]):
        """Save hyperparameter search results to file."""
        results_file = self.results_dir / f"{method_name}_{pde_family}_search_results.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in results:
            serializable_results.append(asdict(result))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Search results saved to {results_file}")
    
    def _analyze_search_results(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze hyperparameter search results.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary containing analysis results
        """
        if not results:
            return {}
        
        # Extract metrics
        validation_losses = [r.validation_loss for r in results]
        l2_errors = [r.l2_error for r in results]
        training_times = [r.training_time for r in results]
        
        # Analyze hyperparameter importance
        hyperparameter_analysis = self._analyze_hyperparameter_importance(results)
        
        return {
            'num_configurations': len(results),
            'best_validation_loss': min(validation_losses),
            'worst_validation_loss': max(validation_losses),
            'mean_validation_loss': np.mean(validation_losses),
            'std_validation_loss': np.std(validation_losses),
            'best_l2_error': min(l2_errors),
            'mean_l2_error': np.mean(l2_errors),
            'mean_training_time': np.mean(training_times),
            'hyperparameter_importance': hyperparameter_analysis
        }
    
    def _analyze_hyperparameter_importance(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Analyze the importance of different hyperparameters.
        
        Args:
            results: List of search results
            
        Returns:
            Dictionary containing hyperparameter importance analysis
        """
        if len(results) < 2:
            return {}
        
        # Group results by hyperparameter values
        hyperparameter_groups = {}
        
        for result in results:
            for param_name, param_value in result.config.items():
                if param_name not in hyperparameter_groups:
                    hyperparameter_groups[param_name] = {}
                
                param_key = str(param_value)
                if param_key not in hyperparameter_groups[param_name]:
                    hyperparameter_groups[param_name][param_key] = []
                
                hyperparameter_groups[param_name][param_key].append(result.validation_loss)
        
        # Compute statistics for each hyperparameter
        importance_analysis = {}
        for param_name, param_groups in hyperparameter_groups.items():
            if len(param_groups) > 1:  # Only analyze if there are multiple values
                group_means = []
                for param_value, losses in param_groups.items():
                    group_means.append(np.mean(losses))
                
                # Compute variance between groups as importance measure
                importance = np.var(group_means)
                best_value = min(param_groups.keys(), 
                               key=lambda x: np.mean(param_groups[x]))
                
                importance_analysis[param_name] = {
                    'importance_score': importance,
                    'best_value': best_value,
                    'group_statistics': {
                        param_value: {
                            'mean_loss': np.mean(losses),
                            'std_loss': np.std(losses),
                            'count': len(losses)
                        }
                        for param_value, losses in param_groups.items()
                    }
                }
        
        return importance_analysis
    
    def generate_search_plots(self, method_name: str, pde_family: str) -> Dict[str, str]:
        """Generate plots showing hyperparameter search results.
        
        Args:
            method_name: Name of the baseline method
            pde_family: Name of the PDE family
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        # Load search results
        results_file = self.results_dir / f"{method_name}_{pde_family}_search_results.json"
        if not results_file.exists():
            print(f"No search results found for {method_name} on {pde_family}")
            return {}
        
        with open(results_file, 'r') as f:
            results_data = json.load(f)
        
        results = [SearchResult(**data) for data in results_data]
        
        plot_files = {}
        
        # Plot 1: Validation loss vs learning rate
        plot_files['learning_rate_plot'] = self._plot_hyperparameter_effect(
            results, 'learning_rate', 'validation_loss', 
            f"{method_name}_{pde_family}_learning_rate_effect.png"
        )
        
        # Plot 2: Validation loss vs batch size
        plot_files['batch_size_plot'] = self._plot_hyperparameter_effect(
            results, 'batch_size', 'validation_loss',
            f"{method_name}_{pde_family}_batch_size_effect.png"
        )
        
        # Plot 3: Training time vs validation loss (Pareto frontier)
        plot_files['pareto_plot'] = self._plot_pareto_frontier(
            results, f"{method_name}_{pde_family}_pareto_frontier.png"
        )
        
        return plot_files
    
    def _plot_hyperparameter_effect(self, results: List[SearchResult], 
                                   param_name: str, metric_name: str, 
                                   filename: str) -> str:
        """Plot the effect of a hyperparameter on a metric."""
        # Group results by hyperparameter value
        param_groups = {}
        for result in results:
            if param_name in result.config:
                param_value = result.config[param_name]
                if param_value not in param_groups:
                    param_groups[param_value] = []
                
                metric_value = getattr(result, metric_name)
                param_groups[param_value].append(metric_value)
        
        # Compute statistics
        param_values = sorted(param_groups.keys())
        means = [np.mean(param_groups[val]) for val in param_values]
        stds = [np.std(param_groups[val]) for val in param_values]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_values, means, yerr=stds, marker='o', capsize=5)
        plt.xlabel(param_name.replace('_', ' ').title())
        plt.ylabel(metric_name.replace('_', ' ').title())
        plt.title(f'Effect of {param_name} on {metric_name}')
        plt.grid(True, alpha=0.3)
        
        if param_name == 'learning_rate':
            plt.xscale('log')
        
        plot_path = self.results_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _plot_pareto_frontier(self, results: List[SearchResult], filename: str) -> str:
        """Plot Pareto frontier of training time vs validation loss."""
        training_times = [r.training_time for r in results]
        validation_losses = [r.validation_loss for r in results]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(training_times, validation_losses, alpha=0.6)
        plt.xlabel('Training Time (seconds)')
        plt.ylabel('Validation Loss')
        plt.title('Training Time vs Validation Loss')
        plt.grid(True, alpha=0.3)
        
        plot_path = self.results_dir / filename
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)


class ValidationMethodology:
    """Standardized validation methodology for baseline hyperparameter search."""
    
    def __init__(self, num_parameter_values: int = 20, validation_split: float = 0.2):
        """Initialize validation methodology.
        
        Args:
            num_parameter_values: Number of random parameter values for validation
            validation_split: Fraction of tasks to use for validation
        """
        self.num_parameter_values = num_parameter_values
        self.validation_split = validation_split
        
    def create_validation_tasks(self, pde_family: str, parameter_ranges: Dict[str, Tuple[float, float]]) -> List[Task]:
        """Create validation tasks with random parameter values.
        
        Args:
            pde_family: Name of the PDE family
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            
        Returns:
            List of validation tasks
        """
        validation_tasks = []
        
        for i in range(self.num_parameter_values):
            # Sample random parameters within ranges
            params = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                params[param_name] = np.random.uniform(min_val, max_val)
            
            # Create task with these parameters
            # This would need to be implemented based on the actual Task creation interface
            # For now, this is a placeholder
            task = self._create_task_with_parameters(pde_family, params)
            if task is not None:
                validation_tasks.append(task)
        
        return validation_tasks
    
    def _create_task_with_parameters(self, pde_family: str, params: Dict[str, float]) -> Optional[Task]:
        """Create a task with specific parameters.
        
        This is a placeholder that would need to be implemented based on
        the actual task creation interface.
        """
        # Placeholder implementation
        return None
    
    def document_convergence_criteria(self) -> Dict[str, Any]:
        """Document the convergence criteria used for all methods.
        
        Returns:
            Dictionary describing convergence criteria
        """
        return {
            'early_stopping': {
                'patience': 50,
                'min_delta': 1e-6,
                'monitor': 'validation_loss'
            },
            'max_epochs': 1000,
            'convergence_threshold': 1e-6,
            'learning_rate_scheduling': {
                'scheduler': 'StepLR',
                'step_size': 100,
                'gamma': 0.5
            },
            'stopping_conditions': [
                'validation_loss_plateau',
                'max_epochs_reached',
                'gradient_norm_threshold'
            ]
        }


def run_comprehensive_baseline_search(pde_families: List[str], 
                                    baseline_methods: List[str],
                                    results_dir: str = "baseline_hyperparameter_search") -> Dict[str, Any]:
    """Run comprehensive hyperparameter search for all baseline methods and PDE families.
    
    Args:
        pde_families: List of PDE family names
        baseline_methods: List of baseline method names
        results_dir: Directory to save results
        
    Returns:
        Dictionary containing all search results
    """
    searcher = BaselineHyperparameterSearch(results_dir)
    validator = ValidationMethodology()
    
    all_results = {}
    
    for pde_family in pde_families:
        all_results[pde_family] = {}
        
        for method_name in baseline_methods:
            print(f"\n{'='*60}")
            print(f"Searching hyperparameters for {method_name} on {pde_family}")
            print(f"{'='*60}")
            
            try:
                # This would need actual task loading implementation
                train_tasks = []  # Load training tasks for pde_family
                validation_tasks = []  # Create validation tasks
                
                # Run hyperparameter search
                search_result = searcher.search_baseline_hyperparameters(
                    method_name, pde_family, train_tasks, validation_tasks
                )
                
                all_results[pde_family][method_name] = search_result
                
                # Generate plots
                plot_files = searcher.generate_search_plots(method_name, pde_family)
                search_result['plot_files'] = plot_files
                
            except Exception as e:
                print(f"Error in hyperparameter search for {method_name} on {pde_family}: {e}")
                all_results[pde_family][method_name] = {'error': str(e)}
    
    # Save comprehensive results
    results_file = Path(results_dir) / "comprehensive_search_results.json"
    with open(results_file, 'w') as f:
        # Convert to serializable format
        serializable_results = {}
        for pde_family, methods in all_results.items():
            serializable_results[pde_family] = {}
            for method_name, result in methods.items():
                if 'search_results' in result:
                    # Convert SearchResult objects to dicts
                    result['search_results'] = [asdict(r) for r in result['search_results']]
                serializable_results[pde_family][method_name] = result
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\nComprehensive search results saved to {results_file}")
    
    return all_results


if __name__ == "__main__":
    # Example usage
    pde_families = ['heat', 'burgers', 'poisson', 'navier_stokes']
    baseline_methods = ['fno', 'deeponet']
    
    results = run_comprehensive_baseline_search(pde_families, baseline_methods)
    print("Hyperparameter search completed!")