"""Comprehensive ablation studies for meta-learning PINNs.

This module implements systematic ablation studies to understand the contribution
of different architectural components and hyperparameters to meta-learning performance.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field

from .config import MetaPINNConfig, PhysicsInformedMetaLearnerConfig
from .meta_pinn import MetaPINN
from .physics_informed_meta_learner import PhysicsInformedMetaLearner
from .task import Task, TaskData, TaskBatch
from ..utils.metrics import compute_l2_relative_error


@dataclass
class AblationConfig:
    """Configuration for ablation studies."""
    
    # Architecture ablations
    test_adaptive_weights: bool = True
    test_physics_regularization: bool = True
    test_multiscale_loss: bool = True
    test_network_architectures: bool = True
    
    # Hyperparameter sensitivity
    test_inner_steps: bool = True
    test_meta_batch_sizes: bool = True
    test_learning_rates: bool = True
    
    # Evaluation parameters
    n_test_tasks: int = 20
    n_evaluation_runs: int = 3
    adaptation_steps_list: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    k_shots_list: List[int] = field(default_factory=lambda: [1, 5, 10, 25])
    
    # Architecture configurations to test
    network_configs: List[Dict[str, Any]] = field(default_factory=lambda: [
        {'layers': [2, 32, 32, 32, 1], 'name': 'small_3x32'},
        {'layers': [2, 64, 64, 64, 1], 'name': 'medium_3x64'},
        {'layers': [2, 128, 128, 128, 1], 'name': 'medium_3x128'},
        {'layers': [2, 256, 256, 256, 1], 'name': 'medium_3x256'},
        {'layers': [2, 64, 64, 64, 64, 64, 64, 64, 64, 1], 'name': 'deep_8x64'},
        {'layers': [2, 256, 256, 256, 256, 256, 256, 256, 256, 1], 'name': 'deep_8x256'}
    ])
    
    # Hyperparameter ranges
    inner_steps_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    meta_batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    learning_rates: List[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3, 5e-3])


class AblationStudyManager:
    """Manages comprehensive ablation studies for meta-learning PINNs.
    
    This class systematically tests the impact of different architectural components
    and hyperparameters on meta-learning performance.
    """
    
    def __init__(self, base_config: PhysicsInformedMetaLearnerConfig, 
                 ablation_config: AblationConfig = None):
        """Initialize ablation study manager.
        
        Args:
            base_config: Base configuration for the meta-learning model
            ablation_config: Configuration for ablation studies
        """
        self.base_config = base_config
        self.ablation_config = ablation_config or AblationConfig()
        
        # Results storage
        self.architecture_results = {}
        self.hyperparameter_results = {}
        self.component_impact_results = {}
        
        # Baseline performance (full model)
        self.baseline_performance = None
        
        # Device configuration
        self.device = torch.device(base_config.device)
        
    def run_architecture_ablations(self, train_tasks: List[Task], 
                                 test_tasks: List[Task]) -> Dict[str, Any]:
        """Run ablation studies on architecture components.
        
        Args:
            train_tasks: Training tasks for meta-learning
            test_tasks: Test tasks for evaluation
            
        Returns:
            Dictionary containing ablation results
        """
        print("Running architecture component ablations...")
        
        # Test full model as baseline
        print("Evaluating full model (baseline)...")
        baseline_config = copy.deepcopy(self.base_config)
        self.baseline_performance = self._evaluate_configuration(
            baseline_config, train_tasks, test_tasks, "full_model"
        )
        
        ablation_results = {'full_model': self.baseline_performance}
        
        # Test removing adaptive weights
        if self.ablation_config.test_adaptive_weights:
            print("Testing without adaptive constraint weighting...")
            no_adaptive_config = copy.deepcopy(self.base_config)
            no_adaptive_config.adaptive_constraint_weighting = False
            
            results = self._evaluate_configuration(
                no_adaptive_config, train_tasks, test_tasks, "no_adaptive_weights"
            )
            ablation_results['no_adaptive_weights'] = results
            
            # Compute impact percentage
            impact = self._compute_component_impact(
                self.baseline_performance, results, "adaptive_weights"
            )
            self.component_impact_results['adaptive_weights'] = impact
        
        # Test removing physics regularization
        if self.ablation_config.test_physics_regularization:
            print("Testing without physics regularization...")
            no_physics_reg_config = copy.deepcopy(self.base_config)
            no_physics_reg_config.physics_regularization_weight = 0.0
            
            results = self._evaluate_configuration(
                no_physics_reg_config, train_tasks, test_tasks, "no_physics_regularization"
            )
            ablation_results['no_physics_regularization'] = results
            
            # Compute impact percentage
            impact = self._compute_component_impact(
                self.baseline_performance, results, "physics_regularization"
            )
            self.component_impact_results['physics_regularization'] = impact
        
        # Test removing multi-scale loss
        if self.ablation_config.test_multiscale_loss:
            print("Testing without multi-scale loss...")
            no_multiscale_config = copy.deepcopy(self.base_config)
            no_multiscale_config.multi_scale_handling = False
            
            results = self._evaluate_configuration(
                no_multiscale_config, train_tasks, test_tasks, "no_multiscale_loss"
            )
            ablation_results['no_multiscale_loss'] = results
            
            # Compute impact percentage
            impact = self._compute_component_impact(
                self.baseline_performance, results, "multiscale_loss"
            )
            self.component_impact_results['multiscale_loss'] = impact
        
        # Test different network architectures
        if self.ablation_config.test_network_architectures:
            print("Testing different network architectures...")
            for arch_config in self.ablation_config.network_configs:
                print(f"Testing architecture: {arch_config['name']}")
                
                arch_test_config = copy.deepcopy(self.base_config)
                arch_test_config.layers = arch_config['layers']
                
                results = self._evaluate_configuration(
                    arch_test_config, train_tasks, test_tasks, arch_config['name']
                )
                ablation_results[arch_config['name']] = results
        
        self.architecture_results = ablation_results
        return ablation_results
    
    def run_hyperparameter_sensitivity(self, train_tasks: List[Task], 
                                     test_tasks: List[Task]) -> Dict[str, Any]:
        """Run hyperparameter sensitivity analysis.
        
        Args:
            train_tasks: Training tasks for meta-learning
            test_tasks: Test tasks for evaluation
            
        Returns:
            Dictionary containing sensitivity analysis results
        """
        print("Running hyperparameter sensitivity analysis...")
        
        sensitivity_results = {}
        
        # Test inner steps K values
        if self.ablation_config.test_inner_steps:
            print("Testing inner steps sensitivity...")
            inner_steps_results = {}
            
            for k_value in self.ablation_config.inner_steps_values:
                print(f"Testing K = {k_value} inner steps...")
                
                config = copy.deepcopy(self.base_config)
                config.adaptation_steps = k_value
                
                results = self._evaluate_configuration(
                    config, train_tasks, test_tasks, f"inner_steps_{k_value}"
                )
                inner_steps_results[k_value] = results
            
            sensitivity_results['inner_steps'] = inner_steps_results
        
        # Test meta batch sizes
        if self.ablation_config.test_meta_batch_sizes:
            print("Testing meta batch size sensitivity...")
            batch_size_results = {}
            
            for batch_size in self.ablation_config.meta_batch_sizes:
                print(f"Testing batch size = {batch_size}...")
                
                config = copy.deepcopy(self.base_config)
                config.meta_batch_size = batch_size
                
                results = self._evaluate_configuration(
                    config, train_tasks, test_tasks, f"batch_size_{batch_size}"
                )
                batch_size_results[batch_size] = results
            
            sensitivity_results['meta_batch_size'] = batch_size_results
        
        # Test learning rates
        if self.ablation_config.test_learning_rates:
            print("Testing learning rate sensitivity...")
            lr_results = {}
            
            for lr in self.ablation_config.learning_rates:
                print(f"Testing learning rate = {lr}...")
                
                config = copy.deepcopy(self.base_config)
                config.meta_lr = lr
                config.adapt_lr = lr * 10  # Keep 10x ratio
                
                results = self._evaluate_configuration(
                    config, train_tasks, test_tasks, f"lr_{lr}"
                )
                lr_results[lr] = results
            
            sensitivity_results['learning_rate'] = lr_results
        
        self.hyperparameter_results = sensitivity_results
        return sensitivity_results
    
    def _evaluate_configuration(self, config: PhysicsInformedMetaLearnerConfig,
                              train_tasks: List[Task], test_tasks: List[Task],
                              config_name: str) -> Dict[str, Any]:
        """Evaluate a specific configuration.
        
        Args:
            config: Configuration to evaluate
            train_tasks: Training tasks
            test_tasks: Test tasks
            config_name: Name for this configuration
            
        Returns:
            Dictionary containing evaluation results
        """
        results = {
            'config_name': config_name,
            'l2_errors': [],
            'adaptation_times': [],
            'training_times': [],
            'memory_usage': []
        }
        
        # Run multiple evaluation runs for statistical significance
        for run in range(self.ablation_config.n_evaluation_runs):
            print(f"  Run {run + 1}/{self.ablation_config.n_evaluation_runs}")
            
            # Initialize model with current configuration
            model = PhysicsInformedMetaLearner(config)
            
            # Measure training time
            start_time = time.time()
            
            # Quick meta-training (reduced iterations for ablation studies)
            train_subset = train_tasks[:20]  # Use subset for faster evaluation
            model.meta_train(
                train_subset, 
                val_tasks=None,
                meta_iterations=100  # Reduced for ablation studies
            )
            
            training_time = time.time() - start_time
            results['training_times'].append(training_time)
            
            # Evaluate on test tasks
            test_subset = test_tasks[:self.ablation_config.n_test_tasks]
            run_l2_errors = []
            run_adaptation_times = []
            
            for task in test_subset:
                # Set PDE problem
                model.set_pde_problem(task.metadata.get('pde_problem'))
                
                # Measure adaptation time
                adapt_start = time.time()
                
                # Fast adaptation with K=5 shots, S=5 steps (standard for ablation)
                adapted_params = model.fast_adapt(
                    task.support_data, task, k_shots=5, adaptation_steps=5
                )
                
                adaptation_time = time.time() - adapt_start
                run_adaptation_times.append(adaptation_time)
                
                # Evaluate adapted model
                with torch.no_grad():
                    predictions = model.forward(task.query_data.inputs, adapted_params)
                    l2_error = compute_l2_relative_error(
                        predictions.cpu().numpy(), 
                        task.query_data.outputs.cpu().numpy()
                    )
                    run_l2_errors.append(l2_error)
            
            # Store run results
            results['l2_errors'].extend(run_l2_errors)
            results['adaptation_times'].extend(run_adaptation_times)
            
            # Measure memory usage (approximate)
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                results['memory_usage'].append(memory_mb)
                torch.cuda.reset_peak_memory_stats()
        
        # Compute aggregate statistics
        results['mean_l2_error'] = np.mean(results['l2_errors'])
        results['std_l2_error'] = np.std(results['l2_errors'])
        results['mean_adaptation_time'] = np.mean(results['adaptation_times'])
        results['std_adaptation_time'] = np.std(results['adaptation_times'])
        results['mean_training_time'] = np.mean(results['training_times'])
        results['std_training_time'] = np.std(results['training_times'])
        
        if results['memory_usage']:
            results['mean_memory_usage'] = np.mean(results['memory_usage'])
            results['std_memory_usage'] = np.std(results['memory_usage'])
        
        return results
    
    def _compute_component_impact(self, baseline_results: Dict[str, Any],
                                ablated_results: Dict[str, Any],
                                component_name: str) -> Dict[str, float]:
        """Compute the impact of removing a component.
        
        Args:
            baseline_results: Results with full model
            ablated_results: Results with component removed
            component_name: Name of the component
            
        Returns:
            Dictionary containing impact metrics
        """
        baseline_error = baseline_results['mean_l2_error']
        ablated_error = ablated_results['mean_l2_error']
        
        # Compute percentage impact (positive means component helps)
        error_increase = ablated_error - baseline_error
        percentage_impact = (error_increase / baseline_error) * 100
        
        # Compute relative performance degradation
        relative_degradation = ablated_error / baseline_error
        
        impact = {
            'component': component_name,
            'baseline_l2_error': baseline_error,
            'ablated_l2_error': ablated_error,
            'error_increase': error_increase,
            'percentage_impact': percentage_impact,
            'relative_degradation': relative_degradation,
            'is_beneficial': error_increase > 0  # Component helps if removing it increases error
        }
        
        return impact
    
    def generate_sensitivity_curves(self) -> Dict[str, Any]:
        """Generate sensitivity curves showing performance vs hyperparameter values.
        
        Returns:
            Dictionary containing curve data for plotting
        """
        curves = {}
        
        # Inner steps sensitivity curve
        if 'inner_steps' in self.hyperparameter_results:
            inner_steps_data = self.hyperparameter_results['inner_steps']
            curves['inner_steps'] = {
                'x_values': list(inner_steps_data.keys()),
                'y_values': [results['mean_l2_error'] for results in inner_steps_data.values()],
                'y_errors': [results['std_l2_error'] for results in inner_steps_data.values()],
                'xlabel': 'Inner Steps (K)',
                'ylabel': 'L2 Relative Error',
                'title': 'Sensitivity to Number of Inner Steps'
            }
        
        # Meta batch size sensitivity curve
        if 'meta_batch_size' in self.hyperparameter_results:
            batch_size_data = self.hyperparameter_results['meta_batch_size']
            curves['meta_batch_size'] = {
                'x_values': list(batch_size_data.keys()),
                'y_values': [results['mean_l2_error'] for results in batch_size_data.values()],
                'y_errors': [results['std_l2_error'] for results in batch_size_data.values()],
                'xlabel': 'Meta Batch Size',
                'ylabel': 'L2 Relative Error',
                'title': 'Sensitivity to Meta Batch Size'
            }
        
        # Learning rate sensitivity curve
        if 'learning_rate' in self.hyperparameter_results:
            lr_data = self.hyperparameter_results['learning_rate']
            curves['learning_rate'] = {
                'x_values': list(lr_data.keys()),
                'y_values': [results['mean_l2_error'] for results in lr_data.values()],
                'y_errors': [results['std_l2_error'] for results in lr_data.values()],
                'xlabel': 'Learning Rate',
                'ylabel': 'L2 Relative Error',
                'title': 'Sensitivity to Learning Rate',
                'xscale': 'log'  # Use log scale for learning rates
            }
        
        return curves
    
    def get_component_ranking(self) -> List[Dict[str, Any]]:
        """Get ranking of components by their impact on performance.
        
        Returns:
            List of components ranked by importance (highest impact first)
        """
        if not self.component_impact_results:
            return []
        
        # Sort components by percentage impact (descending)
        ranked_components = sorted(
            self.component_impact_results.values(),
            key=lambda x: x['percentage_impact'],
            reverse=True
        )
        
        return ranked_components
    
    def get_optimal_hyperparameters(self) -> Dict[str, Any]:
        """Get optimal hyperparameter values based on sensitivity analysis.
        
        Returns:
            Dictionary containing optimal hyperparameter values
        """
        optimal_params = {}
        
        # Find optimal inner steps
        if 'inner_steps' in self.hyperparameter_results:
            inner_steps_data = self.hyperparameter_results['inner_steps']
            best_k = min(inner_steps_data.keys(), 
                        key=lambda k: inner_steps_data[k]['mean_l2_error'])
            optimal_params['inner_steps'] = {
                'value': best_k,
                'l2_error': inner_steps_data[best_k]['mean_l2_error']
            }
        
        # Find optimal meta batch size
        if 'meta_batch_size' in self.hyperparameter_results:
            batch_size_data = self.hyperparameter_results['meta_batch_size']
            best_batch_size = min(batch_size_data.keys(),
                                key=lambda bs: batch_size_data[bs]['mean_l2_error'])
            optimal_params['meta_batch_size'] = {
                'value': best_batch_size,
                'l2_error': batch_size_data[best_batch_size]['mean_l2_error']
            }
        
        # Find optimal learning rate
        if 'learning_rate' in self.hyperparameter_results:
            lr_data = self.hyperparameter_results['learning_rate']
            best_lr = min(lr_data.keys(),
                         key=lambda lr: lr_data[lr]['mean_l2_error'])
            optimal_params['learning_rate'] = {
                'value': best_lr,
                'l2_error': lr_data[best_lr]['mean_l2_error']
            }
        
        return optimal_params
    
    def generate_ablation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary of ablation study results.
        
        Returns:
            Dictionary containing complete ablation study summary
        """
        summary = {
            'baseline_performance': self.baseline_performance,
            'component_impacts': self.component_impact_results,
            'component_ranking': self.get_component_ranking(),
            'optimal_hyperparameters': self.get_optimal_hyperparameters(),
            'architecture_results': self.architecture_results,
            'hyperparameter_results': self.hyperparameter_results,
            'sensitivity_curves': self.generate_sensitivity_curves()
        }
        
        # Add interpretation
        if self.component_impact_results:
            most_important = max(self.component_impact_results.values(),
                               key=lambda x: x['percentage_impact'])
            least_important = min(self.component_impact_results.values(),
                                key=lambda x: x['percentage_impact'])
            
            summary['interpretation'] = {
                'most_critical_component': most_important['component'],
                'most_critical_impact': most_important['percentage_impact'],
                'least_critical_component': least_important['component'],
                'least_critical_impact': least_important['percentage_impact']
            }
        
        return summary
    
    def save_results(self, filepath: str):
        """Save ablation study results to file.
        
        Args:
            filepath: Path to save results
        """
        import json
        
        # Convert results to JSON-serializable format
        results = {
            'architecture_results': self._convert_to_serializable(self.architecture_results),
            'hyperparameter_results': self._convert_to_serializable(self.hyperparameter_results),
            'component_impact_results': self._convert_to_serializable(self.component_impact_results),
            'summary': self._convert_to_serializable(self.generate_ablation_summary())
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def _convert_to_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to lists."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        else:
            return obj


def run_comprehensive_ablation_study(base_config: PhysicsInformedMetaLearnerConfig,
                                   train_tasks: List[Task],
                                   test_tasks: List[Task],
                                   output_dir: str = "ablation_results") -> Dict[str, Any]:
    """Run comprehensive ablation study with all components.
    
    Args:
        base_config: Base configuration for meta-learning model
        train_tasks: Training tasks
        test_tasks: Test tasks
        output_dir: Directory to save results
        
    Returns:
        Complete ablation study results
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ablation study manager
    ablation_config = AblationConfig()
    manager = AblationStudyManager(base_config, ablation_config)
    
    print("Starting comprehensive ablation study...")
    
    # Run architecture ablations
    print("\n=== Architecture Component Ablations ===")
    architecture_results = manager.run_architecture_ablations(train_tasks, test_tasks)
    
    # Run hyperparameter sensitivity analysis
    print("\n=== Hyperparameter Sensitivity Analysis ===")
    hyperparameter_results = manager.run_hyperparameter_sensitivity(train_tasks, test_tasks)
    
    # Generate comprehensive summary
    print("\n=== Generating Summary ===")
    summary = manager.generate_ablation_summary()
    
    # Save results
    results_file = os.path.join(output_dir, "ablation_results.json")
    manager.save_results(results_file)
    
    print(f"\nAblation study completed. Results saved to {results_file}")
    
    # Print key findings
    print("\n=== Key Findings ===")
    if summary.get('interpretation'):
        interp = summary['interpretation']
        print(f"Most critical component: {interp['most_critical_component']} "
              f"({interp['most_critical_impact']:.1f}% impact)")
        print(f"Least critical component: {interp['least_critical_component']} "
              f"({interp['least_critical_impact']:.1f}% impact)")
    
    if summary.get('optimal_hyperparameters'):
        print("\nOptimal hyperparameters:")
        for param, info in summary['optimal_hyperparameters'].items():
            print(f"  {param}: {info['value']} (L2 error: {info['l2_error']:.4f})")
    
    return summary

class HyperparameterSensitivityAnalyzer:
    """Specialized class for detailed hyperparameter sensitivity analysis."""
    
    def __init__(self, base_config: PhysicsInformedMetaLearnerConfig):
        """Initialize hyperparameter sensitivity analyzer.
        
        Args:
            base_config: Base configuration for analysis
        """
        self.base_config = base_config
        self.sensitivity_results = {}
        self.performance_curves = {}
        
    def analyze_inner_steps_sensitivity(self, train_tasks: List[Task], 
                                      test_tasks: List[Task],
                                      k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """Analyze sensitivity to number of inner adaptation steps.
        
        Args:
            train_tasks: Training tasks
            test_tasks: Test tasks  
            k_values: List of K values to test
            
        Returns:
            Detailed sensitivity analysis results
        """
        print("Analyzing inner steps sensitivity...")
        
        results = {}
        performance_data = {'k_values': [], 'l2_errors': [], 'std_errors': [], 
                          'adaptation_times': [], 'convergence_rates': []}
        
        for k in k_values:
            print(f"Testing K = {k} inner steps...")
            
            config = copy.deepcopy(self.base_config)
            config.adaptation_steps = k
            
            # Evaluate configuration
            k_results = self._evaluate_hyperparameter_config(
                config, train_tasks, test_tasks, f"inner_steps_{k}"
            )
            
            results[k] = k_results
            
            # Store performance curve data
            performance_data['k_values'].append(k)
            performance_data['l2_errors'].append(k_results['mean_l2_error'])
            performance_data['std_errors'].append(k_results['std_l2_error'])
            performance_data['adaptation_times'].append(k_results['mean_adaptation_time'])
            
            # Compute convergence rate (improvement per step)
            if k > 1:
                prev_error = results[k_values[k_values.index(k)-1]]['mean_l2_error']
                convergence_rate = (prev_error - k_results['mean_l2_error']) / (k - k_values[k_values.index(k)-1])
                performance_data['convergence_rates'].append(convergence_rate)
            else:
                performance_data['convergence_rates'].append(0.0)
        
        # Find optimal K value
        optimal_k = min(results.keys(), key=lambda k: results[k]['mean_l2_error'])
        
        # Analyze diminishing returns
        diminishing_returns_analysis = self._analyze_diminishing_returns(
            performance_data['k_values'], performance_data['l2_errors']
        )
        
        analysis = {
            'results': results,
            'performance_curve': performance_data,
            'optimal_k': optimal_k,
            'optimal_performance': results[optimal_k]['mean_l2_error'],
            'diminishing_returns': diminishing_returns_analysis,
            'recommendation': self._generate_k_recommendation(results, optimal_k)
        }
        
        self.sensitivity_results['inner_steps'] = analysis
        return analysis
    
    def analyze_batch_size_sensitivity(self, train_tasks: List[Task],
                                     test_tasks: List[Task],
                                     batch_sizes: List[int] = [8, 16, 32, 64]) -> Dict[str, Any]:
        """Analyze sensitivity to meta batch size.
        
        Args:
            train_tasks: Training tasks
            test_tasks: Test tasks
            batch_sizes: List of batch sizes to test
            
        Returns:
            Detailed sensitivity analysis results
        """
        print("Analyzing meta batch size sensitivity...")
        
        results = {}
        performance_data = {'batch_sizes': [], 'l2_errors': [], 'std_errors': [],
                          'training_times': [], 'memory_usage': []}
        
        for batch_size in batch_sizes:
            print(f"Testing batch size = {batch_size}...")
            
            config = copy.deepcopy(self.base_config)
            config.meta_batch_size = batch_size
            
            # Evaluate configuration
            bs_results = self._evaluate_hyperparameter_config(
                config, train_tasks, test_tasks, f"batch_size_{batch_size}"
            )
            
            results[batch_size] = bs_results
            
            # Store performance curve data
            performance_data['batch_sizes'].append(batch_size)
            performance_data['l2_errors'].append(bs_results['mean_l2_error'])
            performance_data['std_errors'].append(bs_results['std_l2_error'])
            performance_data['training_times'].append(bs_results['mean_training_time'])
            
            if 'mean_memory_usage' in bs_results:
                performance_data['memory_usage'].append(bs_results['mean_memory_usage'])
        
        # Find optimal batch size
        optimal_bs = min(results.keys(), key=lambda bs: results[bs]['mean_l2_error'])
        
        # Analyze efficiency vs performance trade-off
        efficiency_analysis = self._analyze_efficiency_tradeoff(
            performance_data['batch_sizes'], 
            performance_data['l2_errors'],
            performance_data['training_times']
        )
        
        analysis = {
            'results': results,
            'performance_curve': performance_data,
            'optimal_batch_size': optimal_bs,
            'optimal_performance': results[optimal_bs]['mean_l2_error'],
            'efficiency_analysis': efficiency_analysis,
            'recommendation': self._generate_batch_size_recommendation(results, optimal_bs)
        }
        
        self.sensitivity_results['batch_size'] = analysis
        return analysis
    
    def analyze_learning_rate_sensitivity(self, train_tasks: List[Task],
                                        test_tasks: List[Task],
                                        learning_rates: List[float] = [1e-4, 5e-4, 1e-3, 5e-3]) -> Dict[str, Any]:
        """Analyze sensitivity to learning rates.
        
        Args:
            train_tasks: Training tasks
            test_tasks: Test tasks
            learning_rates: List of learning rates to test
            
        Returns:
            Detailed sensitivity analysis results
        """
        print("Analyzing learning rate sensitivity...")
        
        results = {}
        performance_data = {'learning_rates': [], 'l2_errors': [], 'std_errors': [],
                          'convergence_speeds': [], 'stability_scores': []}
        
        for lr in learning_rates:
            print(f"Testing learning rate = {lr}...")
            
            config = copy.deepcopy(self.base_config)
            config.meta_lr = lr
            config.adapt_lr = lr * 10  # Maintain 10x ratio
            
            # Evaluate configuration with convergence tracking
            lr_results = self._evaluate_hyperparameter_config(
                config, train_tasks, test_tasks, f"lr_{lr}", track_convergence=True
            )
            
            results[lr] = lr_results
            
            # Store performance curve data
            performance_data['learning_rates'].append(lr)
            performance_data['l2_errors'].append(lr_results['mean_l2_error'])
            performance_data['std_errors'].append(lr_results['std_l2_error'])
            
            # Compute convergence speed and stability
            if 'convergence_history' in lr_results:
                convergence_speed = self._compute_convergence_speed(lr_results['convergence_history'])
                stability_score = self._compute_stability_score(lr_results['convergence_history'])
            else:
                convergence_speed = 0.0
                stability_score = 0.0
            
            performance_data['convergence_speeds'].append(convergence_speed)
            performance_data['stability_scores'].append(stability_score)
        
        # Find optimal learning rate
        optimal_lr = min(results.keys(), key=lambda lr: results[lr]['mean_l2_error'])
        
        # Analyze learning rate landscape
        landscape_analysis = self._analyze_lr_landscape(
            performance_data['learning_rates'],
            performance_data['l2_errors'],
            performance_data['stability_scores']
        )
        
        analysis = {
            'results': results,
            'performance_curve': performance_data,
            'optimal_lr': optimal_lr,
            'optimal_performance': results[optimal_lr]['mean_l2_error'],
            'landscape_analysis': landscape_analysis,
            'recommendation': self._generate_lr_recommendation(results, optimal_lr)
        }
        
        self.sensitivity_results['learning_rate'] = analysis
        return analysis
    
    def _evaluate_hyperparameter_config(self, config: PhysicsInformedMetaLearnerConfig,
                                      train_tasks: List[Task], test_tasks: List[Task],
                                      config_name: str, track_convergence: bool = False) -> Dict[str, Any]:
        """Evaluate a hyperparameter configuration with detailed tracking.
        
        Args:
            config: Configuration to evaluate
            train_tasks: Training tasks
            test_tasks: Test tasks
            config_name: Name for this configuration
            track_convergence: Whether to track convergence history
            
        Returns:
            Detailed evaluation results
        """
        results = {
            'config_name': config_name,
            'l2_errors': [],
            'adaptation_times': [],
            'training_times': [],
            'memory_usage': []
        }
        
        if track_convergence:
            results['convergence_history'] = []
        
        # Run multiple evaluation runs
        n_runs = 3  # Reduced for hyperparameter sensitivity
        
        for run in range(n_runs):
            # Initialize model
            model = PhysicsInformedMetaLearner(config)
            
            # Track training time and convergence
            start_time = time.time()
            
            if track_convergence:
                # Custom training with convergence tracking
                convergence_history = self._train_with_convergence_tracking(
                    model, train_tasks[:15], iterations=50
                )
                results['convergence_history'].append(convergence_history)
            else:
                # Standard training
                model.meta_train(train_tasks[:15], meta_iterations=50)
            
            training_time = time.time() - start_time
            results['training_times'].append(training_time)
            
            # Evaluate on test tasks
            test_subset = test_tasks[:10]  # Smaller subset for sensitivity analysis
            run_l2_errors = []
            run_adaptation_times = []
            
            for task in test_subset:
                model.set_pde_problem(task.metadata.get('pde_problem'))
                
                adapt_start = time.time()
                adapted_params = model.fast_adapt(task.support_data, task, k_shots=5, adaptation_steps=5)
                adaptation_time = time.time() - adapt_start
                run_adaptation_times.append(adaptation_time)
                
                with torch.no_grad():
                    predictions = model.forward(task.query_data.inputs, adapted_params)
                    l2_error = compute_l2_relative_error(
                        predictions.cpu().numpy(), 
                        task.query_data.outputs.cpu().numpy()
                    )
                    run_l2_errors.append(l2_error)
            
            results['l2_errors'].extend(run_l2_errors)
            results['adaptation_times'].extend(run_adaptation_times)
            
            # Memory tracking
            if torch.cuda.is_available():
                memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
                results['memory_usage'].append(memory_mb)
                torch.cuda.reset_peak_memory_stats()
        
        # Compute statistics
        results['mean_l2_error'] = np.mean(results['l2_errors'])
        results['std_l2_error'] = np.std(results['l2_errors'])
        results['mean_adaptation_time'] = np.mean(results['adaptation_times'])
        results['std_adaptation_time'] = np.std(results['adaptation_times'])
        results['mean_training_time'] = np.mean(results['training_times'])
        results['std_training_time'] = np.std(results['training_times'])
        
        if results['memory_usage']:
            results['mean_memory_usage'] = np.mean(results['memory_usage'])
            results['std_memory_usage'] = np.std(results['memory_usage'])
        
        return results
    
    def _train_with_convergence_tracking(self, model: PhysicsInformedMetaLearner,
                                       train_tasks: List[Task], iterations: int) -> List[float]:
        """Train model while tracking convergence.
        
        Args:
            model: Model to train
            train_tasks: Training tasks
            iterations: Number of iterations
            
        Returns:
            List of loss values showing convergence
        """
        convergence_history = []
        
        for iteration in range(iterations):
            # Sample batch of tasks
            task_indices = np.random.choice(len(train_tasks), 
                                          min(model.config.meta_batch_size, len(train_tasks)), 
                                          replace=False)
            task_batch = TaskBatch([train_tasks[i] for i in task_indices])
            
            # Perform meta-training step
            metrics = model.meta_train_step(task_batch)
            convergence_history.append(metrics['meta_loss'])
        
        return convergence_history
    
    def _analyze_diminishing_returns(self, x_values: List, y_values: List) -> Dict[str, Any]:
        """Analyze diminishing returns in performance improvement.
        
        Args:
            x_values: X-axis values (e.g., number of steps)
            y_values: Y-axis values (e.g., L2 errors)
            
        Returns:
            Diminishing returns analysis
        """
        if len(x_values) < 3:
            return {'analysis': 'Insufficient data for diminishing returns analysis'}
        
        # Compute improvement rates
        improvement_rates = []
        for i in range(1, len(y_values)):
            rate = (y_values[i-1] - y_values[i]) / (x_values[i] - x_values[i-1])
            improvement_rates.append(rate)
        
        # Find knee point (where improvement rate drops significantly)
        knee_point = None
        if len(improvement_rates) >= 2:
            for i in range(1, len(improvement_rates)):
                if improvement_rates[i] < 0.5 * improvement_rates[i-1]:
                    knee_point = x_values[i]
                    break
        
        return {
            'improvement_rates': improvement_rates,
            'knee_point': knee_point,
            'diminishing_returns_detected': knee_point is not None,
            'recommendation': f"Consider using {knee_point} steps for optimal efficiency" if knee_point else "No clear diminishing returns detected"
        }
    
    def _analyze_efficiency_tradeoff(self, batch_sizes: List[int], 
                                   l2_errors: List[float], 
                                   training_times: List[float]) -> Dict[str, Any]:
        """Analyze efficiency vs performance trade-off.
        
        Args:
            batch_sizes: List of batch sizes
            l2_errors: Corresponding L2 errors
            training_times: Corresponding training times
            
        Returns:
            Efficiency analysis
        """
        # Compute efficiency scores (lower error / higher time = better efficiency)
        efficiency_scores = []
        for error, time in zip(l2_errors, training_times):
            efficiency = 1.0 / (error * time)  # Higher is better
            efficiency_scores.append(efficiency)
        
        # Find most efficient configuration
        best_efficiency_idx = np.argmax(efficiency_scores)
        most_efficient_bs = batch_sizes[best_efficiency_idx]
        
        # Find best performance configuration
        best_performance_idx = np.argmin(l2_errors)
        best_performance_bs = batch_sizes[best_performance_idx]
        
        return {
            'efficiency_scores': efficiency_scores,
            'most_efficient_batch_size': most_efficient_bs,
            'best_performance_batch_size': best_performance_bs,
            'efficiency_vs_performance_tradeoff': most_efficient_bs != best_performance_bs,
            'recommendation': f"Use batch size {most_efficient_bs} for best efficiency, {best_performance_bs} for best performance"
        }
    
    def _analyze_lr_landscape(self, learning_rates: List[float],
                            l2_errors: List[float],
                            stability_scores: List[float]) -> Dict[str, Any]:
        """Analyze learning rate landscape.
        
        Args:
            learning_rates: List of learning rates
            l2_errors: Corresponding L2 errors
            stability_scores: Corresponding stability scores
            
        Returns:
            Learning rate landscape analysis
        """
        # Find stable region (high stability, low error)
        combined_scores = []
        for error, stability in zip(l2_errors, stability_scores):
            # Combined score: lower error + higher stability
            score = stability / (1.0 + error)
            combined_scores.append(score)
        
        best_combined_idx = np.argmax(combined_scores)
        optimal_lr = learning_rates[best_combined_idx]
        
        # Detect unstable regions
        unstable_lrs = [lr for lr, stability in zip(learning_rates, stability_scores) 
                       if stability < 0.5]
        
        return {
            'combined_scores': combined_scores,
            'optimal_lr_combined': optimal_lr,
            'unstable_learning_rates': unstable_lrs,
            'stable_range': [min(lr for lr, s in zip(learning_rates, stability_scores) if s > 0.7),
                           max(lr for lr, s in zip(learning_rates, stability_scores) if s > 0.7)] if any(s > 0.7 for s in stability_scores) else None,
            'recommendation': f"Use learning rate {optimal_lr} for best stability-performance balance"
        }
    
    def _compute_convergence_speed(self, convergence_history: List[float]) -> float:
        """Compute convergence speed from training history.
        
        Args:
            convergence_history: List of loss values during training
            
        Returns:
            Convergence speed metric
        """
        if len(convergence_history) < 10:
            return 0.0
        
        # Compute average improvement rate over first half of training
        first_half = convergence_history[:len(convergence_history)//2]
        if len(first_half) < 2:
            return 0.0
        
        initial_loss = first_half[0]
        final_loss = first_half[-1]
        
        if initial_loss <= final_loss:
            return 0.0
        
        improvement_rate = (initial_loss - final_loss) / len(first_half)
        return improvement_rate
    
    def _compute_stability_score(self, convergence_history: List[float]) -> float:
        """Compute stability score from training history.
        
        Args:
            convergence_history: List of loss values during training
            
        Returns:
            Stability score (0-1, higher is more stable)
        """
        if len(convergence_history) < 5:
            return 0.0
        
        # Compute coefficient of variation in second half (after initial convergence)
        second_half = convergence_history[len(convergence_history)//2:]
        
        if len(second_half) < 2:
            return 0.0
        
        mean_loss = np.mean(second_half)
        std_loss = np.std(second_half)
        
        if mean_loss == 0:
            return 1.0
        
        cv = std_loss / mean_loss
        stability_score = 1.0 / (1.0 + cv)  # Higher stability = lower coefficient of variation
        
        return stability_score
    
    def _generate_k_recommendation(self, results: Dict[int, Dict], optimal_k: int) -> str:
        """Generate recommendation for inner steps K.
        
        Args:
            results: Results for different K values
            optimal_k: Optimal K value
            
        Returns:
            Recommendation string
        """
        baseline_k = 5  # Standard baseline
        
        if optimal_k == baseline_k:
            return f"The standard K={baseline_k} steps provides optimal performance."
        elif optimal_k < baseline_k:
            improvement = results[baseline_k]['mean_l2_error'] - results[optimal_k]['mean_l2_error']
            return f"Use K={optimal_k} steps for {improvement:.4f} improvement in L2 error with faster adaptation."
        else:
            improvement = results[baseline_k]['mean_l2_error'] - results[optimal_k]['mean_l2_error']
            return f"Use K={optimal_k} steps for {improvement:.4f} improvement in L2 error, though adaptation will be slower."
    
    def _generate_batch_size_recommendation(self, results: Dict[int, Dict], optimal_bs: int) -> str:
        """Generate recommendation for batch size.
        
        Args:
            results: Results for different batch sizes
            optimal_bs: Optimal batch size
            
        Returns:
            Recommendation string
        """
        return f"Use batch size {optimal_bs} for optimal performance. " \
               f"L2 error: {results[optimal_bs]['mean_l2_error']:.4f}, " \
               f"Training time: {results[optimal_bs]['mean_training_time']:.2f}s"
    
    def _generate_lr_recommendation(self, results: Dict[float, Dict], optimal_lr: float) -> str:
        """Generate recommendation for learning rate.
        
        Args:
            results: Results for different learning rates
            optimal_lr: Optimal learning rate
            
        Returns:
            Recommendation string
        """
        return f"Use learning rate {optimal_lr} for optimal performance. " \
               f"L2 error: {results[optimal_lr]['mean_l2_error']:.4f}. " \
               f"Maintain 10x ratio for adaptation learning rate ({optimal_lr * 10})."
    
    def generate_sensitivity_report(self) -> str:
        """Generate comprehensive sensitivity analysis report.
        
        Returns:
            Formatted report string
        """
        report = "# Hyperparameter Sensitivity Analysis Report\n\n"
        
        if 'inner_steps' in self.sensitivity_results:
            inner_analysis = self.sensitivity_results['inner_steps']
            report += "## Inner Steps (K) Sensitivity\n"
            report += f"- Optimal K: {inner_analysis['optimal_k']}\n"
            report += f"- Optimal Performance: {inner_analysis['optimal_performance']:.4f}\n"
            report += f"- Recommendation: {inner_analysis['recommendation']}\n\n"
        
        if 'batch_size' in self.sensitivity_results:
            batch_analysis = self.sensitivity_results['batch_size']
            report += "## Meta Batch Size Sensitivity\n"
            report += f"- Optimal Batch Size: {batch_analysis['optimal_batch_size']}\n"
            report += f"- Optimal Performance: {batch_analysis['optimal_performance']:.4f}\n"
            report += f"- Recommendation: {batch_analysis['recommendation']}\n\n"
        
        if 'learning_rate' in self.sensitivity_results:
            lr_analysis = self.sensitivity_results['learning_rate']
            report += "## Learning Rate Sensitivity\n"
            report += f"- Optimal Learning Rate: {lr_analysis['optimal_lr']}\n"
            report += f"- Optimal Performance: {lr_analysis['optimal_performance']:.4f}\n"
            report += f"- Recommendation: {lr_analysis['recommendation']}\n\n"
        
        return report
class
 AblationTableGenerator:
    """Generates LaTeX tables for ablation study results."""
    
    def __init__(self, ablation_manager: AblationStudyManager):
        """Initialize table generator.
        
        Args:
            ablation_manager: Ablation study manager with results
        """
        self.manager = ablation_manager
        
    def generate_architecture_ablation_table(self) -> str:
        """Generate Table D.13: Architecture Component Ablations.
        
        Returns:
            LaTeX table string
        """
        if not self.manager.architecture_results:
            return "% No architecture ablation results available"
        
        table_header = r"""
\begin{table}[htbp]
\centering
\caption{Architecture Component Ablation Study Results}
\label{tab:architecture_ablations}
\begin{tabular}{lcccc}
\toprule
Configuration & L2 Error ($\downarrow$) & Std Dev & Impact (\%) & Critical \\
\midrule
"""
        
        table_rows = []
        baseline_error = None
        
        # Get baseline performance
        if 'full_model' in self.manager.architecture_results:
            baseline_error = self.manager.architecture_results['full_model']['mean_l2_error']
            baseline_std = self.manager.architecture_results['full_model']['std_l2_error']
            table_rows.append(f"Full Model (Baseline) & {baseline_error:.4f} & {baseline_std:.4f} & -- & -- \\\\")
        
        # Add component ablations
        component_order = ['no_adaptive_weights', 'no_physics_regularization', 'no_multiscale_loss']
        component_names = {
            'no_adaptive_weights': 'w/o Adaptive Weights',
            'no_physics_regularization': 'w/o Physics Regularization', 
            'no_multiscale_loss': 'w/o Multi-scale Loss'
        }
        
        for component in component_order:
            if component in self.manager.architecture_results:
                results = self.manager.architecture_results[component]
                error = results['mean_l2_error']
                std = results['std_l2_error']
                
                # Compute impact percentage
                if baseline_error is not None:
                    impact = ((error - baseline_error) / baseline_error) * 100
                    critical = "Yes" if impact > 5.0 else "No"  # >5% impact considered critical
                else:
                    impact = 0.0
                    critical = "Unknown"
                
                name = component_names.get(component, component)
                table_rows.append(f"{name} & {error:.4f} & {std:.4f} & {impact:+.1f} & {critical} \\\\")
        
        # Add architecture variations
        arch_configs = ['small_3x32', 'medium_3x128', 'medium_3x256', 'deep_8x64', 'deep_8x256']
        arch_names = {
            'small_3x32': '3 layers × 32 width',
            'medium_3x128': '3 layers × 128 width',
            'medium_3x256': '3 layers × 256 width', 
            'deep_8x64': '8 layers × 64 width',
            'deep_8x256': '8 layers × 256 width'
        }
        
        if any(arch in self.manager.architecture_results for arch in arch_configs):
            table_rows.append("\\midrule")
            table_rows.append("\\multicolumn{5}{c}{\\textit{Architecture Variations}} \\\\")
            table_rows.append("\\midrule")
            
            for arch in arch_configs:
                if arch in self.manager.architecture_results:
                    results = self.manager.architecture_results[arch]
                    error = results['mean_l2_error']
                    std = results['std_l2_error']
                    
                    if baseline_error is not None:
                        impact = ((error - baseline_error) / baseline_error) * 100
                        critical = "Yes" if abs(impact) > 10.0 else "No"
                    else:
                        impact = 0.0
                        critical = "Unknown"
                    
                    name = arch_names.get(arch, arch)
                    table_rows.append(f"{name} & {error:.4f} & {std:.4f} & {impact:+.1f} & {critical} \\\\")
        
        table_footer = r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Impact percentage shows performance change relative to full model baseline.
\item Critical components have $>5\%$ impact on L2 error when removed.
\item Lower L2 error indicates better performance.
\end{tablenotes}
\end{table}
"""
        
        return table_header + "\n".join(table_rows) + "\n" + table_footer
    
    def generate_hyperparameter_sensitivity_table(self) -> str:
        """Generate Table D.14: Hyperparameter Sensitivity Analysis.
        
        Returns:
            LaTeX table string
        """
        if not self.manager.hyperparameter_results:
            return "% No hyperparameter sensitivity results available"
        
        table_header = r"""
\begin{table}[htbp]
\centering
\caption{Hyperparameter Sensitivity Analysis Results}
\label{tab:hyperparameter_sensitivity}
\begin{tabular}{lccccc}
\toprule
Hyperparameter & Value & L2 Error ($\downarrow$) & Std Dev & Adapt Time (s) & Optimal \\
\midrule
"""
        
        table_rows = []
        
        # Inner steps sensitivity
        if 'inner_steps' in self.manager.hyperparameter_results:
            inner_results = self.manager.hyperparameter_results['inner_steps']
            optimal_k = min(inner_results.keys(), key=lambda k: inner_results[k]['mean_l2_error'])
            
            table_rows.append("\\multicolumn{6}{c}{\\textit{Inner Steps (K)}} \\\\")
            table_rows.append("\\midrule")
            
            for k in sorted(inner_results.keys()):
                results = inner_results[k]
                error = results['mean_l2_error']
                std = results['std_l2_error']
                adapt_time = results['mean_adaptation_time']
                is_optimal = "\\checkmark" if k == optimal_k else ""
                
                table_rows.append(f"K & {k} & {error:.4f} & {std:.4f} & {adapt_time:.3f} & {is_optimal} \\\\")
        
        # Meta batch size sensitivity
        if 'meta_batch_size' in self.manager.hyperparameter_results:
            batch_results = self.manager.hyperparameter_results['meta_batch_size']
            optimal_bs = min(batch_results.keys(), key=lambda bs: batch_results[bs]['mean_l2_error'])
            
            if table_rows:  # Add separator if previous section exists
                table_rows.append("\\midrule")
            
            table_rows.append("\\multicolumn{6}{c}{\\textit{Meta Batch Size}} \\\\")
            table_rows.append("\\midrule")
            
            for bs in sorted(batch_results.keys()):
                results = batch_results[bs]
                error = results['mean_l2_error']
                std = results['std_l2_error']
                adapt_time = results.get('mean_adaptation_time', 0.0)
                is_optimal = "\\checkmark" if bs == optimal_bs else ""
                
                table_rows.append(f"Batch Size & {bs} & {error:.4f} & {std:.4f} & {adapt_time:.3f} & {is_optimal} \\\\")
        
        # Learning rate sensitivity
        if 'learning_rate' in self.manager.hyperparameter_results:
            lr_results = self.manager.hyperparameter_results['learning_rate']
            optimal_lr = min(lr_results.keys(), key=lambda lr: lr_results[lr]['mean_l2_error'])
            
            if table_rows:  # Add separator if previous section exists
                table_rows.append("\\midrule")
            
            table_rows.append("\\multicolumn{6}{c}{\\textit{Learning Rate}} \\\\")
            table_rows.append("\\midrule")
            
            for lr in sorted(lr_results.keys()):
                results = lr_results[lr]
                error = results['mean_l2_error']
                std = results['std_l2_error']
                adapt_time = results.get('mean_adaptation_time', 0.0)
                is_optimal = "\\checkmark" if lr == optimal_lr else ""
                
                table_rows.append(f"Meta LR & {lr:.0e} & {error:.4f} & {std:.4f} & {adapt_time:.3f} & {is_optimal} \\\\")
        
        table_footer = r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Optimal configurations marked with \checkmark.
\item Adaptation time measured for K=5 shots, S=5 steps.
\item Learning rate includes both meta-learning rate and adaptation rate (10× ratio).
\end{tablenotes}
\end{table}
"""
        
        return table_header + "\n".join(table_rows) + "\n" + table_footer
    
    def generate_component_impact_summary_table(self) -> str:
        """Generate summary table of component impacts.
        
        Returns:
            LaTeX table string
        """
        if not self.manager.component_impact_results:
            return "% No component impact results available"
        
        table_header = r"""
\begin{table}[htbp]
\centering
\caption{Component Impact Summary}
\label{tab:component_impact_summary}
\begin{tabular}{lccc}
\toprule
Component & Impact (\%) & Degradation & Critical \\
\midrule
"""
        
        table_rows = []
        
        # Sort components by impact (most important first)
        sorted_components = sorted(
            self.manager.component_impact_results.values(),
            key=lambda x: x['percentage_impact'],
            reverse=True
        )
        
        component_display_names = {
            'adaptive_weights': 'Adaptive Constraint Weighting',
            'physics_regularization': 'Physics Regularization',
            'multiscale_loss': 'Multi-scale Loss Handling'
        }
        
        for component_data in sorted_components:
            component = component_data['component']
            impact = component_data['percentage_impact']
            degradation = component_data['relative_degradation']
            is_critical = "Yes" if impact > 5.0 else "No"
            
            display_name = component_display_names.get(component, component.replace('_', ' ').title())
            table_rows.append(f"{display_name} & {impact:+.1f} & {degradation:.2f}× & {is_critical} \\\\")
        
        table_footer = r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Impact percentage: performance change when component is removed.
\item Degradation: ratio of ablated performance to baseline performance.
\item Critical: components with $>5\%$ performance impact.
\end{tablenotes}
\end{table}
"""
        
        return table_header + "\n".join(table_rows) + "\n" + table_footer
    
    def generate_optimal_configuration_table(self) -> str:
        """Generate table showing optimal hyperparameter configuration.
        
        Returns:
            LaTeX table string
        """
        optimal_params = self.manager.get_optimal_hyperparameters()
        
        if not optimal_params:
            return "% No optimal hyperparameter results available"
        
        table_header = r"""
\begin{table}[htbp]
\centering
\caption{Optimal Hyperparameter Configuration}
\label{tab:optimal_hyperparameters}
\begin{tabular}{lcc}
\toprule
Hyperparameter & Optimal Value & L2 Error \\
\midrule
"""
        
        table_rows = []
        
        param_display_names = {
            'inner_steps': 'Inner Steps (K)',
            'meta_batch_size': 'Meta Batch Size',
            'learning_rate': 'Meta Learning Rate'
        }
        
        for param_name, param_info in optimal_params.items():
            display_name = param_display_names.get(param_name, param_name.replace('_', ' ').title())
            value = param_info['value']
            error = param_info['l2_error']
            
            if param_name == 'learning_rate':
                value_str = f"{value:.0e}"
            else:
                value_str = str(value)
            
            table_rows.append(f"{display_name} & {value_str} & {error:.4f} \\\\")
        
        table_footer = r"""
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Configuration achieving lowest L2 relative error in sensitivity analysis.
\item Adaptation learning rate maintains 10× ratio with meta learning rate.
\end{tablenotes}
\end{table}
"""
        
        return table_header + "\n".join(table_rows) + "\n" + table_footer
    
    def generate_all_ablation_tables(self) -> Dict[str, str]:
        """Generate all ablation study tables.
        
        Returns:
            Dictionary mapping table names to LaTeX strings
        """
        tables = {
            'architecture_ablations': self.generate_architecture_ablation_table(),
            'hyperparameter_sensitivity': self.generate_hyperparameter_sensitivity_table(),
            'component_impact_summary': self.generate_component_impact_summary_table(),
            'optimal_configuration': self.generate_optimal_configuration_table()
        }
        
        return tables
    
    def save_tables_to_files(self, output_dir: str):
        """Save all tables to separate LaTeX files.
        
        Args:
            output_dir: Directory to save table files
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        tables = self.generate_all_ablation_tables()
        
        for table_name, table_content in tables.items():
            filepath = os.path.join(output_dir, f"table_{table_name}.tex")
            with open(filepath, 'w') as f:
                f.write(table_content)
            print(f"Saved {table_name} table to {filepath}")
    
    def generate_ablation_appendix_section(self) -> str:
        """Generate complete appendix section with all ablation tables.
        
        Returns:
            LaTeX appendix section string
        """
        section_header = r"""
\section{Ablation Studies}
\label{sec:ablation_studies}

This appendix presents comprehensive ablation studies analyzing the contribution of different architectural components and hyperparameter sensitivity for our meta-learning approach.

\subsection{Architecture Component Analysis}

Table~\ref{tab:architecture_ablations} shows the impact of removing key architectural components from our PhysicsInformedMetaLearner model. We systematically remove adaptive constraint weighting, physics regularization, and multi-scale loss handling to quantify their individual contributions.

"""
        
        architecture_table = self.generate_architecture_ablation_table()
        
        component_analysis = r"""

Table~\ref{tab:component_impact_summary} summarizes the relative importance of each component based on performance degradation when removed.

"""
        
        component_table = self.generate_component_impact_summary_table()
        
        hyperparameter_section = r"""

\subsection{Hyperparameter Sensitivity Analysis}

Table~\ref{tab:hyperparameter_sensitivity} presents sensitivity analysis results for key hyperparameters: number of inner adaptation steps (K), meta batch size, and learning rates.

"""
        
        hyperparameter_table = self.generate_hyperparameter_sensitivity_table()
        
        optimal_section = r"""

Table~\ref{tab:optimal_hyperparameters} shows the optimal hyperparameter configuration identified through our sensitivity analysis.

"""
        
        optimal_table = self.generate_optimal_configuration_table()
        
        interpretation_section = r"""

\subsection{Key Findings and Interpretation}

Our ablation studies reveal several important insights:

\begin{itemize}
\item \textbf{Critical Components}: """ + self._generate_critical_components_text() + r"""
\item \textbf{Optimal Hyperparameters}: """ + self._generate_optimal_hyperparameters_text() + r"""
\item \textbf{Architecture Sensitivity}: """ + self._generate_architecture_sensitivity_text() + r"""
\end{itemize}

These findings validate our design choices and provide guidance for practitioners implementing meta-learning PINNs in different domains.
"""
        
        return (section_header + architecture_table + component_analysis + component_table + 
                hyperparameter_section + hyperparameter_table + optimal_section + optimal_table + 
                interpretation_section)
    
    def _generate_critical_components_text(self) -> str:
        """Generate text describing critical components."""
        if not self.manager.component_impact_results:
            return "Component analysis not available."
        
        ranking = self.manager.get_component_ranking()
        if not ranking:
            return "No component ranking available."
        
        most_critical = ranking[0]
        component_names = {
            'adaptive_weights': 'adaptive constraint weighting',
            'physics_regularization': 'physics regularization',
            'multiscale_loss': 'multi-scale loss handling'
        }
        
        name = component_names.get(most_critical['component'], most_critical['component'])
        impact = most_critical['percentage_impact']
        
        return f"The most critical component is {name}, with {impact:.1f}\\% performance impact when removed."
    
    def _generate_optimal_hyperparameters_text(self) -> str:
        """Generate text describing optimal hyperparameters."""
        optimal_params = self.manager.get_optimal_hyperparameters()
        
        if not optimal_params:
            return "Optimal hyperparameter analysis not available."
        
        text_parts = []
        
        if 'inner_steps' in optimal_params:
            k = optimal_params['inner_steps']['value']
            text_parts.append(f"K={k} inner adaptation steps")
        
        if 'meta_batch_size' in optimal_params:
            bs = optimal_params['meta_batch_size']['value']
            text_parts.append(f"batch size of {bs}")
        
        if 'learning_rate' in optimal_params:
            lr = optimal_params['learning_rate']['value']
            text_parts.append(f"meta learning rate of {lr:.0e}")
        
        if text_parts:
            return "Optimal configuration uses " + ", ".join(text_parts) + "."
        else:
            return "Optimal configuration analysis not available."
    
    def _generate_architecture_sensitivity_text(self) -> str:
        """Generate text describing architecture sensitivity."""
        if not self.manager.architecture_results:
            return "Architecture sensitivity analysis not available."
        
        # Find best and worst architectures
        arch_results = {k: v for k, v in self.manager.architecture_results.items() 
                       if k not in ['full_model', 'no_adaptive_weights', 'no_physics_regularization', 'no_multiscale_loss']}
        
        if not arch_results:
            return "No architecture variations tested."
        
        best_arch = min(arch_results.keys(), key=lambda k: arch_results[k]['mean_l2_error'])
        worst_arch = max(arch_results.keys(), key=lambda k: arch_results[k]['mean_l2_error'])
        
        best_error = arch_results[best_arch]['mean_l2_error']
        worst_error = arch_results[worst_arch]['mean_l2_error']
        
        return f"Architecture variations show {((worst_error - best_error) / best_error * 100):.1f}\\% performance range, with {best_arch} performing best."


def generate_comprehensive_ablation_tables(ablation_manager: AblationStudyManager,
                                         output_dir: str = "ablation_tables") -> Dict[str, str]:
    """Generate comprehensive ablation study tables and appendix section.
    
    Args:
        ablation_manager: Ablation study manager with results
        output_dir: Directory to save table files
        
    Returns:
        Dictionary containing all generated tables and sections
    """
    table_generator = AblationTableGenerator(ablation_manager)
    
    # Generate all tables
    tables = table_generator.generate_all_ablation_tables()
    
    # Generate appendix section
    appendix_section = table_generator.generate_ablation_appendix_section()
    tables['appendix_section'] = appendix_section
    
    # Save to files
    table_generator.save_tables_to_files(output_dir)
    
    # Save appendix section
    import os
    appendix_file = os.path.join(output_dir, "ablation_appendix.tex")
    with open(appendix_file, 'w') as f:
        f.write(appendix_section)
    print(f"Saved ablation appendix section to {appendix_file}")
    
    return tables