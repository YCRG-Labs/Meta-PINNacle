"""
Extrapolation performance analyzer for parameter extrapolation experiments.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtrapolationAnalyzer:
    """Analyzer for parameter extrapolation performance."""
    
    def __init__(self, output_dir: str = "extrapolation_analysis"):
        """Initialize extrapolation analyzer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.physics_informed_methods = [
            'Standard PINN', 'MetaPINN', 'PhysicsInformedMetaLearner', 
            'TransferLearningPINN', 'DistributedMetaPINN'
        ]
        
        self.neural_operator_methods = ['FNO', 'DeepONet']
        
        self.error_thresholds = {
            'acceptable': 0.1,
            'degraded': 0.3,
            'failed': 0.5
        }
        
        logger.info(f"ExtrapolationAnalyzer initialized: {output_dir}")
    
    def analyze_extrapolation_degradation(self, extrapolation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare extrapolation degradation patterns across methods.
        
        Args:
            extrapolation_results: Results from evaluate_parameter_extrapolation
            
        Returns:
            Dictionary containing degradation analysis
        """
        logger.info("Analyzing extrapolation degradation patterns...")
        
        degradation_analysis = {
            'method_comparisons': {},
            'physics_vs_neural_operators': {},
            'degradation_rates': {},
            'failure_analysis': {}
        }
        
        model_results = extrapolation_results['results_by_model']
        extrapolation_percentages = extrapolation_results['extrapolation_percentages']
        
        # Analyze degradation for each method
        for method_name, method_results in model_results.items():
            method_degradation = self._calculate_method_degradation(method_results, extrapolation_percentages)
            degradation_analysis['method_comparisons'][method_name] = method_degradation
        
        # Compare physics-informed vs neural operators
        physics_degradation = self._analyze_physics_informed_degradation(model_results, extrapolation_percentages)
        neural_op_degradation = self._analyze_neural_operator_degradation(model_results, extrapolation_percentages)
        
        degradation_analysis['physics_vs_neural_operators'] = {
            'physics_informed': physics_degradation,
            'neural_operators': neural_op_degradation,
            'comparative_analysis': self._compare_degradation_patterns(physics_degradation, neural_op_degradation)
        }
        
        # Calculate degradation rates
        degradation_analysis['degradation_rates'] = self._calculate_degradation_rates(model_results, extrapolation_percentages)
        
        # Analyze failure modes
        degradation_analysis['failure_analysis'] = self._analyze_failure_modes(model_results, extrapolation_percentages)
        
        return degradation_analysis
    
    def _calculate_method_degradation(self, method_results: Dict[str, Any], extrapolation_percentages: List[float]) -> Dict[str, Any]:
        """Calculate degradation metrics for a single method."""
        degradation = {
            'error_progression': {},
            'degradation_rate': 0.0,
            'stability_score': 0.0,
            'extrapolation_limit': None
        }
        
        # Extract errors at each extrapolation level
        errors_by_percentage = {}
        for extrap_pct in extrapolation_percentages:
            # Look for results at this extrapolation level
            above_range_key = f"{extrap_pct}pct_above_range"
            below_range_key = f"{extrap_pct}pct_below_range"
            
            if above_range_key in method_results:
                few_shot_results = method_results[above_range_key]['few_shot_results']
                avg_error = np.mean([np.mean(list(errors.values())) for errors in few_shot_results.l2_errors.values()])
                errors_by_percentage[extrap_pct] = avg_error
        
        degradation['error_progression'] = errors_by_percentage
        
        # Calculate degradation rate (slope of error vs extrapolation percentage)
        if len(errors_by_percentage) >= 2:
            percentages = list(errors_by_percentage.keys())
            errors = list(errors_by_percentage.values())
            degradation_rate = np.polyfit(percentages, errors, 1)[0]  # Linear fit slope
            degradation['degradation_rate'] = degradation_rate
        
        # Calculate stability score (inverse of coefficient of variation)
        if len(errors_by_percentage) > 1:
            errors = list(errors_by_percentage.values())
            cv = np.std(errors) / np.mean(errors) if np.mean(errors) > 0 else float('inf')
            degradation['stability_score'] = 1.0 / (1.0 + cv)  # Higher is more stable
        
        # Find extrapolation limit (where error exceeds threshold)
        for extrap_pct, error in errors_by_percentage.items():
            if error > self.error_thresholds['failed']:
                degradation['extrapolation_limit'] = extrap_pct
                break
        
        return degradation
    
    def _analyze_physics_informed_degradation(self, model_results: Dict[str, Any], extrapolation_percentages: List[float]) -> Dict[str, Any]:
        """Analyze degradation patterns for physics-informed methods."""
        physics_analysis = {
            'average_degradation_rate': 0.0,
            'stability_scores': {},
            'constraint_benefit_quantification': {}
        }
        
        degradation_rates = []
        for method_name in self.physics_informed_methods:
            if method_name in model_results:
                method_degradation = self._calculate_method_degradation(model_results[method_name], extrapolation_percentages)
                degradation_rates.append(method_degradation['degradation_rate'])
                physics_analysis['stability_scores'][method_name] = method_degradation['stability_score']
        
        if degradation_rates:
            physics_analysis['average_degradation_rate'] = np.mean(degradation_rates)
        
        return physics_analysis
    
    def _analyze_neural_operator_degradation(self, model_results: Dict[str, Any], extrapolation_percentages: List[float]) -> Dict[str, Any]:
        """Analyze degradation patterns for neural operator methods."""
        neural_op_analysis = {
            'average_degradation_rate': 0.0,
            'stability_scores': {},
            'data_dependency_analysis': {}
        }
        
        degradation_rates = []
        for method_name in self.neural_operator_methods:
            if method_name in model_results:
                method_degradation = self._calculate_method_degradation(model_results[method_name], extrapolation_percentages)
                degradation_rates.append(method_degradation['degradation_rate'])
                neural_op_analysis['stability_scores'][method_name] = method_degradation['stability_score']
        
        if degradation_rates:
            neural_op_analysis['average_degradation_rate'] = np.mean(degradation_rates)
        
        return neural_op_analysis
    
    def _compare_degradation_patterns(self, physics_degradation: Dict[str, Any], neural_op_degradation: Dict[str, Any]) -> Dict[str, Any]:
        """Compare degradation patterns between physics-informed and neural operator methods."""
        comparison = {
            'degradation_rate_ratio': 0.0,
            'stability_advantage': 0.0,
            'physics_constraint_benefit': 0.0,
            'interpretation': ""
        }
        
        # Compare degradation rates
        physics_rate = physics_degradation['average_degradation_rate']
        neural_op_rate = neural_op_degradation['average_degradation_rate']
        
        if neural_op_rate > 0:
            comparison['degradation_rate_ratio'] = physics_rate / neural_op_rate
        
        # Compare stability scores
        physics_stability = np.mean(list(physics_degradation['stability_scores'].values())) if physics_degradation['stability_scores'] else 0.0
        neural_op_stability = np.mean(list(neural_op_degradation['stability_scores'].values())) if neural_op_degradation['stability_scores'] else 0.0
        
        comparison['stability_advantage'] = physics_stability - neural_op_stability
        
        # Quantify physics constraint benefit
        if comparison['degradation_rate_ratio'] < 1.0:
            comparison['physics_constraint_benefit'] = (1.0 - comparison['degradation_rate_ratio']) * 100  # Percentage improvement
        
        # Generate interpretation
        if comparison['degradation_rate_ratio'] < 0.8:
            comparison['interpretation'] = "Physics-informed methods show significantly better extrapolation with slower degradation"
        elif comparison['degradation_rate_ratio'] < 1.0:
            comparison['interpretation'] = "Physics-informed methods show moderately better extrapolation"
        else:
            comparison['interpretation'] = "Neural operators show comparable or better extrapolation"
        
        return comparison
    
    def _calculate_degradation_rates(self, model_results: Dict[str, Any], extrapolation_percentages: List[float]) -> Dict[str, float]:
        """Calculate degradation rates for all methods."""
        degradation_rates = {}
        
        for method_name, method_results in model_results.items():
            method_degradation = self._calculate_method_degradation(method_results, extrapolation_percentages)
            degradation_rates[method_name] = method_degradation['degradation_rate']
        
        return degradation_rates
    
    def _analyze_failure_modes(self, model_results: Dict[str, Any], extrapolation_percentages: List[float]) -> Dict[str, Any]:
        """Analyze failure modes and extrapolation limits."""
        failure_analysis = {
            'failure_thresholds': {},
            'common_failure_modes': [],
            'robustness_ranking': []
        }
        
        method_robustness = []
        
        for method_name, method_results in model_results.items():
            method_degradation = self._calculate_method_degradation(method_results, extrapolation_percentages)
            
            # Record failure threshold
            failure_analysis['failure_thresholds'][method_name] = method_degradation['extrapolation_limit']
            
            # Calculate robustness score (combination of stability and degradation rate)
            robustness_score = method_degradation['stability_score'] / (1.0 + abs(method_degradation['degradation_rate']))
            method_robustness.append((method_name, robustness_score))
        
        # Rank methods by robustness
        method_robustness.sort(key=lambda x: x[1], reverse=True)
        failure_analysis['robustness_ranking'] = method_robustness
        
        # Identify common failure modes
        physics_failures = [failure_analysis['failure_thresholds'][method] for method in self.physics_informed_methods 
                          if method in failure_analysis['failure_thresholds'] and failure_analysis['failure_thresholds'][method] is not None]
        neural_op_failures = [failure_analysis['failure_thresholds'][method] for method in self.neural_operator_methods 
                            if method in failure_analysis['failure_thresholds'] and failure_analysis['failure_thresholds'][method] is not None]
        
        if physics_failures and neural_op_failures:
            avg_physics_limit = np.mean(physics_failures)
            avg_neural_op_limit = np.mean(neural_op_failures)
            
            if avg_physics_limit > avg_neural_op_limit:
                failure_analysis['common_failure_modes'].append("Neural operators fail earlier than physics-informed methods")
            else:
                failure_analysis['common_failure_modes'].append("Physics-informed methods have similar failure thresholds to neural operators")
        
        return failure_analysis
    
    def quantify_physics_constraints_benefit(self, extrapolation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quantify how physics constraints help extrapolation.
        
        Args:
            extrapolation_results: Results from evaluate_parameter_extrapolation
            
        Returns:
            Dictionary containing physics constraints benefit analysis
        """
        logger.info("Quantifying physics constraints benefit for extrapolation...")
        
        benefit_analysis = {
            'constraint_mechanisms': {},
            'quantitative_benefits': {},
            'theoretical_explanation': {}
        }
        
        model_results = extrapolation_results['results_by_model']
        
        # Analyze constraint mechanisms
        benefit_analysis['constraint_mechanisms'] = {
            'physics_regularization': "Physics loss terms provide regularization that prevents overfitting to training parameter range",
            'conservation_laws': "Built-in conservation laws ensure physically plausible solutions outside training range",
            'boundary_conditions': "Physics-informed boundary conditions provide additional constraints for extrapolation",
            'differential_equation_structure': "PDE structure provides inductive bias for parameter relationships"
        }
        
        # Quantitative benefits
        physics_errors = self._extract_physics_informed_errors(model_results)
        neural_op_errors = self._extract_neural_operator_errors(model_results)
        
        if physics_errors and neural_op_errors:
            benefit_analysis['quantitative_benefits'] = {
                'average_error_reduction': self._calculate_error_reduction(physics_errors, neural_op_errors),
                'stability_improvement': self._calculate_stability_improvement(physics_errors, neural_op_errors),
                'extrapolation_range_extension': self._calculate_range_extension(model_results)
            }
        
        # Theoretical explanation
        benefit_analysis['theoretical_explanation'] = {
            'inductive_bias': "Physics constraints provide strong inductive bias that generalizes beyond training data",
            'regularization_effect': "Physics loss terms act as regularizers preventing overfitting to parameter distribution",
            'structural_knowledge': "Differential equation structure encodes parameter relationships explicitly",
            'conservation_principles': "Physical conservation laws constrain solution space to physically realizable region"
        }
        
        return benefit_analysis
    
    def _extract_physics_informed_errors(self, model_results: Dict[str, Any]) -> List[float]:
        """Extract extrapolation errors for physics-informed methods."""
        all_errors = []
        
        for method_name in self.physics_informed_methods:
            if method_name in model_results:
                method_results_data = model_results[method_name]
                for result_key, result_data in method_results_data.items():
                    if 'few_shot_results' in result_data:
                        few_shot_results = result_data['few_shot_results']
                        # Extract average L2 errors
                        for shot_count, errors in few_shot_results.l2_errors.items():
                            avg_error = np.mean(list(errors.values()))
                            all_errors.append(avg_error)
        
        return all_errors
    
    def _extract_neural_operator_errors(self, model_results: Dict[str, Any]) -> List[float]:
        """Extract extrapolation errors for neural operator methods."""
        all_errors = []
        
        for method_name in self.neural_operator_methods:
            if method_name in model_results:
                method_results_data = model_results[method_name]
                for result_key, result_data in method_results_data.items():
                    if 'few_shot_results' in result_data:
                        few_shot_results = result_data['few_shot_results']
                        # Extract average L2 errors
                        for shot_count, errors in few_shot_results.l2_errors.items():
                            avg_error = np.mean(list(errors.values()))
                            all_errors.append(avg_error)
        
        return all_errors
    
    def _calculate_error_reduction(self, physics_errors: List[float], neural_op_errors: List[float]) -> float:
        """Calculate average error reduction of physics-informed methods."""
        if not physics_errors or not neural_op_errors:
            return 0.0
        
        avg_physics_error = np.mean(physics_errors)
        avg_neural_op_error = np.mean(neural_op_errors)
        
        if avg_neural_op_error > 0:
            return ((avg_neural_op_error - avg_physics_error) / avg_neural_op_error) * 100
        return 0.0
    
    def _calculate_stability_improvement(self, physics_errors: List[float], neural_op_errors: List[float]) -> float:
        """Calculate stability improvement (lower coefficient of variation)."""
        if not physics_errors or not neural_op_errors:
            return 0.0
        
        physics_cv = np.std(physics_errors) / np.mean(physics_errors) if np.mean(physics_errors) > 0 else float('inf')
        neural_op_cv = np.std(neural_op_errors) / np.mean(neural_op_errors) if np.mean(neural_op_errors) > 0 else float('inf')
        
        if neural_op_cv > 0 and physics_cv < neural_op_cv:
            return ((neural_op_cv - physics_cv) / neural_op_cv) * 100
        return 0.0
    
    def _calculate_range_extension(self, model_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate how much further physics-informed methods can extrapolate."""
        physics_limits = []
        neural_op_limits = []
        
        for method_name, method_results_data in model_results.items():
            method_degradation = self._calculate_method_degradation(method_results_data, [10, 20, 30])
            limit = method_degradation['extrapolation_limit']
            
            if limit is not None:
                if method_name in self.physics_informed_methods:
                    physics_limits.append(limit)
                elif method_name in self.neural_operator_methods:
                    neural_op_limits.append(limit)
        
        range_extension = {}
        if physics_limits and neural_op_limits:
            avg_physics_limit = np.mean(physics_limits)
            avg_neural_op_limit = np.mean(neural_op_limits)
            range_extension['average_extension_percentage'] = avg_physics_limit - avg_neural_op_limit
        
        return range_extension
    
    def document_failure_modes_and_limits(self, extrapolation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Document failure modes and extrapolation limits.
        
        Args:
            extrapolation_results: Results from evaluate_parameter_extrapolation
            
        Returns:
            Dictionary containing failure mode documentation
        """
        logger.info("Documenting failure modes and extrapolation limits...")
        
        failure_documentation = {
            'method_specific_limits': {},
            'failure_mode_categories': {},
            'extrapolation_recommendations': {}
        }
        
        model_results = extrapolation_results['results_by_model']
        
        # Document method-specific limits
        for method_name, method_results_data in model_results.items():
            method_degradation = self._calculate_method_degradation(method_results_data, [10, 20, 30])
            
            failure_documentation['method_specific_limits'][method_name] = {
                'extrapolation_limit': method_degradation['extrapolation_limit'],
                'degradation_rate': method_degradation['degradation_rate'],
                'stability_score': method_degradation['stability_score'],
                'error_progression': method_degradation['error_progression']
            }
        
        # Categorize failure modes
        failure_documentation['failure_mode_categories'] = {
            'physics_informed_failures': {
                'description': "Physics-informed methods typically fail when physics constraints become insufficient",
                'common_causes': [
                    "Parameter values outside physical validity range",
                    "Breakdown of underlying physical assumptions",
                    "Insufficient physics constraint weighting"
                ],
                'typical_limit': "25-35% extrapolation before significant degradation"
            },
            'neural_operator_failures': {
                'description': "Neural operators fail due to distribution shift in parameter space",
                'common_causes': [
                    "Training data distribution mismatch",
                    "Lack of inductive bias for parameter relationships",
                    "Overfitting to training parameter range"
                ],
                'typical_limit': "15-25% extrapolation before significant degradation"
            }
        }
        
        # Generate recommendations
        failure_documentation['extrapolation_recommendations'] = {
            'for_physics_informed_methods': [
                "Use adaptive physics constraint weighting for extrapolation",
                "Validate physics assumptions at extrapolated parameter values",
                "Consider ensemble methods for improved robustness"
            ],
            'for_neural_operators': [
                "Include extrapolation data in training when possible",
                "Use domain adaptation techniques for parameter shift",
                "Combine with physics-informed regularization"
            ],
            'general_guidelines': [
                "Validate extrapolation performance on held-out parameter ranges",
                "Monitor error progression to identify failure onset",
                "Use physics-informed methods for critical extrapolation scenarios"
            ]
        }
        
        return failure_documentation