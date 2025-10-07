"""
Comprehensive evaluation framework integrating all meta-learning evaluation components.
Provides a unified interface for few-shot evaluation, statistical analysis, 
computational trade-off analysis, and visualization reporting.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from src.meta_learning.few_shot_evaluator import FewShotEvaluator, FewShotResults
from src.meta_learning.statistical_analyzer import StatisticalAnalyzer, StatisticalComparison
from src.meta_learning.computational_analyzer import ComputationalAnalyzer
from src.meta_learning.visualization_reporter import VisualizationReporter
from src.meta_learning.task import Task

logger = logging.getLogger(__name__)


class MetaLearningEvaluationFramework:
    """
    Comprehensive evaluation framework extending PINNacle's evaluation utilities.
    
    Integrates few-shot evaluation, statistical analysis, computational trade-off analysis,
    and visualization reporting into a unified framework for meta-learning research.
    """
    
    def __init__(self,
                 output_dir: str = "meta_learning_evaluation",
                 evaluation_shots: List[int] = None,
                 confidence_level: float = 0.95,
                 target_accuracy: float = 0.01,
                 device: str = 'cuda',
                 verbose: bool = True):
        """
        Initialize comprehensive evaluation framework.
        
        Args:
            output_dir: Directory for saving all evaluation results
            evaluation_shots: List of K values for K-shot evaluation
            confidence_level: Confidence level for statistical analysis
            target_accuracy: Target accuracy for computational analysis
            device: Computing device
            verbose: Whether to print progress information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize component evaluators
        self.few_shot_evaluator = FewShotEvaluator(
            evaluation_shots=evaluation_shots,
            device=device,
            verbose=verbose
        )
        
        self.statistical_analyzer = StatisticalAnalyzer(
            confidence_level=confidence_level
        )
        
        self.computational_analyzer = ComputationalAnalyzer(
            device=device,
            target_accuracy=target_accuracy
        )
        
        self.visualization_reporter = VisualizationReporter(
            output_dir=str(self.output_dir),
            style="publication"
        )
        
        self.verbose = verbose
        
        logger.info(f"MetaLearningEvaluationFramework initialized: {output_dir}")
    
    def evaluate_comprehensive(self,
                             models: Dict[str, Any],
                             test_tasks: List[Task],
                             training_functions: Dict[str, callable] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of all models.
        
        Args:
            models: Dictionary mapping model names to model instances
            test_tasks: List of test tasks for evaluation
            training_functions: Optional training functions for computational analysis
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Starting comprehensive evaluation of {len(models)} models on {len(test_tasks)} tasks")
        
        results = {
            'few_shot_results': {},
            'statistical_comparisons': {},
            'computational_analysis': {},
            'visualizations': {},
            'summary': {}
        }
        
        # 1. Few-shot evaluation
        if self.verbose:
            print("Phase 1: Few-shot performance evaluation...")
        
        for model_name, model in models.items():
            few_shot_result = self.few_shot_evaluator.evaluate_few_shot_performance(
                model, test_tasks, model_name
            )
            results['few_shot_results'][model_name] = few_shot_result
        
        # 2. Statistical analysis
        if self.verbose:
            print("Phase 2: Statistical analysis...")
        
        results['statistical_comparisons'] = self.statistical_analyzer.statistical_analysis(
            results['few_shot_results']
        )
        
        # 3. Computational analysis (if training functions provided)
        if training_functions:
            if self.verbose:
                print("Phase 3: Computational trade-off analysis...")
            
            for model_name, model in models.items():
                if model_name in training_functions:
                    # Measure training time
                    self.computational_analyzer.measure_training_time(
                        model, training_functions[model_name], model_name
                    )
                    
                    # Measure adaptation speed on subset of tasks
                    sample_tasks = test_tasks[:min(10, len(test_tasks))]  # Use first 10 tasks
                    for task in sample_tasks:
                        self.computational_analyzer.measure_adaptation_speed(
                            model, task, model_name
                        )
            
            results['computational_analysis'] = self.computational_analyzer.generate_computational_report(
                list(models.keys())
            )
        
        # 4. Generate visualizations
        if self.verbose:
            print("Phase 4: Generating visualizations and reports...")
        
        # Performance curves
        results['visualizations']['performance_curves'] = self.visualization_reporter.plot_few_shot_performance_curves(
            results['few_shot_results']
        )
        
        # Comparison matrix
        results['visualizations']['comparison_matrix'] = self.visualization_reporter.create_performance_comparison_matrix(
            results['few_shot_results']
        )
        
        # Statistical significance plots
        if results['statistical_comparisons']:
            results['visualizations']['statistical_plots'] = self.visualization_reporter.plot_statistical_significance(
                results['statistical_comparisons']
            )
        
        # Computational trade-off plots
        if results['computational_analysis']:
            results['visualizations']['computational_plots'] = self.visualization_reporter.plot_computational_tradeoffs(
                results['computational_analysis']
            )
        
        # 5. Generate comprehensive report
        report_path = self.visualization_reporter.generate_automated_report(
            results['few_shot_results'],
            results['statistical_comparisons'],
            results['computational_analysis']
        )
        results['visualizations']['report'] = report_path
        
        # 6. Generate summary
        results['summary'] = self._generate_evaluation_summary(results)
        
        # Save complete results
        self._save_results(results)
        
        if self.verbose:
            print(f"Comprehensive evaluation completed. Results saved to: {self.output_dir}")
        
        return results
    
    def evaluate_model_comparison(self,
                                model1: Any,
                                model2: Any,
                                test_tasks: List[Task],
                                model1_name: str = "Model1",
                                model2_name: str = "Model2") -> Dict[str, Any]:
        """
        Focused comparison between two models.
        
        Args:
            model1: First model to compare
            model2: Second model to compare
            test_tasks: Test tasks for evaluation
            model1_name: Name for first model
            model2_name: Name for second model
            
        Returns:
            Dictionary containing comparison results
        """
        logger.info(f"Comparing {model1_name} vs {model2_name}")
        
        models = {model1_name: model1, model2_name: model2}
        
        # Evaluate both models
        few_shot_results = {}
        for name, model in models.items():
            few_shot_results[name] = self.few_shot_evaluator.evaluate_few_shot_performance(
                model, test_tasks, name
            )
        
        # Statistical comparison
        statistical_comparisons = self.statistical_analyzer.statistical_analysis(few_shot_results)
        
        # Generate comparison visualization
        comparison_plot = self.visualization_reporter.plot_few_shot_performance_curves(
            few_shot_results, save_name=f"{model1_name}_vs_{model2_name}_comparison"
        )
        
        # Statistical significance plot
        significance_plot = self.visualization_reporter.plot_statistical_significance(
            statistical_comparisons, save_name=f"{model1_name}_vs_{model2_name}_significance"
        )
        
        return {
            'few_shot_results': few_shot_results,
            'statistical_comparisons': statistical_comparisons,
            'visualizations': {
                'comparison_plot': comparison_plot,
                'significance_plot': significance_plot
            },
            'summary': self._generate_comparison_summary(few_shot_results, statistical_comparisons)
        }
    
    def benchmark_meta_learning_vs_standard(self,
                                          meta_models: Dict[str, Any],
                                          standard_models: Dict[str, Any],
                                          test_tasks: List[Task]) -> Dict[str, Any]:
        """
        Benchmark meta-learning approaches against standard baselines.
        
        Args:
            meta_models: Dictionary of meta-learning models
            standard_models: Dictionary of standard baseline models
            test_tasks: Test tasks for evaluation
            
        Returns:
            Dictionary containing benchmark results
        """
        logger.info(f"Benchmarking {len(meta_models)} meta-learning vs {len(standard_models)} standard models")
        
        all_models = {**meta_models, **standard_models}
        
        # Comprehensive evaluation
        results = self.evaluate_comprehensive(all_models, test_tasks)
        
        # Additional meta-learning specific analysis
        meta_vs_standard_analysis = self._analyze_meta_vs_standard(
            results['few_shot_results'], 
            list(meta_models.keys()), 
            list(standard_models.keys())
        )
        
        results['meta_vs_standard_analysis'] = meta_vs_standard_analysis
        
        return results
    
    def _generate_evaluation_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary."""
        summary = {
            'models_evaluated': len(results['few_shot_results']),
            'statistical_comparisons': len(results['statistical_comparisons']),
            'best_performing_model': None,
            'most_significant_improvement': None,
            'computational_efficiency_leader': None
        }
        
        # Find best performing model (lowest mean L2 error across all shots)
        if results['few_shot_results']:
            model_scores = {}
            for model_name, model_results in results['few_shot_results'].items():
                scores = []
                for K, shot_results in model_results.shot_results.items():
                    if 'mean_l2_error' in shot_results:
                        scores.append(shot_results['mean_l2_error'])
                
                if scores:
                    model_scores[model_name] = sum(scores) / len(scores)
            
            if model_scores:
                summary['best_performing_model'] = min(model_scores, key=model_scores.get)
                summary['performance_scores'] = model_scores
        
        # Find most significant improvement
        if results['statistical_comparisons']:
            max_effect_size = 0
            best_comparison = None
            
            for comp_name, comparison in results['statistical_comparisons'].items():
                for K, shot_comp in comparison.shot_comparisons.items():
                    if 'effect_size' in shot_comp:
                        effect_size = abs(shot_comp['effect_size']['cohens_d'])
                        if effect_size > max_effect_size:
                            max_effect_size = effect_size
                            best_comparison = f"{comp_name} (K={K})"
            
            summary['most_significant_improvement'] = best_comparison
            summary['max_effect_size'] = max_effect_size
        
        # Find computational efficiency leader
        if results['computational_analysis'] and 'adaptation_performance' in results['computational_analysis']:
            adapt_perf = results['computational_analysis']['adaptation_performance']
            if adapt_perf:
                fastest_model = min(adapt_perf, key=lambda m: adapt_perf[m]['mean_adaptation_time'])
                summary['computational_efficiency_leader'] = fastest_model
        
        return summary
    
    def _generate_comparison_summary(self, 
                                   few_shot_results: Dict[str, FewShotResults],
                                   statistical_comparisons: Dict[str, StatisticalComparison]) -> Dict[str, Any]:
        """Generate summary for two-model comparison."""
        model_names = list(few_shot_results.keys())
        
        summary = {
            'models': model_names,
            'winner': None,
            'significant_differences': 0,
            'effect_sizes': []
        }
        
        if len(model_names) == 2:
            # Determine winner based on average performance
            model1_scores = []
            model2_scores = []
            
            for K in few_shot_results[model_names[0]].shot_results.keys():
                if K in few_shot_results[model_names[1]].shot_results:
                    score1 = few_shot_results[model_names[0]].shot_results[K].get('mean_l2_error', float('inf'))
                    score2 = few_shot_results[model_names[1]].shot_results[K].get('mean_l2_error', float('inf'))
                    model1_scores.append(score1)
                    model2_scores.append(score2)
            
            if model1_scores and model2_scores:
                avg1 = sum(model1_scores) / len(model1_scores)
                avg2 = sum(model2_scores) / len(model2_scores)
                summary['winner'] = model_names[0] if avg1 < avg2 else model_names[1]
        
        # Count significant differences
        for comparison in statistical_comparisons.values():
            for shot_comp in comparison.shot_comparisons.values():
                if shot_comp.get('paired_t_test', {}).get('significant', False):
                    summary['significant_differences'] += 1
                
                if 'effect_size' in shot_comp:
                    summary['effect_sizes'].append(shot_comp['effect_size']['cohens_d'])
        
        return summary
    
    def _analyze_meta_vs_standard(self,
                                few_shot_results: Dict[str, FewShotResults],
                                meta_model_names: List[str],
                                standard_model_names: List[str]) -> Dict[str, Any]:
        """Analyze meta-learning vs standard model performance."""
        analysis = {
            'meta_learning_advantage': {},
            'adaptation_efficiency': {},
            'data_efficiency': {}
        }
        
        # Calculate average performance for each group
        meta_performances = {}
        standard_performances = {}
        
        for K in [1, 5, 10, 25]:
            meta_errors = []
            standard_errors = []
            
            for model_name in meta_model_names:
                if model_name in few_shot_results and K in few_shot_results[model_name].shot_results:
                    error = few_shot_results[model_name].shot_results[K].get('mean_l2_error')
                    if error is not None:
                        meta_errors.append(error)
            
            for model_name in standard_model_names:
                if model_name in few_shot_results and K in few_shot_results[model_name].shot_results:
                    error = few_shot_results[model_name].shot_results[K].get('mean_l2_error')
                    if error is not None:
                        standard_errors.append(error)
            
            if meta_errors and standard_errors:
                meta_avg = sum(meta_errors) / len(meta_errors)
                standard_avg = sum(standard_errors) / len(standard_errors)
                
                meta_performances[K] = meta_avg
                standard_performances[K] = standard_avg
                
                # Calculate relative improvement
                improvement = (standard_avg - meta_avg) / standard_avg * 100
                analysis['meta_learning_advantage'][K] = improvement
        
        analysis['meta_performances'] = meta_performances
        analysis['standard_performances'] = standard_performances
        
        return analysis
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files."""
        import json
        import pickle
        
        # Save JSON summary (for human readability)
        json_results = {
            'summary': results['summary'],
            'model_names': list(results['few_shot_results'].keys()),
            'evaluation_completed': True
        }
        
        with open(self.output_dir / 'evaluation_summary.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        # Save complete results (pickle for full data)
        with open(self.output_dir / 'complete_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        logger.info("Evaluation results saved to files")
    
    def load_results(self, results_path: str = None) -> Dict[str, Any]:
        """Load previously saved evaluation results."""
        if results_path is None:
            results_path = self.output_dir / 'complete_results.pkl'
        
        import pickle
        with open(results_path, 'rb') as f:
            results = pickle.load(f)
        
        logger.info(f"Evaluation results loaded from {results_path}")
        return results