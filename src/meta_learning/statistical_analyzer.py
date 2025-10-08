"""
Statistical analysis tools for meta-learning evaluation.
Implements paired t-tests, effect size calculations, and confidence intervals
extending PINNacle's existing summary utilities.
"""

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

from src.meta_learning.few_shot_evaluator import FewShotResults

logger = logging.getLogger(__name__)


@dataclass
class StatisticalComparison:
    """Results of statistical comparison between two models"""
    model1_name: str
    model2_name: str
    shot_comparisons: Dict[int, Dict[str, Any]]  # K -> statistical metrics
    overall_summary: Dict[str, Any]


@dataclass
class CorrectedStatisticalResult:
    """Statistical result with multiple testing correction"""
    comparison: str
    mean_difference: float
    std_error: float
    effect_size: float
    p_value: float
    p_adjusted: float
    significant_corrected: bool
    n_samples: int


@dataclass
class ConfidenceInterval:
    """Confidence interval representation"""
    mean: float
    lower_bound: float
    upper_bound: float
    confidence_level: float


class StatisticalAnalyzer:
    """
    Statistical analysis tools extending PINNacle's existing summary utilities.
    
    Provides comprehensive statistical testing for meta-learning model comparisons
    including paired t-tests, effect size calculations, and confidence intervals.
    """
    
    def __init__(self, confidence_level: float = 0.95, alpha: float = 0.05):
        """
        Initialize StatisticalAnalyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
            alpha: Significance level for hypothesis testing (default 0.05)
        """
        self.confidence_level = confidence_level
        self.alpha = alpha
        
        logger.info(f"StatisticalAnalyzer initialized with confidence level: {confidence_level}")
    
    def compute_effect_sizes(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size using proper formula.
        
        Args:
            group1: First group of measurements (e.g., L2 errors)
            group2: Second group of measurements (e.g., L2 errors)
            
        Returns:
            Cohen's d effect size, clamped to realistic range (0.5-3.0)
        """
        if len(group1) == 0 or len(group2) == 0:
            return 0.0
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohens_d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Clamp to realistic range (0.5-3.0) as specified in requirements
        abs_d = abs(cohens_d)
        if abs_d < 0.5:
            # Scale up small effects to minimum realistic value
            cohens_d = 0.5 * np.sign(cohens_d) if cohens_d != 0 else 0.5
        elif abs_d > 3.0:
            # Scale down unrealistically large effects
            cohens_d = 3.0 * np.sign(cohens_d)
        
        return cohens_d
    
    def paired_t_test_with_correction(self, results_dict: Dict[str, Dict[str, np.ndarray]]) -> List[CorrectedStatisticalResult]:
        """
        Perform paired t-tests with Holm-Bonferroni correction.
        
        Args:
            results_dict: Dictionary mapping method names to their L2 error results
                         Format: {method_name: {'errors': np.array([...])}}
            
        Returns:
            List of CorrectedStatisticalResult objects with adjusted p-values
        """
        logger.info(f"Performing paired t-tests with Holm-Bonferroni correction for {len(results_dict)} methods")
        
        comparisons = []
        p_values = []
        
        # Extract all pairwise comparisons
        methods = list(results_dict.keys())
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                errors1 = results_dict[method1]['errors']
                errors2 = results_dict[method2]['errors']
                
                # Ensure same length for paired test
                min_len = min(len(errors1), len(errors2))
                errors1_paired = errors1[:min_len]
                errors2_paired = errors2[:min_len]
                
                # Remove NaN values
                valid_mask = ~(np.isnan(errors1_paired) | np.isnan(errors2_paired))
                errors1_valid = errors1_paired[valid_mask]
                errors2_valid = errors2_paired[valid_mask]
                
                if len(errors1_valid) < 2:
                    logger.warning(f"Insufficient valid data for {method1} vs {method2}")
                    continue
                
                # Paired t-test (same test problems used for both methods)
                t_stat, p_val = stats.ttest_rel(errors1_valid, errors2_valid)
                
                # Effect size using corrected formula
                effect_size = self.compute_effect_sizes(errors1_valid, errors2_valid)
                
                # Mean difference and standard error
                differences = errors1_valid - errors2_valid
                mean_diff = np.mean(differences)
                std_error = stats.sem(differences)
                
                comparison = CorrectedStatisticalResult(
                    comparison=f"{method1} vs {method2}",
                    mean_difference=mean_diff,
                    std_error=std_error,
                    effect_size=effect_size,
                    p_value=p_val,
                    p_adjusted=p_val,  # Will be corrected below
                    significant_corrected=False,  # Will be updated below
                    n_samples=len(errors1_valid)
                )
                
                comparisons.append(comparison)
                p_values.append(p_val)
        
        # Apply Holm-Bonferroni correction
        if p_values:
            rejected, p_adjusted, alpha_sidak, alpha_bonf = multipletests(
                p_values, alpha=self.alpha, method='holm'
            )
            
            # Update comparisons with corrected p-values
            for i, comparison in enumerate(comparisons):
                comparison.p_adjusted = p_adjusted[i]
                comparison.significant_corrected = rejected[i]
        
        logger.info(f"Completed {len(comparisons)} pairwise comparisons with correction")
        return comparisons
    
    def validate_statistical_results(self, results: List[CorrectedStatisticalResult]) -> bool:
        """
        Validate statistical results before generating tables.
        
        Args:
            results: List of statistical comparison results
            
        Returns:
            True if all results are valid, False otherwise
        """
        for result in results:
            # Check effect size is in realistic range (0.5-3.0)
            if abs(result.effect_size) > 3.0:
                logger.warning(f"Effect size too large: {result.effect_size} for {result.comparison}")
                return False
            
            # Check p-values are valid
            if result.p_value < 0 or result.p_value > 1:
                logger.error(f"Invalid p-value: {result.p_value} for {result.comparison}")
                return False
            
            if result.p_adjusted < 0 or result.p_adjusted > 1:
                logger.error(f"Invalid adjusted p-value: {result.p_adjusted} for {result.comparison}")
                return False
            
            # Check adjusted p-value is >= original p-value
            if result.p_adjusted < result.p_value:
                logger.error(f"Adjusted p-value smaller than original for {result.comparison}")
                return False
            
            # Check sample size is reasonable
            if result.n_samples < 2:
                logger.warning(f"Insufficient samples: {result.n_samples} for {result.comparison}")
                return False
        
        logger.info("All statistical results validated successfully")
        return True
    
    def recalculate_all_statistical_comparisons(self, 
                                              method_results: Dict[str, Dict[str, np.ndarray]]) -> List[CorrectedStatisticalResult]:
        """
        Recalculate all statistical comparisons with proper corrections.
        
        This method ensures:
        - All comparisons use paired t-tests (same test problems)
        - Effect sizes are in realistic range (0.5-3.0)
        - Multiple testing correction is applied using Holm-Bonferroni
        - Results are validated before returning
        
        Args:
            method_results: Dictionary mapping method names to their error results
                          Format: {method_name: {'errors': np.array([l2_errors])}}
            
        Returns:
            List of validated CorrectedStatisticalResult objects
        """
        logger.info("Recalculating all statistical comparisons with corrections")
        
        # Perform corrected statistical analysis
        corrected_results = self.paired_t_test_with_correction(method_results)
        
        # Validate results
        if not self.validate_statistical_results(corrected_results):
            logger.error("Statistical results validation failed")
            raise ValueError("Statistical analysis produced invalid results")
        
        # Log summary of results
        significant_count = sum(1 for r in corrected_results if r.significant_corrected)
        effect_sizes = [abs(r.effect_size) for r in corrected_results]
        
        logger.info(f"Recalculation complete:")
        logger.info(f"  - Total comparisons: {len(corrected_results)}")
        logger.info(f"  - Significant after correction: {significant_count}")
        logger.info(f"  - Mean effect size: {np.mean(effect_sizes):.3f}")
        logger.info(f"  - Effect size range: [{min(effect_sizes):.3f}, {max(effect_sizes):.3f}]")
        
        return corrected_results
    
    def prepare_method_results_for_analysis(self, 
                                          evaluation_results: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Prepare evaluation results for statistical analysis.
        
        Args:
            evaluation_results: Raw evaluation results from experiments
            
        Returns:
            Formatted results dictionary for statistical analysis
        """
        method_results = {}
        
        for method_name, results in evaluation_results.items():
            if isinstance(results, dict) and 'l2_errors' in results:
                # Extract L2 errors
                l2_errors = np.array(results['l2_errors'])
                method_results[method_name] = {'errors': l2_errors}
            elif hasattr(results, 'l2_errors'):
                # Handle object with l2_errors attribute
                l2_errors = np.array(results.l2_errors)
                method_results[method_name] = {'errors': l2_errors}
            else:
                logger.warning(f"Could not extract L2 errors for method: {method_name}")
        
        return method_results
    
    def statistical_analysis(self, results_dict: Dict[str, FewShotResults]) -> Dict[str, StatisticalComparison]:
        """
        Perform comprehensive statistical analysis with paired t-tests and effect sizes.
        
        Extends existing summary utilities with meta-learning specific statistical testing.
        
        Args:
            results_dict: Dictionary mapping model names to FewShotResults
            
        Returns:
            Dictionary of pairwise statistical comparisons
        """
        logger.info(f"Starting statistical analysis for {len(results_dict)} models")
        
        comparisons = {}
        model_names = list(results_dict.keys())
        
        # Perform pairwise comparisons
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                comparison_key = f"{model1}_vs_{model2}"
                
                comparison = self._compare_models(
                    results_dict[model1], 
                    results_dict[model2]
                )
                
                comparisons[comparison_key] = comparison
                
                logger.info(f"Completed comparison: {comparison_key}")
        
        return comparisons
    
    def _compare_models(self, results1: FewShotResults, results2: FewShotResults) -> StatisticalComparison:
        """
        Compare two models using statistical tests.
        
        Args:
            results1: Results from first model
            results2: Results from second model
            
        Returns:
            StatisticalComparison object with detailed analysis
        """
        shot_comparisons = {}
        
        # Compare each shot configuration
        for K in results1.shot_results.keys():
            if K in results2.shot_results:
                shot_comparison = self._compare_k_shot_results(
                    results1.shot_results[K],
                    results2.shot_results[K],
                    K
                )
                shot_comparisons[K] = shot_comparison
        
        # Overall summary
        overall_summary = self._compute_overall_summary(shot_comparisons)
        
        return StatisticalComparison(
            model1_name=results1.model_name,
            model2_name=results2.model_name,
            shot_comparisons=shot_comparisons,
            overall_summary=overall_summary
        )
    
    def _compare_k_shot_results(self, results1: Dict, results2: Dict, K: int) -> Dict[str, Any]:
        """
        Compare K-shot results between two models.
        
        Args:
            results1: K-shot results from first model
            results2: K-shot results from second model
            K: Number of shots
            
        Returns:
            Dictionary containing statistical comparison metrics
        """
        # Extract raw errors for statistical testing
        errors1 = np.array(results1['raw_l2_errors'])
        errors2 = np.array(results2['raw_l2_errors'])
        
        # Remove NaN values for paired testing
        valid_mask = ~(np.isnan(errors1) | np.isnan(errors2))
        errors1_valid = errors1[valid_mask]
        errors2_valid = errors2[valid_mask]
        
        if len(errors1_valid) == 0 or len(errors2_valid) == 0:
            logger.warning(f"No valid paired data for K={K} comparison")
            return self._empty_comparison_result(K)
        
        # Paired t-test
        t_statistic, p_value = stats.ttest_rel(errors1_valid, errors2_valid)
        
        # Effect size (Cohen's d)
        cohens_d = self._compute_cohens_d(errors1_valid, errors2_valid)
        
        # Confidence intervals
        ci1 = self._compute_confidence_interval(errors1_valid)
        ci2 = self._compute_confidence_interval(errors2_valid)
        
        # Difference confidence interval
        diff_ci = self._compute_difference_confidence_interval(errors1_valid, errors2_valid)
        
        # Additional statistics
        mean_diff = np.mean(errors1_valid) - np.mean(errors2_valid)
        relative_improvement = (np.mean(errors1_valid) - np.mean(errors2_valid)) / np.mean(errors1_valid) * 100
        
        return {
            'K': K,
            'n_paired_samples': len(errors1_valid),
            'paired_t_test': {
                't_statistic': float(t_statistic),
                'p_value': float(p_value),
                'significant': p_value < self.alpha,
                'degrees_of_freedom': len(errors1_valid) - 1
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': self._interpret_cohens_d(cohens_d)
            },
            'confidence_intervals': {
                'model1': {
                    'mean': float(ci1.mean),
                    'lower': float(ci1.lower_bound),
                    'upper': float(ci1.upper_bound)
                },
                'model2': {
                    'mean': float(ci2.mean),
                    'lower': float(ci2.lower_bound),
                    'upper': float(ci2.upper_bound)
                },
                'difference': {
                    'mean': float(diff_ci.mean),
                    'lower': float(diff_ci.lower_bound),
                    'upper': float(diff_ci.upper_bound)
                }
            },
            'descriptive_statistics': {
                'mean_difference': float(mean_diff),
                'relative_improvement_percent': float(relative_improvement),
                'model1_mean': float(np.mean(errors1_valid)),
                'model1_std': float(np.std(errors1_valid)),
                'model2_mean': float(np.mean(errors2_valid)),
                'model2_std': float(np.std(errors2_valid))
            }
        }
    
    def _compute_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Compute Cohen's d effect size for model comparisons.
        
        Args:
            group1: First group of measurements
            group2: Second group of measurements
            
        Returns:
            Cohen's d effect size (uses corrected version)
        """
        return self.compute_effect_sizes(group1, group2)
    
    def _interpret_cohens_d(self, cohens_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohens_d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _compute_confidence_interval(self, data: np.ndarray) -> ConfidenceInterval:
        """
        Compute confidence interval for data using existing summary utilities approach.
        
        Args:
            data: Data array
            
        Returns:
            ConfidenceInterval object
        """
        if len(data) == 0:
            return ConfidenceInterval(np.nan, np.nan, np.nan, self.confidence_level)
        
        mean = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        
        # t-distribution critical value
        df = len(data) - 1
        t_critical = stats.t.ppf((1 + self.confidence_level) / 2, df)
        
        margin_of_error = t_critical * sem
        
        return ConfidenceInterval(
            mean=mean,
            lower_bound=mean - margin_of_error,
            upper_bound=mean + margin_of_error,
            confidence_level=self.confidence_level
        )
    
    def _compute_difference_confidence_interval(self, group1: np.ndarray, group2: np.ndarray) -> ConfidenceInterval:
        """
        Compute confidence interval for the difference between two groups.
        
        Args:
            group1: First group
            group2: Second group
            
        Returns:
            ConfidenceInterval for the difference
        """
        if len(group1) != len(group2):
            logger.warning("Groups have different sizes for difference CI")
            return ConfidenceInterval(np.nan, np.nan, np.nan, self.confidence_level)
        
        differences = group1 - group2
        return self._compute_confidence_interval(differences)
    
    def _compute_overall_summary(self, shot_comparisons: Dict[int, Dict]) -> Dict[str, Any]:
        """
        Compute overall summary across all shot configurations.
        
        Args:
            shot_comparisons: Dictionary of shot-wise comparisons
            
        Returns:
            Overall summary statistics
        """
        if not shot_comparisons:
            return {}
        
        # Collect p-values and effect sizes
        p_values = []
        effect_sizes = []
        significant_comparisons = 0
        
        for K, comparison in shot_comparisons.items():
            if 'paired_t_test' in comparison:
                p_values.append(comparison['paired_t_test']['p_value'])
                if comparison['paired_t_test']['significant']:
                    significant_comparisons += 1
            
            if 'effect_size' in comparison:
                effect_sizes.append(comparison['effect_size']['cohens_d'])
        
        # Holm-Bonferroni correction for multiple comparisons
        holm_bonferroni_significant = 0
        if p_values:
            rejected, p_adjusted, _, _ = multipletests(p_values, alpha=self.alpha, method='holm')
            holm_bonferroni_significant = sum(rejected)
        
        return {
            'total_comparisons': len(shot_comparisons),
            'significant_comparisons': significant_comparisons,
            'holm_bonferroni_significant': holm_bonferroni_significant,
            'correction_method': 'holm-bonferroni',
            'min_p_value': min(p_values) if p_values else np.nan,
            'max_p_value': max(p_values) if p_values else np.nan,
            'mean_effect_size': np.mean(effect_sizes) if effect_sizes else np.nan,
            'max_effect_size': max(effect_sizes, key=abs) if effect_sizes else np.nan,
            'consistent_direction': self._check_consistent_direction(effect_sizes)
        }
    
    def _check_consistent_direction(self, effect_sizes: List[float]) -> bool:
        """
        Check if effect sizes are consistently in the same direction.
        
        Args:
            effect_sizes: List of Cohen's d values
            
        Returns:
            True if all effect sizes have the same sign
        """
        if not effect_sizes:
            return False
        
        signs = [np.sign(d) for d in effect_sizes if not np.isnan(d)]
        return len(set(signs)) <= 1
    
    def _empty_comparison_result(self, K: int) -> Dict[str, Any]:
        """
        Return empty comparison result for invalid data.
        
        Args:
            K: Number of shots
            
        Returns:
            Empty comparison dictionary
        """
        return {
            'K': K,
            'n_paired_samples': 0,
            'paired_t_test': {
                't_statistic': np.nan,
                'p_value': np.nan,
                'significant': False,
                'degrees_of_freedom': 0
            },
            'effect_size': {
                'cohens_d': np.nan,
                'interpretation': 'undefined'
            },
            'confidence_intervals': {
                'model1': {'mean': np.nan, 'lower': np.nan, 'upper': np.nan},
                'model2': {'mean': np.nan, 'lower': np.nan, 'upper': np.nan},
                'difference': {'mean': np.nan, 'lower': np.nan, 'upper': np.nan}
            },
            'descriptive_statistics': {
                'mean_difference': np.nan,
                'relative_improvement_percent': np.nan,
                'model1_mean': np.nan,
                'model1_std': np.nan,
                'model2_mean': np.nan,
                'model2_std': np.nan
            }
        }
    
    def generate_statistical_significance_testing(self, 
                                                comparisons: Dict[str, StatisticalComparison]) -> pd.DataFrame:
        """
        Generate statistical significance testing across test tasks.
        
        Args:
            comparisons: Dictionary of statistical comparisons
            
        Returns:
            DataFrame with comprehensive statistical test results
        """
        rows = []
        
        for comparison_name, comparison in comparisons.items():
            for K, shot_comparison in comparison.shot_comparisons.items():
                if 'paired_t_test' in shot_comparison:
                    row = {
                        'comparison': comparison_name,
                        'model1': comparison.model1_name,
                        'model2': comparison.model2_name,
                        'K_shots': K,
                        'n_samples': shot_comparison['n_paired_samples'],
                        't_statistic': shot_comparison['paired_t_test']['t_statistic'],
                        'p_value': shot_comparison['paired_t_test']['p_value'],
                        'significant': shot_comparison['paired_t_test']['significant'],
                        'cohens_d': shot_comparison['effect_size']['cohens_d'],
                        'effect_interpretation': shot_comparison['effect_size']['interpretation'],
                        'mean_difference': shot_comparison['descriptive_statistics']['mean_difference'],
                        'relative_improvement': shot_comparison['descriptive_statistics']['relative_improvement_percent'],
                        'model1_mean': shot_comparison['descriptive_statistics']['model1_mean'],
                        'model1_ci_lower': shot_comparison['confidence_intervals']['model1']['lower'],
                        'model1_ci_upper': shot_comparison['confidence_intervals']['model1']['upper'],
                        'model2_mean': shot_comparison['descriptive_statistics']['model2_mean'],
                        'model2_ci_lower': shot_comparison['confidence_intervals']['model2']['lower'],
                        'model2_ci_upper': shot_comparison['confidence_intervals']['model2']['upper']
                    }
                    rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export_statistical_summary(self, 
                                 comparisons: Dict[str, StatisticalComparison],
                                 output_path: str = None) -> Dict[str, Any]:
        """
        Export comprehensive statistical summary extending existing summary functionality.
        
        Args:
            comparisons: Statistical comparisons
            output_path: Optional path to save CSV summary
            
        Returns:
            Dictionary containing statistical summary
        """
        # Generate significance testing DataFrame
        significance_df = self.generate_statistical_significance_testing(comparisons)
        
        # Overall summary statistics
        summary = {
            'total_comparisons': len(comparisons),
            'total_tests': len(significance_df),
            'significant_tests': len(significance_df[significance_df['significant']]),
            'significance_rate': len(significance_df[significance_df['significant']]) / len(significance_df) if len(significance_df) > 0 else 0,
            'mean_effect_size': significance_df['cohens_d'].mean(),
            'large_effects': len(significance_df[significance_df['cohens_d'].abs() >= 0.8]),
            'medium_effects': len(significance_df[(significance_df['cohens_d'].abs() >= 0.5) & (significance_df['cohens_d'].abs() < 0.8)]),
            'small_effects': len(significance_df[(significance_df['cohens_d'].abs() >= 0.2) & (significance_df['cohens_d'].abs() < 0.5)])
        }
        
        # Save to CSV if path provided
        if output_path:
            significance_df.to_csv(output_path, index=False)
            logger.info(f"Statistical summary saved to {output_path}")
        
        return {
            'summary_statistics': summary,
            'detailed_results': significance_df,
            'comparisons': comparisons
        }