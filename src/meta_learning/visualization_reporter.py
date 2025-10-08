"""
Visualization and reporting tools for meta-learning evaluation.
Extends existing plotting utilities for few-shot performance curves,
publication-ready figures, and automated report generation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import os
import logging
from pathlib import Path

try:
    from src.utils.plot import plot_lines, plot_heatmap
except ImportError:
    # Fallback if plot utilities not available
    plot_lines = None
    plot_heatmap = None

try:
    from src.meta_learning.few_shot_evaluator import FewShotResults
    from src.meta_learning.statistical_analyzer import StatisticalComparison, CorrectedStatisticalResult
    from src.meta_learning.computational_analyzer import ComputationalAnalyzer
    from src.utils.profiling import TimingProfiler, MethodTimingResults
except ImportError:
    # Define minimal classes for standalone usage
    class FewShotResults:
        def __init__(self):
            self.shot_results = {}
    
    class StatisticalComparison:
        def __init__(self):
            self.shot_comparisons = {}
    
    class CorrectedStatisticalResult:
        pass
    
    class ComputationalAnalyzer:
        pass
    
    class TimingProfiler:
        pass
    
    class MethodTimingResults:
        def __init__(self):
            self.total_time = 0
            self.adaptation_time = 0
            self.inference_time = 0

logger = logging.getLogger(__name__)

# Set publication-ready style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
except:
    # Fallback if seaborn not available
    plt.style.use('default')


class VisualizationReporter:
    """
    Visualization and reporting tools extending existing plotting utilities.
    
    Creates publication-ready figures, performance curves, comparison matrices,
    and automated reports for meta-learning evaluation results.
    """
    
    def __init__(self, 
                 output_dir: str = "meta_learning_results",
                 figure_format: str = "png",
                 dpi: int = 300,
                 style: str = "publication"):
        """
        Initialize VisualizationReporter.
        
        Args:
            output_dir: Directory for saving figures and reports
            figure_format: Figure format ('png', 'pdf', 'svg')
            dpi: Figure resolution
            style: Plotting style ('publication', 'presentation', 'paper')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.figure_format = figure_format
        self.dpi = dpi
        self.style = style
        
        # Configure matplotlib for publication quality
        self._configure_matplotlib()
        
        logger.info(f"VisualizationReporter initialized: {output_dir}")
    
    def _configure_matplotlib(self):
        """Configure matplotlib for publication-ready figures."""
        if self.style == "publication":
            plt.rcParams.update({
                'font.size': 12,
                'axes.titlesize': 14,
                'axes.labelsize': 12,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.titlesize': 16,
                'font.family': 'serif',
                'font.serif': ['Times New Roman'],
                'text.usetex': False,  # Set to True if LaTeX is available
                'figure.figsize': (8, 6),
                'axes.grid': True,
                'grid.alpha': 0.3
            })
        elif self.style == "presentation":
            plt.rcParams.update({
                'font.size': 14,
                'axes.titlesize': 18,
                'axes.labelsize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 12,
                'figure.titlesize': 20,
                'figure.figsize': (10, 7)
            })
    
    def plot_few_shot_performance_curves(self, 
                                       results_dict: Dict[str, Any],
                                       metric: str = "mean_l2_error",
                                       title: str = None,
                                       save_name: str = "few_shot_performance") -> str:
        """
        Extend existing plotting utilities for few-shot performance curves.
        
        Args:
            results_dict: Dictionary of model results
            metric: Metric to plot ('mean_l2_error', 'mean_adaptation_time', etc.)
            title: Plot title
            save_name: Filename for saving
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Plotting few-shot performance curves for {len(results_dict)} models")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        shots = None
        for model_name, results in results_dict.items():
            if hasattr(results, 'shot_results'):
                shot_results = results.shot_results
            else:
                shot_results = results
            
            if shots is None:
                shots = sorted(shot_results.keys())
            
            values = []
            errors = []
            
            for K in shots:
                if K in shot_results:
                    shot_data = shot_results[K]
                    values.append(shot_data.get(metric, np.nan))
                    
                    # Add error bars if std available
                    std_metric = metric.replace('mean_', 'std_')
                    errors.append(shot_data.get(std_metric, 0))
                else:
                    values.append(np.nan)
                    errors.append(0)
            
            # Plot with error bars
            ax.errorbar(shots, values, yerr=errors, 
                       marker='o', linewidth=2, markersize=8,
                       label=model_name, capsize=5)
        
        ax.set_xlabel('Number of Support Samples (K)')
        ax.set_ylabel(self._format_metric_label(metric))
        ax.set_title(title or f'Few-Shot Performance: {self._format_metric_label(metric)}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log scale for error metrics
        if 'error' in metric.lower():
            ax.set_yscale('log')
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Few-shot performance curve saved: {save_path}")
        return str(save_path)
    
    def create_performance_comparison_matrix(self,
                                           results_dict: Dict[str, Any],
                                           metric: str = "mean_l2_error",
                                           save_name: str = "performance_matrix") -> str:
        """
        Generate performance tables and comparison matrices.
        
        Args:
            results_dict: Dictionary of model results
            metric: Metric for comparison
            save_name: Filename for saving
            
        Returns:
            Path to saved figure
        """
        logger.info(f"Creating performance comparison matrix for {metric}")
        
        # Prepare data matrix
        models = list(results_dict.keys())
        first_model = list(results_dict.values())[0]
        
        if hasattr(first_model, 'shot_results'):
            shots = sorted(first_model.shot_results.keys())
        else:
            shots = sorted(first_model.keys())
        
        matrix_data = np.zeros((len(models), len(shots)))
        
        for i, model in enumerate(models):
            results = results_dict[model]
            shot_results = results.shot_results if hasattr(results, 'shot_results') else results
            
            for j, K in enumerate(shots):
                if K in shot_results:
                    matrix_data[i, j] = shot_results[K].get(metric, np.nan)
                else:
                    matrix_data[i, j] = np.nan
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Use log scale for error metrics
        if 'error' in metric.lower():
            matrix_data_plot = np.log10(matrix_data + 1e-10)  # Add small value to avoid log(0)
            cmap = 'viridis_r'  # Reverse colormap so lower errors are better (darker)
        else:
            matrix_data_plot = matrix_data
            cmap = 'viridis'
        
        im = ax.imshow(matrix_data_plot, cmap=cmap, aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(shots)))
        ax.set_xticklabels([f"{K}-shot" for K in shots])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        if 'error' in metric.lower():
            cbar.set_label(f'log10({self._format_metric_label(metric)})')
        else:
            cbar.set_label(self._format_metric_label(metric))
        
        # Add text annotations
        for i in range(len(models)):
            for j in range(len(shots)):
                if not np.isnan(matrix_data[i, j]):
                    text = f'{matrix_data[i, j]:.3f}'
                    ax.text(j, i, text, ha="center", va="center", 
                           color="white" if matrix_data_plot[i, j] < np.nanmean(matrix_data_plot) else "black")
        
        ax.set_title(f'Performance Comparison: {self._format_metric_label(metric)}')
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance matrix saved: {save_path}")
        return str(save_path)
    
    def plot_statistical_significance(self,
                                    comparisons: Dict[str, Any],
                                    save_name: str = "statistical_significance") -> str:
        """
        Visualize statistical significance results.
        
        Args:
            comparisons: Statistical comparison results
            save_name: Filename for saving
            
        Returns:
            Path to saved figure
        """
        logger.info("Plotting statistical significance results")
        
        # Prepare data for plotting
        comparison_data = []
        for comp_name, comparison in comparisons.items():
            shot_comparisons = comparison.shot_comparisons if hasattr(comparison, 'shot_comparisons') else comparison
            
            for K, shot_comp in shot_comparisons.items():
                if 'paired_t_test' in shot_comp:
                    comparison_data.append({
                        'comparison': comp_name,
                        'K_shots': K,
                        'p_value': shot_comp['paired_t_test']['p_value'],
                        'cohens_d': shot_comp['effect_size']['cohens_d'],
                        'significant': shot_comp['paired_t_test']['significant']
                    })
        
        if not comparison_data:
            logger.warning("No statistical comparison data available")
            return ""
        
        df = pd.DataFrame(comparison_data)
        
        # Create subplot figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: P-values
        for comp in df['comparison'].unique():
            comp_data = df[df['comparison'] == comp]
            ax1.plot(comp_data['K_shots'], comp_data['p_value'], 
                    marker='o', linewidth=2, label=comp)
        
        ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax1.set_xlabel('Number of Support Samples (K)')
        ax1.set_ylabel('P-value')
        ax1.set_title('Statistical Significance (P-values)')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes
        for comp in df['comparison'].unique():
            comp_data = df[df['comparison'] == comp]
            ax2.plot(comp_data['K_shots'], comp_data['cohens_d'], 
                    marker='s', linewidth=2, label=comp)
        
        # Add effect size interpretation lines
        ax2.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Small effect')
        ax2.axhline(y=0.5, color='orange', linestyle=':', alpha=0.7, label='Medium effect')
        ax2.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Large effect')
        
        ax2.set_xlabel('Number of Support Samples (K)')
        ax2.set_ylabel("Cohen's d")
        ax2.set_title('Effect Sizes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Statistical significance plot saved: {save_path}")
        return str(save_path)
    
    def generate_automated_report(self,
                                results_dict: Dict[str, Any],
                                statistical_comparisons: Dict[str, Any] = None,
                                computational_report: Dict[str, Any] = None,
                                report_title: str = "Meta-Learning Evaluation Report") -> str:
        """
        Add automated report generation extending existing summary functionality.
        
        Args:
            results_dict: Model evaluation results
            statistical_comparisons: Statistical comparison results
            computational_report: Computational analysis report
            report_title: Title for the report
            
        Returns:
            Path to generated HTML report
        """
        logger.info("Generating automated evaluation report")
        
        # Generate all visualizations
        figures = {}
        
        # Performance curves
        figures['performance_l2'] = self.plot_few_shot_performance_curves(
            results_dict, 'mean_l2_error', 'Few-Shot L2 Relative Error', 'performance_l2_error'
        )
        
        figures['performance_time'] = self.plot_few_shot_performance_curves(
            results_dict, 'mean_adaptation_time', 'Few-Shot Adaptation Time', 'performance_adaptation_time'
        )
        
        # Comparison matrix
        figures['comparison_matrix'] = self.create_performance_comparison_matrix(
            results_dict, 'mean_l2_error', 'l2_error_matrix'
        )
        
        # Statistical significance
        if statistical_comparisons:
            figures['statistical'] = self.plot_statistical_significance(
                statistical_comparisons, 'statistical_significance'
            )
        
        # Generate HTML report
        html_content = self._generate_html_report(
            report_title, results_dict, statistical_comparisons, 
            computational_report, figures
        )
        
        # Save HTML report
        report_path = self.output_dir / "evaluation_report.html"
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Automated report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self,
                            title: str,
                            results_dict: Dict[str, Any],
                            statistical_comparisons: Dict[str, Any],
                            computational_report: Dict[str, Any],
                            figures: Dict[str, str]) -> str:
        """Generate HTML report content."""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 10px; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin: 20px 0; }}
                .figure {{ text-align: center; margin: 30px 0; }}
                .figure img {{ max-width: 100%; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric {{ font-weight: bold; color: #2980b9; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p>This report presents a comprehensive evaluation of meta-learning approaches for few-shot adaptation in physics-informed neural networks (PINNs).</p>
                <ul>
                    <li><strong>Models Evaluated:</strong> {len(results_dict)}</li>
                    <li><strong>Statistical Comparisons:</strong> {len(statistical_comparisons) if statistical_comparisons else 0}</li>
                </ul>
            </div>
        """
        
        # Performance Results Section
        html += """
            <h2>Performance Results</h2>
            <div class="figure">
                <h3>Few-Shot L2 Relative Error</h3>
        """
        if 'performance_l2' in figures:
            html += f'<img src="{os.path.basename(figures["performance_l2"])}" alt="L2 Error Performance">'
        
        html += """
            </div>
            <div class="figure">
                <h3>Few-Shot Adaptation Time</h3>
        """
        if 'performance_time' in figures:
            html += f'<img src="{os.path.basename(figures["performance_time"])}" alt="Adaptation Time Performance">'
        
        html += """
            </div>
            <div class="figure">
                <h3>Performance Comparison Matrix</h3>
        """
        if 'comparison_matrix' in figures:
            html += f'<img src="{os.path.basename(figures["comparison_matrix"])}" alt="Performance Matrix">'
        
        html += "</div>"
        
        # Results Table
        html += self._generate_results_table(results_dict)
        
        html += """
            </body>
        </html>
        """
        
        return html
    
    def _generate_results_table(self, results_dict: Dict[str, Any]) -> str:
        """Generate HTML table for results."""
        html = """
            <h3>Detailed Results</h3>
            <table>
                <tr>
                    <th>Model</th>
                    <th>1-shot L2 Error</th>
                    <th>5-shot L2 Error</th>
                    <th>10-shot L2 Error</th>
                    <th>25-shot L2 Error</th>
                    <th>Success Rate</th>
                </tr>
        """
        
        for model_name, results in results_dict.items():
            html += f"<tr><td>{model_name}</td>"
            
            shot_results = results.shot_results if hasattr(results, 'shot_results') else results
            
            for K in [1, 5, 10, 25]:
                if K in shot_results:
                    error = shot_results[K].get('mean_l2_error', np.nan)
                    html += f"<td>{error:.6f}</td>"
                else:
                    html += "<td>N/A</td>"
            
            # Average success rate
            success_rates = [shot_results[K].get('success_rate', 0) 
                           for K in shot_results.keys()]
            avg_success = np.mean(success_rates) if success_rates else 0
            html += f"<td>{avg_success:.3f}</td>"
            
            html += "</tr>"
        
        html += "</table>"
        return html
    
    def _format_metric_label(self, metric: str) -> str:
        """Format metric name for display."""
        label_map = {
            'mean_l2_error': 'L2 Relative Error',
            'mean_adaptation_time': 'Adaptation Time (s)',
            'mean_inference_time': 'Inference Time (s)',
            'success_rate': 'Success Rate',
            'std_l2_error': 'L2 Error Std Dev'
        }
        return label_map.get(metric, metric.replace('_', ' ').title())
    
    def create_publication_ready_figures(self, 
                                       results_dict: Dict[str, Any]) -> List[str]:
        """
        Create publication-ready figure generation using existing plot callbacks.
        
        Args:
            results_dict: Model evaluation results
            
        Returns:
            List of paths to generated figures
        """
        logger.info("Creating publication-ready figures")
        
        # Set publication style
        original_style = self.style
        self.style = "publication"
        self._configure_matplotlib()
        
        figures = []
        
        try:
            # Main performance figure
            fig_path = self.plot_few_shot_performance_curves(
                results_dict, 'mean_l2_error', 
                'Few-Shot Learning Performance', 'publication_performance'
            )
            figures.append(fig_path)
            
            # Comparison matrix
            fig_path = self.create_performance_comparison_matrix(
                results_dict, 'mean_l2_error', 'publication_matrix'
            )
            figures.append(fig_path)
            
        finally:
            # Restore original style
            self.style = original_style
            self._configure_matplotlib()
        
        logger.info(f"Generated {len(figures)} publication-ready figures")
        return figures
    
    def generate_paper_figures(self, 
                             results_dict: Dict[str, Any],
                             save_prefix: str = "paper_fig") -> Dict[str, str]:
        """
        Generate specific figures for paper publication.
        
        Args:
            results_dict: Model evaluation results
            save_prefix: Prefix for saved figure names
            
        Returns:
            Dictionary mapping figure names to file paths
        """
        logger.info("Generating paper figures")
        
        # Set publication style
        original_style = self.style
        self.style = "publication"
        self._configure_matplotlib()
        
        figures = {}
        
        try:
            # Figure 1: Main performance comparison
            figures['main_performance'] = self.plot_few_shot_performance_curves(
                results_dict, 'mean_l2_error',
                'Few-Shot Learning Performance Comparison',
                f'{save_prefix}_main_performance'
            )
            
            # Figure 2: Adaptation time comparison
            figures['adaptation_time'] = self.plot_few_shot_performance_curves(
                results_dict, 'mean_adaptation_time',
                'Adaptation Time Comparison',
                f'{save_prefix}_adaptation_time'
            )
            
            # Figure 3: Performance matrix heatmap
            figures['performance_matrix'] = self.create_performance_comparison_matrix(
                results_dict, 'mean_l2_error',
                f'{save_prefix}_performance_matrix'
            )
            
        finally:
            # Restore original style
            self.style = original_style
            self._configure_matplotlib()
        
        logger.info(f"Generated {len(figures)} paper figures")
        return figures
    
    def plot_convergence_analysis(self,
                                convergence_data: Dict[str, Any],
                                save_name: str = "convergence_analysis") -> str:
        """
        Plot convergence analysis for different methods.
        
        Args:
            convergence_data: Convergence data for different methods
            save_name: Filename for saving
            
        Returns:
            Path to saved figure
        """
        logger.info("Plotting convergence analysis")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Loss convergence
        for method_name, data in convergence_data.items():
            if 'loss_history' in data:
                iterations = range(len(data['loss_history']))
                ax1.plot(iterations, data['loss_history'], 
                        label=method_name, linewidth=2)
        
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Convergence')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Validation error convergence
        for method_name, data in convergence_data.items():
            if 'validation_error' in data:
                iterations = range(len(data['validation_error']))
                ax2.plot(iterations, data['validation_error'], 
                        label=method_name, linewidth=2)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Validation Error')
        ax2.set_title('Validation Error Convergence')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        save_path = self.output_dir / f"{save_name}.{self.figure_format}"
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Convergence analysis plot saved: {save_path}")
        return str(save_path)