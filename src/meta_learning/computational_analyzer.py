"""
Computational trade-off analysis for meta-learning vs standard training.
Extends existing timing utilities to measure training costs, break-even points,
memory usage, and adaptation speed.
"""

import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training performance metrics"""
    training_time: float
    memory_peak: float
    memory_average: float
    gpu_memory_peak: float
    gpu_memory_average: float
    convergence_time: float  # Time to reach target accuracy
    final_accuracy: float


@dataclass
class AdaptationMetrics:
    """Adaptation performance metrics"""
    adaptation_time: float
    memory_usage: float
    gpu_memory_usage: float
    steps_to_target: int
    final_accuracy: float


@dataclass
class BreakEvenAnalysis:
    """Break-even point analysis results"""
    meta_training_cost: float
    standard_training_cost: float
    adaptation_cost_per_task: float
    break_even_tasks: int
    cost_savings_after_break_even: float


class MemoryMonitor:
    """Memory usage monitoring utility"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.monitoring = False
        self.memory_history = []
        self.gpu_memory_history = []
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start memory monitoring in background thread"""
        self.monitoring = True
        self.memory_history.clear()
        self.gpu_memory_history.clear()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Tuple[float, float, float, float]:
        """
        Stop monitoring and return statistics.
        
        Returns:
            Tuple of (peak_memory, avg_memory, peak_gpu_memory, avg_gpu_memory) in MB
        """
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.memory_history:
            return 0.0, 0.0, 0.0, 0.0
        
        peak_memory = max(self.memory_history)
        avg_memory = np.mean(self.memory_history)
        peak_gpu = max(self.gpu_memory_history) if self.gpu_memory_history else 0.0
        avg_gpu = np.mean(self.gpu_memory_history) if self.gpu_memory_history else 0.0
        
        return peak_memory, avg_memory, peak_gpu, avg_gpu
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            # System memory
            memory_mb = psutil.virtual_memory().used / 1024 / 1024
            self.memory_history.append(memory_mb)
            
            # GPU memory
            if self.device.startswith('cuda') and torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    self.gpu_memory_history.append(gpu_memory_mb)
                except:
                    pass
            
            time.sleep(0.1)  # Monitor every 100ms


class ComputationalAnalyzer:
    """
    Computational trade-off analysis extending existing timing utilities.
    
    Measures training costs, break-even points, memory usage, and adaptation speed
    to quantify when meta-learning becomes worthwhile compared to standard training.
    """
    
    def __init__(self, device: str = 'cuda', target_accuracy: float = 0.01):
        """
        Initialize ComputationalAnalyzer.
        
        Args:
            device: Computing device ('cuda' or 'cpu')
            target_accuracy: Target accuracy threshold for convergence analysis
        """
        self.device = device
        self.target_accuracy = target_accuracy
        self.memory_monitor = MemoryMonitor(device)
        
        # Storage for analysis results
        self.training_metrics = {}
        self.adaptation_metrics = defaultdict(list)
        
        logger.info(f"ComputationalAnalyzer initialized for device: {device}")
    
    def measure_training_time(self, 
                            model: Any,
                            training_function: callable,
                            model_name: str,
                            **training_kwargs) -> TrainingMetrics:
        """
        Implement training time measurement extending existing timing utilities.
        
        Args:
            model: Model to train
            training_function: Function to call for training
            model_name: Name identifier for the model
            **training_kwargs: Additional training arguments
            
        Returns:
            TrainingMetrics with comprehensive timing and memory statistics
        """
        logger.info(f"Measuring training time for {model_name}")
        
        # Start monitoring
        self.memory_monitor.start_monitoring()
        start_time = time.time()
        
        # Track convergence
        convergence_time = None
        final_accuracy = None
        
        try:
            # Custom training with convergence tracking
            if hasattr(model, 'train_with_monitoring'):
                result = model.train_with_monitoring(
                    target_accuracy=self.target_accuracy,
                    **training_kwargs
                )
                convergence_time = result.get('convergence_time', None)
                final_accuracy = result.get('final_accuracy', None)
            else:
                # Standard training
                result = training_function(model, **training_kwargs)
                final_accuracy = getattr(result, 'final_accuracy', None)
        
        except Exception as e:
            logger.error(f"Training failed for {model_name}: {e}")
            final_accuracy = np.nan
        
        # Stop monitoring and collect metrics
        training_time = time.time() - start_time
        peak_mem, avg_mem, peak_gpu, avg_gpu = self.memory_monitor.stop_monitoring()
        
        if convergence_time is None:
            convergence_time = training_time  # Use full training time if no convergence tracking
        
        metrics = TrainingMetrics(
            training_time=training_time,
            memory_peak=peak_mem,
            memory_average=avg_mem,
            gpu_memory_peak=peak_gpu,
            gpu_memory_average=avg_gpu,
            convergence_time=convergence_time,
            final_accuracy=final_accuracy if final_accuracy is not None else np.nan
        )
        
        self.training_metrics[model_name] = metrics
        
        logger.info(f"Training completed for {model_name}: {training_time:.2f}s, Peak Memory: {peak_mem:.1f}MB")
        
        return metrics
    
    def measure_adaptation_speed(self,
                               model: Any,
                               task_data: Any,
                               model_name: str,
                               max_steps: int = 100) -> AdaptationMetrics:
        """
        Measure adaptation speed to target accuracy.
        
        Args:
            model: Model to adapt
            task_data: Task data for adaptation
            model_name: Model identifier
            max_steps: Maximum adaptation steps to try
            
        Returns:
            AdaptationMetrics with adaptation performance
        """
        logger.info(f"Measuring adaptation speed for {model_name}")
        
        self.memory_monitor.start_monitoring()
        start_time = time.time()
        
        steps_to_target = max_steps
        final_accuracy = np.nan
        
        try:
            if hasattr(model, 'adapt_with_monitoring'):
                # Use model's built-in monitoring
                result = model.adapt_with_monitoring(
                    task_data,
                    target_accuracy=self.target_accuracy,
                    max_steps=max_steps
                )
                steps_to_target = result.get('steps_to_target', max_steps)
                final_accuracy = result.get('final_accuracy', np.nan)
            
            elif hasattr(model, 'adapt'):
                # Standard adaptation - measure accuracy at each step
                for step in range(1, max_steps + 1):
                    adapted_model = model.adapt(task_data, steps=step)
                    
                    # Evaluate accuracy
                    if hasattr(adapted_model, 'evaluate'):
                        accuracy = adapted_model.evaluate(task_data)
                        if accuracy <= self.target_accuracy:
                            steps_to_target = step
                            final_accuracy = accuracy
                            break
                        final_accuracy = accuracy
            
            else:
                logger.warning(f"Model {model_name} does not support adaptation")
        
        except Exception as e:
            logger.error(f"Adaptation measurement failed for {model_name}: {e}")
        
        adaptation_time = time.time() - start_time
        peak_mem, avg_mem, peak_gpu, avg_gpu = self.memory_monitor.stop_monitoring()
        
        metrics = AdaptationMetrics(
            adaptation_time=adaptation_time,
            memory_usage=peak_mem,
            gpu_memory_usage=peak_gpu,
            steps_to_target=steps_to_target,
            final_accuracy=final_accuracy
        )
        
        self.adaptation_metrics[model_name].append(metrics)
        
        logger.info(f"Adaptation measured for {model_name}: {adaptation_time:.2f}s, {steps_to_target} steps")
        
        return metrics
    
    def calculate_break_even_point(self,
                                 meta_model_name: str,
                                 standard_model_name: str,
                                 num_adaptation_tasks: int = 100) -> BreakEvenAnalysis:
        """
        Create break-even point calculation for meta-learning vs standard training.
        
        Args:
            meta_model_name: Name of meta-learning model
            standard_model_name: Name of standard model
            num_adaptation_tasks: Number of adaptation tasks to consider
            
        Returns:
            BreakEvenAnalysis with cost comparison
        """
        logger.info(f"Calculating break-even point: {meta_model_name} vs {standard_model_name}")
        
        # Get training costs
        meta_training_cost = self.training_metrics.get(meta_model_name, TrainingMetrics(0, 0, 0, 0, 0, 0, 0)).training_time
        standard_training_cost = self.training_metrics.get(standard_model_name, TrainingMetrics(0, 0, 0, 0, 0, 0, 0)).training_time
        
        # Get adaptation costs
        meta_adaptations = self.adaptation_metrics.get(meta_model_name, [])
        standard_adaptations = self.adaptation_metrics.get(standard_model_name, [])
        
        if not meta_adaptations or not standard_adaptations:
            logger.warning("Insufficient adaptation data for break-even analysis")
            return BreakEvenAnalysis(
                meta_training_cost=meta_training_cost,
                standard_training_cost=standard_training_cost,
                adaptation_cost_per_task=0,
                break_even_tasks=np.inf,
                cost_savings_after_break_even=0
            )
        
        # Average adaptation costs
        meta_adapt_cost = np.mean([m.adaptation_time for m in meta_adaptations])
        standard_adapt_cost = np.mean([m.adaptation_time for m in standard_adaptations])
        
        # Calculate break-even point
        # Meta cost: meta_training_cost + N * meta_adapt_cost
        # Standard cost: N * standard_training_cost
        # Break-even when: meta_training_cost + N * meta_adapt_cost = N * standard_training_cost
        # Solve for N: N = meta_training_cost / (standard_training_cost - meta_adapt_cost)
        
        cost_difference_per_task = standard_training_cost - meta_adapt_cost
        
        if cost_difference_per_task <= 0:
            # Meta-learning is never beneficial
            break_even_tasks = np.inf
            cost_savings = 0
        else:
            break_even_tasks = meta_training_cost / cost_difference_per_task
            
            # Cost savings after break-even (per additional task)
            cost_savings = cost_difference_per_task
        
        analysis = BreakEvenAnalysis(
            meta_training_cost=meta_training_cost,
            standard_training_cost=standard_training_cost,
            adaptation_cost_per_task=meta_adapt_cost,
            break_even_tasks=break_even_tasks,
            cost_savings_after_break_even=cost_savings
        )
        
        logger.info(f"Break-even analysis: {break_even_tasks:.1f} tasks needed")
        
        return analysis
    
    def add_memory_usage_tracking(self, model_name: str) -> Dict[str, float]:
        """
        Add memory usage tracking using existing monitoring.
        
        Args:
            model_name: Model to track
            
        Returns:
            Dictionary with memory usage statistics
        """
        if model_name in self.training_metrics:
            training_mem = self.training_metrics[model_name]
            memory_stats = {
                'training_memory_peak_mb': training_mem.memory_peak,
                'training_memory_avg_mb': training_mem.memory_average,
                'training_gpu_memory_peak_mb': training_mem.gpu_memory_peak,
                'training_gpu_memory_avg_mb': training_mem.gpu_memory_average
            }
        else:
            memory_stats = {
                'training_memory_peak_mb': 0,
                'training_memory_avg_mb': 0,
                'training_gpu_memory_peak_mb': 0,
                'training_gpu_memory_avg_mb': 0
            }
        
        # Add adaptation memory statistics
        if model_name in self.adaptation_metrics:
            adaptations = self.adaptation_metrics[model_name]
            memory_stats.update({
                'adaptation_memory_avg_mb': np.mean([a.memory_usage for a in adaptations]),
                'adaptation_memory_max_mb': np.max([a.memory_usage for a in adaptations]),
                'adaptation_gpu_memory_avg_mb': np.mean([a.gpu_memory_usage for a in adaptations]),
                'adaptation_gpu_memory_max_mb': np.max([a.gpu_memory_usage for a in adaptations])
            })
        
        return memory_stats
    
    def generate_computational_report(self, models: List[str]) -> Dict[str, Any]:
        """
        Generate comprehensive computational trade-off report.
        
        Args:
            models: List of model names to include in report
            
        Returns:
            Dictionary containing comprehensive computational analysis
        """
        logger.info(f"Generating computational report for {len(models)} models")
        
        report = {
            'training_performance': {},
            'adaptation_performance': {},
            'memory_analysis': {},
            'break_even_analysis': {},
            'summary_statistics': {}
        }
        
        # Training performance
        for model in models:
            if model in self.training_metrics:
                metrics = self.training_metrics[model]
                report['training_performance'][model] = {
                    'training_time_seconds': metrics.training_time,
                    'convergence_time_seconds': metrics.convergence_time,
                    'final_accuracy': metrics.final_accuracy,
                    'memory_efficiency': metrics.memory_peak / max(metrics.training_time, 1)  # MB per second
                }
        
        # Adaptation performance
        for model in models:
            if model in self.adaptation_metrics:
                adaptations = self.adaptation_metrics[model]
                report['adaptation_performance'][model] = {
                    'mean_adaptation_time': np.mean([a.adaptation_time for a in adaptations]),
                    'std_adaptation_time': np.std([a.adaptation_time for a in adaptations]),
                    'mean_steps_to_target': np.mean([a.steps_to_target for a in adaptations]),
                    'adaptation_success_rate': np.mean([a.final_accuracy <= self.target_accuracy for a in adaptations if not np.isnan(a.final_accuracy)])
                }
        
        # Memory analysis
        for model in models:
            report['memory_analysis'][model] = self.add_memory_usage_tracking(model)
        
        # Break-even analysis (compare meta-learning models with standard models)
        meta_models = [m for m in models if 'meta' in m.lower() or 'maml' in m.lower()]
        standard_models = [m for m in models if 'standard' in m.lower() or 'baseline' in m.lower()]
        
        for meta_model in meta_models:
            for standard_model in standard_models:
                key = f"{meta_model}_vs_{standard_model}"
                report['break_even_analysis'][key] = self.calculate_break_even_point(meta_model, standard_model)
        
        # Summary statistics
        if report['training_performance']:
            training_times = [v['training_time_seconds'] for v in report['training_performance'].values()]
            report['summary_statistics']['training_time_range'] = (min(training_times), max(training_times))
            report['summary_statistics']['training_time_ratio'] = max(training_times) / min(training_times) if min(training_times) > 0 else np.inf
        
        if report['adaptation_performance']:
            adaptation_times = [v['mean_adaptation_time'] for v in report['adaptation_performance'].values()]
            report['summary_statistics']['adaptation_time_range'] = (min(adaptation_times), max(adaptation_times))
        
        logger.info("Computational report generated successfully")
        
        return report
    
    def export_timing_analysis(self, output_path: str = None) -> Dict[str, Any]:
        """
        Export comprehensive timing analysis extending existing summary functionality.
        
        Args:
            output_path: Optional path to save analysis
            
        Returns:
            Dictionary containing timing analysis
        """
        all_models = list(set(list(self.training_metrics.keys()) + list(self.adaptation_metrics.keys())))
        analysis = self.generate_computational_report(all_models)
        
        if output_path:
            import json
            with open(output_path, 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return obj
                
                json.dump(analysis, f, indent=2, default=convert_numpy)
            logger.info(f"Timing analysis exported to {output_path}")
        
        return analysis
    
    def reset_measurements(self):
        """Reset all measurements for new analysis."""
        self.training_metrics.clear()
        self.adaptation_metrics.clear()
        logger.info("Computational measurements reset")