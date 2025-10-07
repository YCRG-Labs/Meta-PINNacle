"""Enhanced logging and monitoring for meta-learning training phases.

This module provides comprehensive logging infrastructure for meta-learning
specific metrics, debugging utilities for convergence issues, and performance
monitoring for distributed training.
"""

import os
import time
import json
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict, deque
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path

from .error_handling import MetaLearningError


class MetaLearningPhase:
    """Enumeration of meta-learning training phases."""
    META_TRAINING = "meta_training"
    ADAPTATION = "adaptation"
    VALIDATION = "validation"
    EVALUATION = "evaluation"
    DISTRIBUTED_SYNC = "distributed_sync"


class MetaLearningMonitor:
    """Comprehensive monitoring for meta-learning training."""
    
    def __init__(self, log_dir: str, experiment_name: str, 
                 rank: int = 0, world_size: int = 1):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.rank = rank
        self.world_size = world_size
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging files
        self.setup_logging()
        
        # Metrics storage
        self.metrics = defaultdict(list)
        self.phase_metrics = defaultdict(lambda: defaultdict(list))
        self.error_metrics = defaultdict(int)
        self.convergence_metrics = defaultdict(list)
        self.performance_metrics = defaultdict(list)
        
        # Timing
        self.phase_timers = {}
        self.operation_timers = {}
        
        # Memory monitoring
        self.memory_snapshots = []
        self.memory_alerts = []
        
        # Convergence monitoring
        self.convergence_window = 100
        self.convergence_history = deque(maxlen=self.convergence_window)
        self.stagnation_threshold = 1e-6
        self.stagnation_patience = 50
        
        # Distributed monitoring
        self.communication_metrics = defaultdict(list)
        self.synchronization_failures = 0
        
        # Thread-safe logging
        self.log_lock = threading.Lock()
        
        # Start monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def setup_logging(self):
        """Setup comprehensive logging infrastructure."""
        # Main logger
        self.logger = logging.getLogger(f"meta_learning_{self.experiment_name}_rank_{self.rank}")
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handlers for different log levels
        log_files = {
            'debug': self.log_dir / f"{self.experiment_name}_rank_{self.rank}_debug.log",
            'info': self.log_dir / f"{self.experiment_name}_rank_{self.rank}_info.log",
            'error': self.log_dir / f"{self.experiment_name}_rank_{self.rank}_error.log",
            'metrics': self.log_dir / f"{self.experiment_name}_rank_{self.rank}_metrics.log"
        }
        
        # Create file handlers
        self.file_handlers = {}
        for level, log_file in log_files.items():
            handler = logging.FileHandler(log_file)
            if level == 'debug':
                handler.setLevel(logging.DEBUG)
            elif level == 'info':
                handler.setLevel(logging.INFO)
            elif level == 'error':
                handler.setLevel(logging.ERROR)
            elif level == 'metrics':
                handler.setLevel(logging.INFO)
            
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            self.file_handlers[level] = handler
            
            if level != 'metrics':  # Don't add metrics handler to main logger
                self.logger.addHandler(handler)
        
        # Console handler (only for rank 0 to avoid spam)
        if self.rank == 0:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Metrics logger
        self.metrics_logger = logging.getLogger(f"metrics_{self.experiment_name}_rank_{self.rank}")
        self.metrics_logger.setLevel(logging.INFO)
        self.metrics_logger.addHandler(self.file_handlers['metrics'])
        self.metrics_logger.propagate = False
    
    def log_phase_start(self, phase: str, context: Optional[Dict[str, Any]] = None):
        """Log the start of a meta-learning phase."""
        with self.log_lock:
            timestamp = time.time()
            self.phase_timers[phase] = timestamp
            
            context_str = f" - Context: {context}" if context else ""
            self.logger.info(f"Phase {phase} started{context_str}")
            
            # Log to metrics
            self.metrics_logger.info(json.dumps({
                'event': 'phase_start',
                'phase': phase,
                'timestamp': timestamp,
                'rank': self.rank,
                'context': context or {}
            }))
    
    def log_phase_end(self, phase: str, metrics: Optional[Dict[str, Any]] = None):
        """Log the end of a meta-learning phase."""
        with self.log_lock:
            end_time = time.time()
            start_time = self.phase_timers.get(phase, end_time)
            duration = end_time - start_time
            
            self.logger.info(f"Phase {phase} completed in {duration:.2f}s")
            
            # Store phase metrics
            if metrics:
                self.phase_metrics[phase].append({
                    'timestamp': end_time,
                    'duration': duration,
                    'metrics': metrics
                })
            
            # Log to metrics
            self.metrics_logger.info(json.dumps({
                'event': 'phase_end',
                'phase': phase,
                'timestamp': end_time,
                'duration': duration,
                'rank': self.rank,
                'metrics': metrics or {}
            }))
    
    def log_meta_training_step(self, iteration: int, meta_loss: float, 
                              task_losses: List[float], adaptation_times: List[float]):
        """Log meta-training step metrics."""
        with self.log_lock:
            metrics = {
                'iteration': iteration,
                'meta_loss': meta_loss,
                'mean_task_loss': np.mean(task_losses),
                'std_task_loss': np.std(task_losses),
                'mean_adaptation_time': np.mean(adaptation_times),
                'std_adaptation_time': np.std(adaptation_times),
                'timestamp': time.time()
            }
            
            self.phase_metrics[MetaLearningPhase.META_TRAINING].append(metrics)
            
            # Check for convergence issues
            self.convergence_history.append(meta_loss)
            self._check_convergence_issues(iteration, meta_loss)
            
            # Log every 10 iterations or if there's an issue
            if iteration % 10 == 0 or self._has_convergence_issue():
                self.logger.info(
                    f"Meta-training step {iteration}: loss={meta_loss:.6f}, "
                    f"mean_task_loss={metrics['mean_task_loss']:.6f}, "
                    f"adaptation_time={metrics['mean_adaptation_time']:.3f}s"
                )
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'meta_training_step',
                'rank': self.rank,
                **metrics
            }))
    
    def log_adaptation_step(self, task_id: str, step: int, loss: float, 
                           gradient_norm: Optional[float] = None):
        """Log adaptation step metrics."""
        with self.log_lock:
            metrics = {
                'task_id': task_id,
                'step': step,
                'loss': loss,
                'gradient_norm': gradient_norm,
                'timestamp': time.time()
            }
            
            self.phase_metrics[MetaLearningPhase.ADAPTATION].append(metrics)
            
            # Check for adaptation issues
            if gradient_norm is not None:
                if gradient_norm > 100:  # Large gradient
                    self.logger.warning(f"Large gradient norm in adaptation: {gradient_norm:.2f}")
                elif gradient_norm < 1e-8:  # Vanishing gradient
                    self.logger.warning(f"Vanishing gradient in adaptation: {gradient_norm:.2e}")
            
            if np.isnan(loss) or np.isinf(loss):
                self.logger.error(f"NaN/Inf loss in adaptation step {step} for task {task_id}")
                self.error_metrics['adaptation_nan_inf'] += 1
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'adaptation_step',
                'rank': self.rank,
                **metrics
            }))
    
    def log_distributed_sync(self, sync_type: str, duration: float, 
                           success: bool, error: Optional[str] = None):
        """Log distributed synchronization events."""
        with self.log_lock:
            metrics = {
                'sync_type': sync_type,
                'duration': duration,
                'success': success,
                'error': error,
                'timestamp': time.time()
            }
            
            self.communication_metrics[sync_type].append(metrics)
            
            if not success:
                self.synchronization_failures += 1
                self.logger.error(f"Distributed sync failed: {sync_type} - {error}")
            else:
                self.logger.debug(f"Distributed sync successful: {sync_type} in {duration:.3f}s")
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'distributed_sync',
                'rank': self.rank,
                **metrics
            }))
    
    def log_memory_usage(self, context: str):
        """Log current memory usage."""
        if not torch.cuda.is_available():
            return
        
        with self.log_lock:
            memory_info = {
                'context': context,
                'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**3,  # GB
                'timestamp': time.time()
            }
            
            self.memory_snapshots.append(memory_info)
            
            # Check for memory issues
            if memory_info['allocated'] > 10:  # More than 10GB
                self.memory_alerts.append({
                    'level': 'warning',
                    'message': f"High memory usage: {memory_info['allocated']:.2f}GB",
                    'context': context,
                    'timestamp': time.time()
                })
                self.logger.warning(f"High memory usage in {context}: {memory_info['allocated']:.2f}GB")
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'memory_usage',
                'rank': self.rank,
                **memory_info
            }))
    
    def log_error(self, error: Exception, context: str, 
                  recovery_attempted: bool = False, recovery_successful: bool = False):
        """Log error with context and recovery information."""
        with self.log_lock:
            error_info = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'recovery_attempted': recovery_attempted,
                'recovery_successful': recovery_successful,
                'traceback': traceback.format_exc(),
                'timestamp': time.time()
            }
            
            self.error_metrics[type(error).__name__] += 1
            
            self.logger.error(
                f"Error in {context}: {type(error).__name__}: {error}\n"
                f"Recovery attempted: {recovery_attempted}, successful: {recovery_successful}"
            )
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'error',
                'rank': self.rank,
                **error_info
            }))
    
    def log_performance_metrics(self, metrics: Dict[str, float], context: str):
        """Log performance metrics."""
        with self.log_lock:
            perf_metrics = {
                'context': context,
                'timestamp': time.time(),
                **metrics
            }
            
            self.performance_metrics[context].append(perf_metrics)
            
            self.logger.info(f"Performance metrics for {context}: {metrics}")
            
            # Log to metrics file
            self.metrics_logger.info(json.dumps({
                'event': 'performance_metrics',
                'rank': self.rank,
                **perf_metrics
            }))
    
    def _check_convergence_issues(self, iteration: int, loss: float):
        """Check for convergence issues."""
        if len(self.convergence_history) < 10:
            return
        
        recent_losses = list(self.convergence_history)[-10:]
        
        # Check for stagnation
        if len(recent_losses) >= self.stagnation_patience:
            loss_std = np.std(recent_losses[-self.stagnation_patience:])
            if loss_std < self.stagnation_threshold:
                self.logger.warning(f"Training stagnation detected at iteration {iteration}")
                self.convergence_metrics['stagnation_warnings'].append({
                    'iteration': iteration,
                    'loss': loss,
                    'loss_std': loss_std,
                    'timestamp': time.time()
                })
        
        # Check for divergence
        if len(recent_losses) >= 5:
            recent_trend = np.polyfit(range(5), recent_losses[-5:], 1)[0]
            if recent_trend > 0.1:  # Loss increasing rapidly
                self.logger.warning(f"Training divergence detected at iteration {iteration}")
                self.convergence_metrics['divergence_warnings'].append({
                    'iteration': iteration,
                    'loss': loss,
                    'trend': recent_trend,
                    'timestamp': time.time()
                })
        
        # Check for NaN/Inf
        if np.isnan(loss) or np.isinf(loss):
            self.logger.error(f"NaN/Inf loss detected at iteration {iteration}")
            self.convergence_metrics['nan_inf_losses'].append({
                'iteration': iteration,
                'loss': loss,
                'timestamp': time.time()
            })
    
    def _has_convergence_issue(self) -> bool:
        """Check if there are recent convergence issues."""
        current_time = time.time()
        recent_threshold = 60  # 1 minute
        
        for issue_type, issues in self.convergence_metrics.items():
            if issues and current_time - issues[-1]['timestamp'] < recent_threshold:
                return True
        
        return False
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Log memory usage periodically
                if torch.cuda.is_available():
                    self.log_memory_usage("periodic_monitoring")
                
                # Check for distributed health
                if self.world_size > 1 and dist.is_initialized():
                    self._check_distributed_health()
                
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer if there's an error
    
    def _check_distributed_health(self):
        """Check distributed training health."""
        try:
            # Simple health check
            test_tensor = torch.tensor([1.0])
            if torch.cuda.is_available():
                test_tensor = test_tensor.cuda()
            
            start_time = time.time()
            dist.all_reduce(test_tensor)
            duration = time.time() - start_time
            
            self.log_distributed_sync("health_check", duration, True)
            
        except Exception as e:
            self.log_distributed_sync("health_check", 0, False, str(e))
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get summary statistics for the experiment."""
        with self.log_lock:
            summary = {
                'experiment_name': self.experiment_name,
                'rank': self.rank,
                'world_size': self.world_size,
                'total_errors': sum(self.error_metrics.values()),
                'error_breakdown': dict(self.error_metrics),
                'synchronization_failures': self.synchronization_failures,
                'convergence_issues': {
                    'stagnation_warnings': len(self.convergence_metrics.get('stagnation_warnings', [])),
                    'divergence_warnings': len(self.convergence_metrics.get('divergence_warnings', [])),
                    'nan_inf_losses': len(self.convergence_metrics.get('nan_inf_losses', []))
                },
                'memory_alerts': len(self.memory_alerts),
                'phase_counts': {phase: len(metrics) for phase, metrics in self.phase_metrics.items()}
            }
            
            return summary
    
    def save_experiment_data(self):
        """Save all experiment data to files."""
        with self.log_lock:
            # Save metrics
            metrics_file = self.log_dir / f"{self.experiment_name}_rank_{self.rank}_all_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump({
                    'phase_metrics': {k: v for k, v in self.phase_metrics.items()},
                    'error_metrics': dict(self.error_metrics),
                    'convergence_metrics': {k: v for k, v in self.convergence_metrics.items()},
                    'performance_metrics': {k: v for k, v in self.performance_metrics.items()},
                    'communication_metrics': {k: v for k, v in self.communication_metrics.items()},
                    'memory_snapshots': self.memory_snapshots,
                    'memory_alerts': self.memory_alerts
                }, f, indent=2, default=str)
            
            # Save summary
            summary_file = self.log_dir / f"{self.experiment_name}_rank_{self.rank}_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(self.get_summary_statistics(), f, indent=2)
    
    def cleanup(self):
        """Cleanup monitoring resources."""
        self.monitoring_active = False
        if self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        # Save final data
        self.save_experiment_data()
        
        # Close file handlers
        for handler in self.file_handlers.values():
            handler.close()


# Global monitor instance
_global_monitor: Optional[MetaLearningMonitor] = None


def setup_meta_learning_monitoring(log_dir: str, experiment_name: str, 
                                 rank: int = 0, world_size: int = 1) -> MetaLearningMonitor:
    """Setup global meta-learning monitoring."""
    global _global_monitor
    _global_monitor = MetaLearningMonitor(log_dir, experiment_name, rank, world_size)
    return _global_monitor


def get_meta_learning_monitor() -> Optional[MetaLearningMonitor]:
    """Get the global meta-learning monitor."""
    return _global_monitor


def cleanup_meta_learning_monitoring():
    """Cleanup global meta-learning monitoring."""
    global _global_monitor
    if _global_monitor:
        _global_monitor.cleanup()
        _global_monitor = None