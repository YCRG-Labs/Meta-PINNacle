import json
import os
import time
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import torch


class MetaLearningLogger:
    """Enhanced logging for meta-learning experiments"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}_meta.log")
        self.metrics_file = os.path.join(log_dir, f"{experiment_name}_metrics.json")
        self.config_file = os.path.join(log_dir, f"{experiment_name}_config.json")
        
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"meta_learning_{experiment_name}")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Metrics storage
        self.metrics = {
            'meta_training': [],
            'validation': [],
            'few_shot_evaluation': {},
            'computational_metrics': {},
            'statistical_analysis': {}
        }
        
        self.start_time = time.time()
    
    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration"""
        self.logger.info("=== Experiment Configuration ===")
        for key, value in config.items():
            self.logger.info(f"{key}: {value}")
        
        # Save config to file
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def log_meta_training_step(self, iteration: int, train_loss: float, 
                              val_loss: Optional[float] = None, 
                              learning_rate: Optional[float] = None,
                              additional_metrics: Optional[Dict] = None):
        """Log meta-training step metrics"""
        metrics = {
            'iteration': iteration,
            'train_loss': train_loss,
            'timestamp': time.time() - self.start_time
        }
        
        if val_loss is not None:
            metrics['val_loss'] = val_loss
        if learning_rate is not None:
            metrics['learning_rate'] = learning_rate
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.metrics['meta_training'].append(metrics)
        
        # Log to console/file
        log_msg = f"Iteration {iteration}: Train Loss = {train_loss:.6f}"
        if val_loss is not None:
            log_msg += f", Val Loss = {val_loss:.6f}"
        if learning_rate is not None:
            log_msg += f", LR = {learning_rate:.2e}"
        
        self.logger.info(log_msg)
    
    def log_few_shot_evaluation(self, model_name: str, problem_name: str, 
                               results: Dict[str, Dict]):
        """Log few-shot evaluation results"""
        if model_name not in self.metrics['few_shot_evaluation']:
            self.metrics['few_shot_evaluation'][model_name] = {}
        
        self.metrics['few_shot_evaluation'][model_name][problem_name] = results
        
        self.logger.info(f"=== Few-Shot Evaluation: {model_name} on {problem_name} ===")
        for k_shot, metrics in results.items():
            self.logger.info(f"{k_shot}: Error = {metrics['mean_error']:.6f} ± {metrics['std_error']:.6f}, "
                           f"Time = {metrics['mean_time']:.3f}s ± {metrics['std_time']:.3f}s")
    
    def log_computational_metrics(self, model_name: str, problem_name: str, 
                                 metrics: Dict[str, float]):
        """Log computational performance metrics"""
        key = f"{model_name}_{problem_name}"
        self.metrics['computational_metrics'][key] = metrics
        
        self.logger.info(f"=== Computational Metrics: {model_name} on {problem_name} ===")
        for metric_name, value in metrics.items():
            if 'time' in metric_name.lower():
                self.logger.info(f"{metric_name}: {value:.3f}s")
            elif 'memory' in metric_name.lower():
                self.logger.info(f"{metric_name}: {value:.2f}MB")
            else:
                self.logger.info(f"{metric_name}: {value}")
    
    def log_statistical_analysis(self, analysis_results: Dict):
        """Log statistical analysis results"""
        self.metrics['statistical_analysis'] = analysis_results
        
        self.logger.info("=== Statistical Analysis ===")
        for comparison, stats in analysis_results.items():
            if 'p_value' in stats:
                significance = "significant" if stats['p_value'] < 0.05 else "not significant"
                self.logger.info(f"{comparison}: p={stats['p_value']:.4f} ({significance}), "
                               f"Cohen's d={stats.get('cohens_d', 'N/A'):.3f}")
    
    def log_task_generation(self, problem_name: str, n_train: int, n_val: int, n_test: int):
        """Log task generation information"""
        self.logger.info(f"Generated tasks for {problem_name}: "
                        f"Train={n_train}, Val={n_val}, Test={n_test}")
    
    def log_adaptation_performance(self, task_id: str, adaptation_steps: int, 
                                 initial_loss: float, final_loss: float, 
                                 adaptation_time: float):
        """Log individual task adaptation performance"""
        improvement = (initial_loss - final_loss) / initial_loss * 100
        self.logger.info(f"Task {task_id}: {adaptation_steps} steps, "
                        f"Loss: {initial_loss:.6f} → {final_loss:.6f} "
                        f"({improvement:.1f}% improvement), Time: {adaptation_time:.3f}s")
    
    def log_distributed_training_info(self, world_size: int, rank: int, 
                                    local_batch_size: int, global_batch_size: int):
        """Log distributed training setup information"""
        self.logger.info(f"=== Distributed Training Setup ===")
        self.logger.info(f"World Size: {world_size}, Rank: {rank}")
        self.logger.info(f"Local Batch Size: {local_batch_size}, Global Batch Size: {global_batch_size}")
    
    def log_memory_usage(self, stage: str, memory_mb: float, gpu_memory_mb: Optional[float] = None):
        """Log memory usage at different stages"""
        log_msg = f"Memory Usage ({stage}): {memory_mb:.2f}MB"
        if gpu_memory_mb is not None:
            log_msg += f", GPU: {gpu_memory_mb:.2f}MB"
        self.logger.info(log_msg)
    
    def log_error(self, error_msg: str, exception: Optional[Exception] = None):
        """Log error messages"""
        self.logger.error(error_msg)
        if exception:
            self.logger.exception(exception)
    
    def log_warning(self, warning_msg: str):
        """Log warning messages"""
        self.logger.warning(warning_msg)
    
    def save_metrics(self):
        """Save all metrics to JSON file"""
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            return obj
        
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=convert_numpy)
        
        self.logger.info(f"Metrics saved to {self.metrics_file}")
    
    def load_metrics(self) -> Dict:
        """Load metrics from JSON file"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {}
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of the experiment"""
        total_time = time.time() - self.start_time
        
        summary = {
            'total_experiment_time': total_time,
            'total_meta_training_iterations': len(self.metrics['meta_training']),
            'models_evaluated': list(self.metrics['few_shot_evaluation'].keys()),
            'problems_evaluated': [],
            'best_performance': {},
            'computational_efficiency': {}
        }
        
        # Extract problem names
        for model_results in self.metrics['few_shot_evaluation'].values():
            summary['problems_evaluated'].extend(model_results.keys())
        summary['problems_evaluated'] = list(set(summary['problems_evaluated']))
        
        # Find best performance for each problem
        for problem in summary['problems_evaluated']:
            best_error = float('inf')
            best_model = None
            
            for model, model_results in self.metrics['few_shot_evaluation'].items():
                if problem in model_results:
                    # Use 5-shot performance as benchmark
                    if 'K_5' in model_results[problem]:
                        error = model_results[problem]['K_5']['mean_error']
                        if error < best_error:
                            best_error = error
                            best_model = model
            
            if best_model:
                summary['best_performance'][problem] = {
                    'model': best_model,
                    'error': best_error
                }
        
        return summary
    
    def close(self):
        """Close logger and save final metrics"""
        self.save_metrics()
        summary = self.get_summary_stats()
        
        self.logger.info("=== Experiment Summary ===")
        self.logger.info(f"Total Time: {summary['total_experiment_time']:.2f}s")
        self.logger.info(f"Meta-training Iterations: {summary['total_meta_training_iterations']}")
        self.logger.info(f"Models Evaluated: {', '.join(summary['models_evaluated'])}")
        self.logger.info(f"Problems Evaluated: {', '.join(summary['problems_evaluated'])}")
        
        if summary['best_performance']:
            self.logger.info("Best Performance per Problem:")
            for problem, best in summary['best_performance'].items():
                self.logger.info(f"  {problem}: {best['model']} (Error: {best['error']:.6f})")
        
        # Close handlers
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class MetaLearningCheckpoint:
    """Checkpoint management for meta-learning experiments"""
    
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        self.checkpoint_dir = checkpoint_dir
        self.experiment_name = experiment_name
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_checkpoint(self, model, optimizer, iteration: int, 
                       val_loss: float, additional_data: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"{self.experiment_name}_iter_{iteration}.pt"
        )
        
        checkpoint_data = {
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'timestamp': time.time()
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        torch.save(checkpoint_data, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        if os.path.exists(checkpoint_path):
            return torch.load(checkpoint_path, map_location='cpu')
        return None
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.experiment_name) and file.endswith('.pt'):
                checkpoints.append(file)
        
        if checkpoints:
            # Sort by iteration number
            checkpoints.sort(key=lambda x: int(x.split('_iter_')[1].split('.pt')[0]))
            return os.path.join(self.checkpoint_dir, checkpoints[-1])
        
        return None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """Remove old checkpoints, keeping only the last N"""
        checkpoints = []
        for file in os.listdir(self.checkpoint_dir):
            if file.startswith(self.experiment_name) and file.endswith('.pt'):
                checkpoints.append(file)
        
        if len(checkpoints) > keep_last_n:
            # Sort by iteration number
            checkpoints.sort(key=lambda x: int(x.split('_iter_')[1].split('.pt')[0]))
            
            # Remove old checkpoints
            for checkpoint in checkpoints[:-keep_last_n]:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint)
                os.remove(checkpoint_path)