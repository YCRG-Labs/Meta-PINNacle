import os
import time
import numpy as np
import torch
from typing import Dict, List, Optional, Any
from deepxde.callbacks import Callback
from src.utils.meta_logging import MetaLearningLogger
from src.meta_learning.evaluation_framework import EvaluationFramework


class MetaLearningCallback(Callback):
    """Base callback for meta-learning experiments"""
    
    def __init__(self, log_every: int = 100, verbose: bool = True):
        super().__init__()
        self.log_every = log_every
        self.verbose = verbose
        self.iteration = 0
        self.start_time = None
        
    def on_meta_train_begin(self, model, logger: MetaLearningLogger):
        """Called at the beginning of meta-training"""
        self.start_time = time.time()
        if self.verbose:
            print(f"Starting meta-training for {type(model).__name__}")
    
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        """Called at the end of each meta-training iteration"""
        self.iteration = iteration
        
        if self.log_every and iteration % self.log_every == 0:
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            if self.verbose:
                log_msg = f"Meta-Iteration {iteration}: Train Loss = {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss = {val_loss:.6f}"
                log_msg += f", Time = {elapsed_time:.2f}s"
                print(log_msg)
    
    def on_meta_train_end(self, model, logger: Optional[MetaLearningLogger] = None):
        """Called at the end of meta-training"""
        total_time = time.time() - self.start_time if self.start_time else 0
        if self.verbose:
            print(f"Meta-training completed in {total_time:.2f}s after {self.iteration} iterations")


class MetaValidationCallback(MetaLearningCallback):
    """Callback for meta-learning validation during training"""
    
    def __init__(self, val_frequency: int = 100, patience: int = 10, 
                 min_delta: float = 1e-6, verbose: bool = True):
        super().__init__(log_every=val_frequency, verbose=verbose)
        self.val_frequency = val_frequency
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.val_losses = []
        self.should_stop = False
        
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        super().on_meta_iteration_end(iteration, train_loss, val_loss, model, logger)
        
        if val_loss is not None and iteration % self.val_frequency == 0:
            self.val_losses.append(val_loss)
            
            # Early stopping logic
            if val_loss < self.best_val_loss - self.min_delta:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                if self.verbose:
                    print(f"New best validation loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(f"Early stopping triggered after {self.patience} iterations without improvement")
    
    def should_stop_training(self) -> bool:
        """Check if training should be stopped early"""
        return self.should_stop


class MetaCheckpointCallback(MetaLearningCallback):
    """Callback for saving meta-learning checkpoints"""
    
    def __init__(self, checkpoint_frequency: int = 1000, 
                 save_best: bool = True, verbose: bool = True):
        super().__init__(log_every=checkpoint_frequency, verbose=verbose)
        self.checkpoint_frequency = checkpoint_frequency
        self.save_best = save_best
        self.best_val_loss = float('inf')
        self.checkpoint_dir = None
        
    def on_meta_train_begin(self, model, logger: MetaLearningLogger):
        super().on_meta_train_begin(model, logger)
        self.checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        super().on_meta_iteration_end(iteration, train_loss, val_loss, model, logger)
        
        should_save = False
        
        # Save at regular intervals
        if iteration % self.checkpoint_frequency == 0:
            should_save = True
            
        # Save best model
        if self.save_best and val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            should_save = True
            
        if should_save and model is not None and self.checkpoint_dir:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f"meta_model_iter_{iteration}.pt"
            )
            
            checkpoint_data = {
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': self.best_val_loss
            }
            
            if hasattr(model, 'meta_optimizer'):
                checkpoint_data['optimizer_state_dict'] = model.meta_optimizer.state_dict()
                
            torch.save(checkpoint_data, checkpoint_path)
            
            if self.verbose:
                print(f"Checkpoint saved: {checkpoint_path}")


class FewShotEvaluationCallback(MetaLearningCallback):
    """Callback for few-shot evaluation during meta-training"""
    
    def __init__(self, evaluation_frequency: int = 1000, 
                 evaluation_shots: List[int] = [1, 5, 10, 25],
                 n_evaluation_tasks: int = 10, verbose: bool = True):
        super().__init__(log_every=evaluation_frequency, verbose=verbose)
        self.evaluation_frequency = evaluation_frequency
        self.evaluation_shots = evaluation_shots
        self.n_evaluation_tasks = n_evaluation_tasks
        self.evaluation_framework = None
        
    def on_meta_train_begin(self, model, logger: MetaLearningLogger):
        super().on_meta_train_begin(model, logger)
        # Initialize evaluation framework
        self.evaluation_framework = EvaluationFramework()
        
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        super().on_meta_iteration_end(iteration, train_loss, val_loss, model, logger)
        
        if (iteration % self.evaluation_frequency == 0 and 
            model is not None and 
            hasattr(model, 'test_tasks') and 
            self.evaluation_framework is not None):
            
            # Sample evaluation tasks
            eval_tasks = model.test_tasks[:self.n_evaluation_tasks]
            
            # Run few-shot evaluation
            results = self.evaluation_framework.evaluate_few_shot_performance(
                model, eval_tasks, shots=self.evaluation_shots
            )
            
            if self.verbose:
                print(f"\n=== Few-Shot Evaluation at Iteration {iteration} ===")
                for k_shot, metrics in results.items():
                    print(f"{k_shot}: Error = {metrics['mean_error']:.6f} ± {metrics['std_error']:.6f}")
                print("=" * 50)
            
            # Log results
            if logger:
                logger.log_few_shot_evaluation(
                    type(model).__name__, 
                    type(model.pde).__name__, 
                    results
                )


class AdaptationProgressCallback(MetaLearningCallback):
    """Callback for monitoring adaptation progress during meta-training"""
    
    def __init__(self, log_every: int = 100, track_gradients: bool = False, 
                 verbose: bool = True):
        super().__init__(log_every=log_every, verbose=verbose)
        self.track_gradients = track_gradients
        self.adaptation_histories = []
        
    def on_task_adaptation_begin(self, task_id: str, model, task_data):
        """Called at the beginning of task adaptation"""
        if self.track_gradients:
            self.current_adaptation = {
                'task_id': task_id,
                'initial_loss': None,
                'losses': [],
                'gradient_norms': []
            }
    
    def on_adaptation_step(self, step: int, loss: float, gradients: Optional[Dict] = None):
        """Called after each adaptation step"""
        if self.track_gradients and hasattr(self, 'current_adaptation'):
            if self.current_adaptation['initial_loss'] is None:
                self.current_adaptation['initial_loss'] = loss
                
            self.current_adaptation['losses'].append(loss)
            
            if gradients:
                grad_norm = sum(torch.norm(grad).item() for grad in gradients.values())
                self.current_adaptation['gradient_norms'].append(grad_norm)
    
    def on_task_adaptation_end(self, task_id: str, final_loss: float, 
                              adaptation_time: float):
        """Called at the end of task adaptation"""
        if self.track_gradients and hasattr(self, 'current_adaptation'):
            self.current_adaptation['final_loss'] = final_loss
            self.current_adaptation['adaptation_time'] = adaptation_time
            self.adaptation_histories.append(self.current_adaptation)
            
            if self.verbose and len(self.adaptation_histories) % self.log_every == 0:
                avg_improvement = np.mean([
                    (h['initial_loss'] - h['final_loss']) / h['initial_loss'] 
                    for h in self.adaptation_histories[-self.log_every:]
                    if h['initial_loss'] > 0
                ])
                avg_time = np.mean([
                    h['adaptation_time'] 
                    for h in self.adaptation_histories[-self.log_every:]
                ])
                print(f"Adaptation Progress: Avg Improvement = {avg_improvement:.3f}, "
                      f"Avg Time = {avg_time:.3f}s")


class MemoryMonitorCallback(MetaLearningCallback):
    """Callback for monitoring memory usage during meta-training"""
    
    def __init__(self, log_every: int = 100, verbose: bool = True):
        super().__init__(log_every=log_every, verbose=verbose)
        self.memory_history = []
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        import psutil
        
        # CPU memory
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / 1024 / 1024
        
        # GPU memory
        gpu_memory_mb = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            
        return {
            'cpu_memory_mb': cpu_memory_mb,
            'gpu_memory_mb': gpu_memory_mb,
            'total_memory_mb': cpu_memory_mb + gpu_memory_mb
        }
    
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        super().on_meta_iteration_end(iteration, train_loss, val_loss, model, logger)
        
        if iteration % self.log_every == 0:
            memory_usage = self.get_memory_usage()
            memory_usage['iteration'] = iteration
            self.memory_history.append(memory_usage)
            
            if self.verbose:
                print(f"Memory Usage - CPU: {memory_usage['cpu_memory_mb']:.1f}MB, "
                      f"GPU: {memory_usage['gpu_memory_mb']:.1f}MB")
            
            if logger:
                logger.log_memory_usage(
                    f"iteration_{iteration}",
                    memory_usage['cpu_memory_mb'],
                    memory_usage['gpu_memory_mb']
                )


class MetaLearningCallbackManager:
    """Manager for coordinating multiple meta-learning callbacks"""
    
    def __init__(self, callbacks: List[MetaLearningCallback]):
        self.callbacks = callbacks
        
    def on_meta_train_begin(self, model, logger: MetaLearningLogger):
        """Notify all callbacks of meta-training start"""
        for callback in self.callbacks:
            callback.on_meta_train_begin(model, logger)
    
    def on_meta_iteration_end(self, iteration: int, train_loss: float, 
                             val_loss: Optional[float] = None, 
                             model=None, logger: Optional[MetaLearningLogger] = None):
        """Notify all callbacks of meta-iteration end"""
        for callback in self.callbacks:
            callback.on_meta_iteration_end(iteration, train_loss, val_loss, model, logger)
    
    def on_meta_train_end(self, model, logger: Optional[MetaLearningLogger] = None):
        """Notify all callbacks of meta-training end"""
        for callback in self.callbacks:
            callback.on_meta_train_end(model, logger)
    
    def should_stop_training(self) -> bool:
        """Check if any callback requests early stopping"""
        for callback in self.callbacks:
            if hasattr(callback, 'should_stop_training') and callback.should_stop_training():
                return True
        return False
    
    def on_task_adaptation_begin(self, task_id: str, model, task_data):
        """Notify callbacks of task adaptation start"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_task_adaptation_begin'):
                callback.on_task_adaptation_begin(task_id, model, task_data)
    
    def on_adaptation_step(self, step: int, loss: float, gradients: Optional[Dict] = None):
        """Notify callbacks of adaptation step"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_adaptation_step'):
                callback.on_adaptation_step(step, loss, gradients)
    
    def on_task_adaptation_end(self, task_id: str, final_loss: float, 
                              adaptation_time: float):
        """Notify callbacks of task adaptation end"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_task_adaptation_end'):
                callback.on_task_adaptation_end(task_id, final_loss, adaptation_time)