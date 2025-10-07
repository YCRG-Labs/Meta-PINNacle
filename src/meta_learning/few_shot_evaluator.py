"""
Few-shot evaluation framework extending PINNacle's existing evaluation utilities.
Implements comprehensive evaluation metrics for meta-learning and transfer learning models.
"""

import time
import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from deepxde.metrics import l2_relative_error
from src.meta_learning.task import Task, TaskData
from src.meta_learning.meta_pinn import MetaPINN
from src.meta_learning.physics_informed_meta_learner import PhysicsInformedMetaLearner
from src.meta_learning.transfer_learning_pinn import TransferLearningPINN

logger = logging.getLogger(__name__)


@dataclass
class FewShotResults:
    """Results container for few-shot evaluation"""
    model_name: str
    shot_results: Dict[int, Dict[str, Any]]  # K -> metrics
    total_evaluation_time: float
    metadata: Dict[str, Any]


class FewShotEvaluator:
    """
    Comprehensive few-shot evaluation framework extending PINNacle's existing utilities.
    
    Evaluates meta-learning and transfer learning models on few-shot adaptation tasks
    with K ∈ {1, 5, 10, 25} shots, measuring L2 relative error and timing metrics.
    """
    
    def __init__(self, 
                 evaluation_shots: List[int] = None,
                 adaptation_steps: List[int] = None,
                 device: str = 'cuda',
                 verbose: bool = True):
        """
        Initialize FewShotEvaluator.
        
        Args:
            evaluation_shots: List of K values for K-shot evaluation [1, 5, 10, 25]
            adaptation_steps: List of adaptation steps for meta-learning models
            device: Device for computation ('cuda' or 'cpu')
            verbose: Whether to print evaluation progress
        """
        self.evaluation_shots = evaluation_shots or [1, 5, 10, 25]
        self.adaptation_steps = adaptation_steps or [1, 5, 10, 25]
        self.device = device
        self.verbose = verbose
        
        # Initialize timing tracking
        self.timing_history = {}
        
        if self.verbose:
            logger.info(f"FewShotEvaluator initialized with shots: {self.evaluation_shots}")
    
    def evaluate_few_shot_performance(self, 
                                    model: Any,
                                    test_tasks: List[Task],
                                    model_name: str = None) -> FewShotResults:
        """
        Evaluate few-shot performance for K ∈ {1, 5, 10, 25} shots.
        
        Extends existing PINNacle evaluation utilities with meta-learning specific metrics.
        
        Args:
            model: Model to evaluate (MetaPINN, TransferLearningPINN, or StandardPINN)
            test_tasks: List of test tasks for evaluation
            model_name: Name identifier for the model
            
        Returns:
            FewShotResults containing comprehensive evaluation metrics
        """
        if model_name is None:
            model_name = type(model).__name__
            
        if self.verbose:
            logger.info(f"Starting few-shot evaluation for {model_name} on {len(test_tasks)} tasks")
        
        start_time = time.time()
        shot_results = {}
        
        for K in self.evaluation_shots:
            if self.verbose:
                logger.info(f"Evaluating {K}-shot performance...")
            
            shot_metrics = self._evaluate_k_shot(model, test_tasks, K)
            shot_results[K] = shot_metrics
            
            if self.verbose:
                logger.info(f"{K}-shot results: L2 Error = {shot_metrics['mean_l2_error']:.6f} ± {shot_metrics['std_l2_error']:.6f}")
        
        total_time = time.time() - start_time
        
        # Compile metadata
        metadata = {
            'num_test_tasks': len(test_tasks),
            'evaluation_shots': self.evaluation_shots,
            'device': self.device,
            'model_type': type(model).__name__
        }
        
        return FewShotResults(
            model_name=model_name,
            shot_results=shot_results,
            total_evaluation_time=total_time,
            metadata=metadata
        )
    
    def _evaluate_k_shot(self, model: Any, test_tasks: List[Task], K: int) -> Dict[str, Any]:
        """
        Evaluate model performance with K support samples per task.
        
        Args:
            model: Model to evaluate
            test_tasks: List of test tasks
            K: Number of support samples
            
        Returns:
            Dictionary containing evaluation metrics for K-shot performance
        """
        l2_errors = []
        adaptation_times = []
        inference_times = []
        
        for task_idx, task in enumerate(test_tasks):
            try:
                # Sample K support points from task
                support_data = self._sample_support_data(task, K)
                
                # Measure adaptation time
                adapt_start = time.time()
                
                if isinstance(model, (MetaPINN, PhysicsInformedMetaLearner)):
                    # Meta-learning adaptation
                    adapted_model = self._adapt_meta_model(model, support_data)
                elif isinstance(model, TransferLearningPINN):
                    # Transfer learning fine-tuning
                    adapted_model = self._finetune_transfer_model(model, support_data)
                else:
                    # Standard PINN training from scratch
                    adapted_model = self._train_standard_model(model, support_data)
                
                adaptation_time = time.time() - adapt_start
                
                # Measure inference time and compute error
                inference_start = time.time()
                l2_error = self._compute_l2_relative_error(adapted_model, task.query_data)
                inference_time = time.time() - inference_start
                
                l2_errors.append(l2_error)
                adaptation_times.append(adaptation_time)
                inference_times.append(inference_time)
                
            except Exception as e:
                logger.warning(f"Error evaluating task {task_idx}: {e}")
                # Use NaN for failed evaluations
                l2_errors.append(np.nan)
                adaptation_times.append(np.nan)
                inference_times.append(np.nan)
        
        # Filter out NaN values for statistics
        valid_errors = np.array(l2_errors)[~np.isnan(l2_errors)]
        valid_adapt_times = np.array(adaptation_times)[~np.isnan(adaptation_times)]
        valid_inference_times = np.array(inference_times)[~np.isnan(inference_times)]
        
        return {
            'mean_l2_error': np.mean(valid_errors) if len(valid_errors) > 0 else np.nan,
            'std_l2_error': np.std(valid_errors) if len(valid_errors) > 0 else np.nan,
            'median_l2_error': np.median(valid_errors) if len(valid_errors) > 0 else np.nan,
            'mean_adaptation_time': np.mean(valid_adapt_times) if len(valid_adapt_times) > 0 else np.nan,
            'std_adaptation_time': np.std(valid_adapt_times) if len(valid_adapt_times) > 0 else np.nan,
            'mean_inference_time': np.mean(valid_inference_times) if len(valid_inference_times) > 0 else np.nan,
            'std_inference_time': np.std(valid_inference_times) if len(valid_inference_times) > 0 else np.nan,
            'success_rate': len(valid_errors) / len(test_tasks),
            'raw_l2_errors': l2_errors,
            'raw_adaptation_times': adaptation_times,
            'raw_inference_times': inference_times
        }
    
    def _sample_support_data(self, task: Task, K: int) -> TaskData:
        """
        Sample K support points from task data.
        
        Args:
            task: Task to sample from
            K: Number of support samples
            
        Returns:
            TaskData with K sampled points
        """
        if hasattr(task, 'sample_support'):
            return task.sample_support(K)
        
        # Fallback implementation
        support_size = len(task.support_data.inputs)
        if K >= support_size:
            return task.support_data
        
        indices = np.random.choice(support_size, K, replace=False)
        
        return TaskData(
            inputs=task.support_data.inputs[indices],
            outputs=task.support_data.outputs[indices],
            collocation_points=task.support_data.collocation_points[indices] if task.support_data.collocation_points is not None else None,
            boundary_data=task.support_data.boundary_data[indices] if task.support_data.boundary_data is not None else None,
            initial_data=task.support_data.initial_data[indices] if task.support_data.initial_data is not None else None
        )
    
    def _adapt_meta_model(self, model: Any, support_data: TaskData) -> Any:
        """
        Adapt meta-learning model to new task using few-shot data.
        
        Args:
            model: Meta-learning model (MetaPINN or PhysicsInformedMetaLearner)
            support_data: Support data for adaptation
            
        Returns:
            Adapted model
        """
        if hasattr(model, 'adapt'):
            # Use model's built-in adaptation method
            return model.adapt(support_data, steps=5)  # Default 5 adaptation steps
        elif hasattr(model, 'adapt_to_task'):
            # Alternative adaptation interface
            return model.adapt_to_task(support_data)
        else:
            logger.warning(f"Model {type(model).__name__} does not have adaptation method")
            return model
    
    def _finetune_transfer_model(self, model: TransferLearningPINN, support_data: TaskData) -> TransferLearningPINN:
        """
        Fine-tune transfer learning model on support data.
        
        Args:
            model: Transfer learning model
            support_data: Support data for fine-tuning
            
        Returns:
            Fine-tuned model
        """
        if hasattr(model, 'fine_tune'):
            return model.fine_tune(support_data, epochs=100)  # Default 100 epochs
        else:
            logger.warning(f"Model {type(model).__name__} does not have fine_tune method")
            return model
    
    def _train_standard_model(self, model: Any, support_data: TaskData) -> Any:
        """
        Train standard PINN from scratch on support data.
        
        Args:
            model: Standard PINN model
            support_data: Support data for training
            
        Returns:
            Trained model
        """
        if hasattr(model, 'train_from_scratch'):
            return model.train_from_scratch(support_data, epochs=5000)  # Default 5000 epochs
        elif hasattr(model, 'train'):
            return model.train(support_data, epochs=5000)
        else:
            logger.warning(f"Model {type(model).__name__} does not have training method")
            return model
    
    def _compute_l2_relative_error(self, model: Any, query_data: TaskData) -> float:
        """
        Compute L2 relative error using existing PINNacle metrics.
        
        Args:
            model: Trained/adapted model
            query_data: Query data for evaluation
            
        Returns:
            L2 relative error
        """
        try:
            # Get model predictions
            if hasattr(model, 'predict'):
                y_pred = model.predict(query_data.inputs)
            elif hasattr(model, 'forward'):
                with torch.no_grad():
                    if isinstance(query_data.inputs, np.ndarray):
                        inputs = torch.tensor(query_data.inputs, dtype=torch.float32, device=self.device)
                    else:
                        inputs = query_data.inputs.to(self.device)
                    y_pred = model.forward(inputs).cpu().numpy()
            else:
                raise ValueError(f"Model {type(model).__name__} has no prediction method")
            
            # Convert query outputs to numpy if needed
            if isinstance(query_data.outputs, torch.Tensor):
                y_true = query_data.outputs.cpu().numpy()
            else:
                y_true = query_data.outputs
            
            # Use existing PINNacle L2 relative error metric
            return l2_relative_error(y_true, y_pred)
            
        except Exception as e:
            logger.error(f"Error computing L2 relative error: {e}")
            return np.nan
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """
        Get summary of timing measurements integrated with existing logging.
        
        Returns:
            Dictionary containing timing statistics
        """
        return {
            'timing_history': self.timing_history,
            'total_evaluations': sum(len(times) for times in self.timing_history.values()),
            'average_evaluation_time': np.mean([
                np.mean(times) for times in self.timing_history.values() 
                if len(times) > 0
            ]) if self.timing_history else 0.0
        }
    
    def reset_timing(self):
        """Reset timing measurements."""
        self.timing_history.clear()