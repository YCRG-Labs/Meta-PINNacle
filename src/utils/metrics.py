"""
L2 error computation system for meta-learning PINNs evaluation.
Provides comprehensive error metrics including L2 relative error, MAE, and RMSE
for replacing accuracy-based metrics in the paper revision.
"""

import numpy as np
import torch
from typing import Dict, List, Any, Union, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class L2ErrorCalculator:
    """
    Computes L2 relative errors from model predictions for paper revision.
    
    Integrates with existing evaluation framework to replace accuracy metrics
    with proper error metrics for PDE solvers.
    """
    
    def __init__(self, reference_solutions_path: Optional[str] = None):
        """
        Initialize L2ErrorCalculator.
        
        Args:
            reference_solutions_path: Path to reference solutions data
        """
        self.reference_solutions_path = reference_solutions_path
        self.reference_data = {}
        
        if reference_solutions_path and Path(reference_solutions_path).exists():
            self.reference_data = self.load_reference_solutions(reference_solutions_path)
            logger.info(f"Loaded reference solutions from {reference_solutions_path}")
    
    def compute_l2_relative_error(self, 
                                 predictions: Union[np.ndarray, torch.Tensor], 
                                 true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute L2 relative error: ||u_pred - u_true||_L2 / ||u_true||_L2
        
        This is the standard metric for PDE solver evaluation, replacing
        meaningless accuracy percentages in the paper.
        
        Args:
            predictions: Model predictions
            true_solutions: Ground truth solutions
            
        Returns:
            L2 relative error (lower is better)
        """
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(true_solutions, torch.Tensor):
            true_solutions = true_solutions.detach().cpu().numpy()
        
        # Ensure same shape
        predictions = np.asarray(predictions)
        true_solutions = np.asarray(true_solutions)
        
        if predictions.shape != true_solutions.shape:
            raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs true {true_solutions.shape}")
        
        # Compute L2 relative error
        numerator = np.linalg.norm(predictions - true_solutions)
        denominator = np.linalg.norm(true_solutions)
        
        if denominator == 0:
            logger.warning("True solution has zero norm, returning infinity")
            return float('inf')
        
        return numerator / denominator
    
    def compute_mae(self, 
                   predictions: Union[np.ndarray, torch.Tensor], 
                   true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute Mean Absolute Error.
        
        Args:
            predictions: Model predictions
            true_solutions: Ground truth solutions
            
        Returns:
            Mean absolute error
        """
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(true_solutions, torch.Tensor):
            true_solutions = true_solutions.detach().cpu().numpy()
        
        return np.mean(np.abs(predictions - true_solutions))
    
    def compute_rmse(self, 
                    predictions: Union[np.ndarray, torch.Tensor], 
                    true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute Root Mean Square Error.
        
        Args:
            predictions: Model predictions
            true_solutions: Ground truth solutions
            
        Returns:
            Root mean square error
        """
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(true_solutions, torch.Tensor):
            true_solutions = true_solutions.detach().cpu().numpy()
        
        return np.sqrt(np.mean((predictions - true_solutions)**2))
    
    def compute_max_error(self, 
                         predictions: Union[np.ndarray, torch.Tensor], 
                         true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Compute maximum absolute error.
        
        Args:
            predictions: Model predictions
            true_solutions: Ground truth solutions
            
        Returns:
            Maximum absolute error
        """
        # Convert to numpy arrays if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(true_solutions, torch.Tensor):
            true_solutions = true_solutions.detach().cpu().numpy()
        
        return np.max(np.abs(predictions - true_solutions))
    
    def compute_all_errors(self, 
                          predictions: Union[np.ndarray, torch.Tensor], 
                          true_solutions: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
        """
        Compute all error metrics at once.
        
        Args:
            predictions: Model predictions
            true_solutions: Ground truth solutions
            
        Returns:
            Dictionary containing all error metrics
        """
        return {
            'l2_relative': self.compute_l2_relative_error(predictions, true_solutions),
            'mae': self.compute_mae(predictions, true_solutions),
            'rmse': self.compute_rmse(predictions, true_solutions),
            'max_error': self.compute_max_error(predictions, true_solutions)
        }
    
    def batch_compute_errors(self, 
                           model_predictions: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]],
                           reference_solutions: Optional[Dict[str, Union[np.ndarray, torch.Tensor]]] = None) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Compute errors for all models and problems in batch.
        
        This method processes saved model predictions and computes L2 relative errors
        to replace accuracy metrics throughout the paper.
        
        Args:
            model_predictions: Dictionary mapping model_name -> problem_name -> predictions
            reference_solutions: Optional reference solutions. If None, uses loaded reference data
            
        Returns:
            Dictionary mapping model_name -> problem_name -> error_metrics
        """
        if reference_solutions is None:
            reference_solutions = self.reference_data
        
        if not reference_solutions:
            raise ValueError("No reference solutions available. Provide reference_solutions or load from file.")
        
        results = {}
        
        for model_name, model_preds in model_predictions.items():
            results[model_name] = {}
            logger.info(f"Computing errors for model: {model_name}")
            
            for problem_name, predictions in model_preds.items():
                if problem_name not in reference_solutions:
                    logger.warning(f"No reference solution for problem: {problem_name}")
                    continue
                
                try:
                    true_solutions = reference_solutions[problem_name]
                    error_metrics = self.compute_all_errors(predictions, true_solutions)
                    results[model_name][problem_name] = error_metrics
                    
                    logger.debug(f"  {problem_name}: L2 error = {error_metrics['l2_relative']:.6f}")
                    
                except Exception as e:
                    logger.error(f"Error computing metrics for {model_name}/{problem_name}: {e}")
                    results[model_name][problem_name] = {
                        'l2_relative': np.nan,
                        'mae': np.nan,
                        'rmse': np.nan,
                        'max_error': np.nan
                    }
        
        return results
    
    def load_reference_solutions(self, reference_path: str) -> Dict[str, np.ndarray]:
        """
        Load reference solutions from file.
        
        Args:
            reference_path: Path to reference solutions
            
        Returns:
            Dictionary mapping problem names to reference solutions
        """
        reference_data = {}
        
        try:
            if reference_path.endswith('.npz'):
                # Load from numpy archive
                data = np.load(reference_path)
                for key in data.files:
                    reference_data[key] = data[key]
            elif reference_path.endswith('.pkl'):
                # Load from pickle
                import pickle
                with open(reference_path, 'rb') as f:
                    reference_data = pickle.load(f)
            else:
                logger.warning(f"Unsupported reference file format: {reference_path}")
                
        except Exception as e:
            logger.error(f"Error loading reference solutions: {e}")
        
        return reference_data
    
    def save_error_results(self, 
                          error_results: Dict[str, Dict[str, Dict[str, float]]], 
                          output_path: str):
        """
        Save computed error results to file.
        
        Args:
            error_results: Error results from batch_compute_errors
            output_path: Path to save results
        """
        try:
            if output_path.endswith('.json'):
                import json
                with open(output_path, 'w') as f:
                    json.dump(error_results, f, indent=2, default=str)
            elif output_path.endswith('.pkl'):
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(error_results, f)
            else:
                # Default to pickle
                import pickle
                with open(output_path + '.pkl', 'wb') as f:
                    pickle.dump(error_results, f)
                    
            logger.info(f"Error results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")


def compute_l2_relative_error(predictions: Union[np.ndarray, torch.Tensor], 
                             true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Standalone function to compute L2 relative error.
    
    Convenience function for direct use without L2ErrorCalculator class.
    
    Args:
        predictions: Model predictions
        true_solutions: Ground truth solutions
        
    Returns:
        L2 relative error
    """
    calculator = L2ErrorCalculator()
    return calculator.compute_l2_relative_error(predictions, true_solutions)


def batch_compute_errors(model_predictions: Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]],
                        reference_solutions: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Standalone function to compute errors for multiple models and problems.
    
    Args:
        model_predictions: Dictionary mapping model_name -> problem_name -> predictions
        reference_solutions: Dictionary mapping problem_name -> reference_solutions
        
    Returns:
        Dictionary mapping model_name -> problem_name -> error_metrics
    """
    calculator = L2ErrorCalculator()
    return calculator.batch_compute_errors(model_predictions, reference_solutions)


# Compatibility functions for existing code
def compute_mae(predictions: Union[np.ndarray, torch.Tensor], 
               true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Mean Absolute Error."""
    calculator = L2ErrorCalculator()
    return calculator.compute_mae(predictions, true_solutions)


def compute_rmse(predictions: Union[np.ndarray, torch.Tensor], 
                true_solutions: Union[np.ndarray, torch.Tensor]) -> float:
    """Compute Root Mean Square Error."""
    calculator = L2ErrorCalculator()
    return calculator.compute_rmse(predictions, true_solutions)


def compute_all_errors(predictions: Union[np.ndarray, torch.Tensor], 
                      true_solutions: Union[np.ndarray, torch.Tensor]) -> Dict[str, float]:
    """Compute all error metrics."""
    calculator = L2ErrorCalculator()
    return calculator.compute_all_errors(predictions, true_solutions)