"""Meta-learning specific error handling and exception classes.

This module provides comprehensive error handling for meta-learning operations,
including physics loss computation failures, gradient computation issues,
and MAML-specific errors.
"""

import logging
import traceback
from typing import Optional, Dict, Any, Callable, Union
import torch
import numpy as np
from functools import wraps

logger = logging.getLogger(__name__)


class MetaLearningError(Exception):
    """Base exception class for meta-learning related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class PhysicsLossError(MetaLearningError):
    """Raised when physics loss computation fails in meta-learning."""
    pass


class GradientComputationError(MetaLearningError):
    """Raised when gradient computation fails in MAML."""
    pass


class AdaptationError(MetaLearningError):
    """Raised when few-shot adaptation fails."""
    pass


class MetaTrainingError(MetaLearningError):
    """Raised when meta-training encounters issues."""
    pass


class TaskDataError(MetaLearningError):
    """Raised when task data is invalid or corrupted."""
    pass


class ConstraintBalancingError(MetaLearningError):
    """Raised when adaptive constraint weighting fails."""
    pass


def safe_physics_loss_computation(func: Callable) -> Callable:
    """Decorator for safe physics loss computation with fallback mechanisms.
    
    Args:
        func: Function that computes physics loss
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "gradient" in str(e).lower() or "backward" in str(e).lower():
                logger.warning(f"Gradient computation failed in physics loss: {e}")
                # Return zero physics loss with gradient tracking
                return torch.tensor(0.0, requires_grad=True, device=args[0].device if hasattr(args[0], 'device') else 'cpu')
            elif "nan" in str(e).lower() or "inf" in str(e).lower():
                logger.warning(f"NaN/Inf detected in physics loss computation: {e}")
                # Return small positive loss to avoid training collapse
                return torch.tensor(1e-6, requires_grad=True, device=args[0].device if hasattr(args[0], 'device') else 'cpu')
            else:
                raise PhysicsLossError(f"Physics loss computation failed: {e}", 
                                     context={"function": func.__name__, "args_types": [type(arg).__name__ for arg in args]})
        except Exception as e:
            raise PhysicsLossError(f"Unexpected error in physics loss computation: {e}",
                                 context={"function": func.__name__, "error_type": type(e).__name__})
    return wrapper


def safe_gradient_computation(func: Callable) -> Callable:
    """Decorator for safe gradient computation in MAML with fallback mechanisms.
    
    Args:
        func: Function that computes gradients
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "one of the variables needed for gradient computation has been modified" in str(e):
                logger.warning("Gradient computation failed due to modified variables, attempting recovery")
                # Try to recompute with fresh parameters
                try:
                    # This is a fallback - the calling function should handle parameter refresh
                    return None
                except Exception:
                    raise GradientComputationError(f"Gradient computation failed and recovery unsuccessful: {e}",
                                                 context={"function": func.__name__})
            elif "grad can be implicitly created only for scalar outputs" in str(e):
                logger.warning("Gradient computation failed due to non-scalar output")
                # Return None to indicate failure - calling function should handle
                return None
            else:
                raise GradientComputationError(f"Gradient computation failed: {e}",
                                             context={"function": func.__name__, "error_details": str(e)})
        except Exception as e:
            raise GradientComputationError(f"Unexpected error in gradient computation: {e}",
                                         context={"function": func.__name__, "error_type": type(e).__name__})
    return wrapper


def safe_adaptation_step(func: Callable) -> Callable:
    """Decorator for safe adaptation steps with error recovery.
    
    Args:
        func: Function that performs adaptation step
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, ValueError) as e:
            if "nan" in str(e).lower() or "inf" in str(e).lower():
                logger.warning(f"NaN/Inf detected during adaptation: {e}")
                # Return previous parameters to avoid corruption
                if len(args) > 0 and hasattr(args[0], 'network'):
                    return {name: param.clone() for name, param in args[0].network.named_parameters()}
                return None
            elif "out of memory" in str(e).lower():
                logger.warning(f"Out of memory during adaptation: {e}")
                # Clear cache and return None to indicate failure
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None
            else:
                raise AdaptationError(f"Adaptation step failed: {e}",
                                    context={"function": func.__name__, "step": kwargs.get('step', 'unknown')})
        except Exception as e:
            raise AdaptationError(f"Unexpected error during adaptation: {e}",
                                context={"function": func.__name__, "error_type": type(e).__name__})
    return wrapper


def validate_task_data(task_data) -> bool:
    """Validate task data integrity.
    
    Args:
        task_data: Task data to validate
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        TaskDataError: If data is critically invalid
    """
    try:
        if task_data is None:
            raise TaskDataError("Task data is None")
        
        # Check for required attributes
        required_attrs = ['inputs', 'outputs']
        for attr in required_attrs:
            if not hasattr(task_data, attr):
                raise TaskDataError(f"Task data missing required attribute: {attr}")
            
            data = getattr(task_data, attr)
            if data is None:
                raise TaskDataError(f"Task data attribute {attr} is None")
            
            # Check for NaN/Inf values
            if torch.is_tensor(data):
                if torch.isnan(data).any():
                    logger.warning(f"NaN values detected in task data attribute: {attr}")
                    return False
                if torch.isinf(data).any():
                    logger.warning(f"Inf values detected in task data attribute: {attr}")
                    return False
            elif isinstance(data, np.ndarray):
                if np.isnan(data).any():
                    logger.warning(f"NaN values detected in task data attribute: {attr}")
                    return False
                if np.isinf(data).any():
                    logger.warning(f"Inf values detected in task data attribute: {attr}")
                    return False
        
        return True
        
    except Exception as e:
        raise TaskDataError(f"Task data validation failed: {e}")


def handle_constraint_balancing_error(func: Callable) -> Callable:
    """Decorator for handling constraint balancing errors.
    
    Args:
        func: Function that performs constraint balancing
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, ValueError) as e:
            if "singular matrix" in str(e).lower() or "not invertible" in str(e).lower():
                logger.warning(f"Constraint balancing failed due to singular matrix: {e}")
                # Return uniform weights as fallback
                num_constraints = kwargs.get('num_constraints', len(args[1]) if len(args) > 1 else 1)
                return torch.ones(num_constraints) / num_constraints
            elif "nan" in str(e).lower() or "inf" in str(e).lower():
                logger.warning(f"NaN/Inf in constraint balancing: {e}")
                # Return uniform weights
                num_constraints = kwargs.get('num_constraints', len(args[1]) if len(args) > 1 else 1)
                return torch.ones(num_constraints) / num_constraints
            else:
                raise ConstraintBalancingError(f"Constraint balancing failed: {e}",
                                             context={"function": func.__name__})
        except Exception as e:
            raise ConstraintBalancingError(f"Unexpected error in constraint balancing: {e}",
                                         context={"function": func.__name__, "error_type": type(e).__name__})
    return wrapper


class ErrorRecoveryManager:
    """Manages error recovery strategies for meta-learning operations."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.error_counts = {}
        self.recovery_strategies = {
            PhysicsLossError: self._recover_physics_loss,
            GradientComputationError: self._recover_gradient_computation,
            AdaptationError: self._recover_adaptation,
            ConstraintBalancingError: self._recover_constraint_balancing
        }
    
    def execute_with_recovery(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with automatic error recovery.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or None if all recovery attempts fail
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except MetaLearningError as e:
                last_error = e
                error_type = type(e)
                
                # Track error frequency
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                if attempt < self.max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed with {error_type.__name__}: {e}")
                    
                    # Apply recovery strategy
                    if error_type in self.recovery_strategies:
                        recovery_success = self.recovery_strategies[error_type](e, attempt)
                        if not recovery_success:
                            logger.error(f"Recovery strategy failed for {error_type.__name__}")
                            break
                    
                    # Exponential backoff
                    import time
                    time.sleep(self.backoff_factor ** attempt)
                else:
                    logger.error(f"All {self.max_retries + 1} attempts failed")
                    break
        
        # Log final failure
        if last_error:
            logger.error(f"Function {func.__name__} failed after all recovery attempts: {last_error}")
        
        return None
    
    def _recover_physics_loss(self, error: PhysicsLossError, attempt: int) -> bool:
        """Recover from physics loss computation error."""
        logger.info(f"Attempting physics loss recovery (attempt {attempt + 1})")
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return True
    
    def _recover_gradient_computation(self, error: GradientComputationError, attempt: int) -> bool:
        """Recover from gradient computation error."""
        logger.info(f"Attempting gradient computation recovery (attempt {attempt + 1})")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
    
    def _recover_adaptation(self, error: AdaptationError, attempt: int) -> bool:
        """Recover from adaptation error."""
        logger.info(f"Attempting adaptation recovery (attempt {attempt + 1})")
        
        # Clear CUDA cache and force garbage collection
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import gc
        gc.collect()
        
        return True
    
    def _recover_constraint_balancing(self, error: ConstraintBalancingError, attempt: int) -> bool:
        """Recover from constraint balancing error."""
        logger.info(f"Attempting constraint balancing recovery (attempt {attempt + 1})")
        return True
    
    def get_error_statistics(self) -> Dict[str, int]:
        """Get error occurrence statistics."""
        return {error_type.__name__: count for error_type, count in self.error_counts.items()}
    
    def reset_error_counts(self):
        """Reset error count statistics."""
        self.error_counts.clear()


# Global error recovery manager instance
error_recovery_manager = ErrorRecoveryManager()


def with_error_recovery(func: Callable) -> Callable:
    """Decorator that adds automatic error recovery to functions.
    
    Args:
        func: Function to wrap with error recovery
        
    Returns:
        Wrapped function with error recovery
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        return error_recovery_manager.execute_with_recovery(func, *args, **kwargs)
    return wrapper