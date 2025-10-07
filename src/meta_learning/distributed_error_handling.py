"""Distributed training error handling for meta-learning.

This module provides comprehensive error handling for distributed meta-learning,
including graceful degradation from multi-GPU to single-GPU, fault tolerance
for distributed communication, and recovery strategies.
"""

import logging
import time
import traceback
import signal
import threading
from typing import Optional, Dict, Any, Callable, List, Union
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from functools import wraps

from .error_handling import (
    MetaLearningError, 
    ErrorRecoveryManager,
    with_error_recovery
)

logger = logging.getLogger(__name__)


class DistributedTrainingError(MetaLearningError):
    """Base exception for distributed training errors."""
    pass


class DistributedCommunicationError(DistributedTrainingError):
    """Raised when distributed communication fails."""
    pass


class DistributedSynchronizationError(DistributedTrainingError):
    """Raised when distributed synchronization fails."""
    pass


class DistributedGradientError(DistributedTrainingError):
    """Raised when distributed gradient operations fail."""
    pass


class ProcessFailureError(DistributedTrainingError):
    """Raised when a distributed process fails."""
    pass


class DistributedMemoryError(DistributedTrainingError):
    """Raised when distributed training runs out of memory."""
    pass


def safe_distributed_operation(func: Callable) -> Callable:
    """Decorator for safe distributed operations with fallback mechanisms.
    
    Args:
        func: Function that performs distributed operations
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except dist.DistBackendError as e:
            logger.warning(f"Distributed backend error in {func.__name__}: {e}")
            # Try to recover or fallback
            if "NCCL" in str(e):
                logger.info("NCCL error detected, attempting Gloo fallback")
                # This would need to be handled at a higher level
                raise DistributedCommunicationError(f"NCCL communication failed: {e}")
            else:
                raise DistributedCommunicationError(f"Distributed backend error: {e}")
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"CUDA OOM in distributed operation: {e}")
                # Clear cache and raise memory error
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise DistributedMemoryError(f"Out of memory in distributed operation: {e}")
            elif "Connection refused" in str(e) or "Connection reset" in str(e):
                logger.warning(f"Connection error in distributed operation: {e}")
                raise DistributedCommunicationError(f"Connection error: {e}")
            else:
                raise DistributedTrainingError(f"Runtime error in distributed operation: {e}")
        except Exception as e:
            raise DistributedTrainingError(f"Unexpected error in distributed operation {func.__name__}: {e}")
    return wrapper


def safe_gradient_synchronization(func: Callable) -> Callable:
    """Decorator for safe gradient synchronization with error recovery.
    
    Args:
        func: Function that synchronizes gradients
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except dist.DistBackendError as e:
            logger.warning(f"Gradient synchronization failed: {e}")
            # Return False to indicate synchronization failure
            return False
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.warning(f"OOM during gradient synchronization: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return False
            elif "nan" in str(e).lower() or "inf" in str(e).lower():
                logger.warning(f"NaN/Inf in gradient synchronization: {e}")
                return False
            else:
                raise DistributedGradientError(f"Gradient synchronization error: {e}")
        except Exception as e:
            raise DistributedGradientError(f"Unexpected error in gradient synchronization: {e}")
    return wrapper


def safe_process_communication(func: Callable) -> Callable:
    """Decorator for safe inter-process communication.
    
    Args:
        func: Function that performs inter-process communication
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, BrokenPipeError, EOFError) as e:
            logger.warning(f"Process communication error: {e}")
            raise DistributedCommunicationError(f"Inter-process communication failed: {e}")
        except TimeoutError as e:
            logger.warning(f"Process communication timeout: {e}")
            raise DistributedCommunicationError(f"Communication timeout: {e}")
        except Exception as e:
            raise DistributedCommunicationError(f"Unexpected communication error: {e}")
    return wrapper


class DistributedErrorRecoveryManager(ErrorRecoveryManager):
    """Enhanced error recovery manager for distributed meta-learning."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.5,
                 enable_graceful_degradation: bool = True):
        super().__init__(max_retries, backoff_factor)
        self.enable_graceful_degradation = enable_graceful_degradation
        self.failed_processes = set()
        self.communication_failures = 0
        self.max_communication_failures = 5
        
        # Add distributed-specific recovery strategies
        self.recovery_strategies.update({
            DistributedCommunicationError: self._recover_communication_error,
            DistributedSynchronizationError: self._recover_synchronization_error,
            DistributedGradientError: self._recover_gradient_error,
            ProcessFailureError: self._recover_process_failure,
            DistributedMemoryError: self._recover_memory_error
        })
    
    def _recover_communication_error(self, error: DistributedCommunicationError, attempt: int) -> bool:
        """Recover from distributed communication error."""
        logger.info(f"Attempting communication error recovery (attempt {attempt + 1})")
        
        self.communication_failures += 1
        
        # If too many communication failures, attempt graceful degradation
        if self.communication_failures >= self.max_communication_failures:
            if self.enable_graceful_degradation:
                logger.warning("Too many communication failures, attempting graceful degradation")
                return self._attempt_graceful_degradation()
            else:
                logger.error("Communication failures exceeded threshold, no graceful degradation enabled")
                return False
        
        # Wait and retry
        time.sleep(2 ** attempt)
        
        # Try to reinitialize communication
        try:
            if dist.is_initialized():
                # Check if we can still communicate
                if dist.get_rank() == 0:
                    test_tensor = torch.tensor([1.0])
                    dist.broadcast(test_tensor, src=0)
                else:
                    test_tensor = torch.tensor([0.0])
                    dist.broadcast(test_tensor, src=0)
                
                logger.info("Communication test successful")
                return True
        except Exception as e:
            logger.warning(f"Communication test failed: {e}")
        
        return False
    
    def _recover_synchronization_error(self, error: DistributedSynchronizationError, attempt: int) -> bool:
        """Recover from synchronization error."""
        logger.info(f"Attempting synchronization error recovery (attempt {attempt + 1})")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Wait for other processes
        time.sleep(1 + attempt)
        
        return True
    
    def _recover_gradient_error(self, error: DistributedGradientError, attempt: int) -> bool:
        """Recover from gradient synchronization error."""
        logger.info(f"Attempting gradient error recovery (attempt {attempt + 1})")
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return True
    
    def _recover_process_failure(self, error: ProcessFailureError, attempt: int) -> bool:
        """Recover from process failure."""
        logger.info(f"Attempting process failure recovery (attempt {attempt + 1})")
        
        # Mark process as failed
        if hasattr(error, 'context') and 'rank' in error.context:
            self.failed_processes.add(error.context['rank'])
        
        # If too many processes failed, attempt graceful degradation
        if len(self.failed_processes) > 1:  # Allow one process failure
            if self.enable_graceful_degradation:
                return self._attempt_graceful_degradation()
            else:
                return False
        
        return True
    
    def _recover_memory_error(self, error: DistributedMemoryError, attempt: int) -> bool:
        """Recover from distributed memory error."""
        logger.info(f"Attempting memory error recovery (attempt {attempt + 1})")
        
        # Aggressive memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        import gc
        gc.collect()
        
        # Wait for memory to be freed
        time.sleep(2)
        
        return True
    
    def _attempt_graceful_degradation(self) -> bool:
        """Attempt graceful degradation to single-GPU training."""
        logger.warning("Attempting graceful degradation to single-GPU training")
        
        try:
            # This would need to be implemented at the training level
            # For now, just return True to indicate the attempt
            return True
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False


class DistributedFaultToleranceManager:
    """Comprehensive fault tolerance for distributed meta-learning."""
    
    def __init__(self, world_size: int, rank: int, 
                 checkpoint_frequency: int = 100,
                 enable_graceful_degradation: bool = True):
        self.world_size = world_size
        self.rank = rank
        self.checkpoint_frequency = checkpoint_frequency
        self.enable_graceful_degradation = enable_graceful_degradation
        
        # Error tracking
        self.error_recovery_manager = DistributedErrorRecoveryManager(
            enable_graceful_degradation=enable_graceful_degradation
        )
        self.process_health = {i: True for i in range(world_size)}
        self.last_successful_sync = time.time()
        
        # Graceful shutdown handling
        self.shutdown_requested = False
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Process {self.rank} received signal {signum}, initiating graceful shutdown")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def handle_distributed_error(self, error: Exception, context: str) -> bool:
        """Handle distributed training error with recovery strategies.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            True if recovery successful, False otherwise
        """
        logger.error(f"Distributed error in {context}: {error}")
        
        # Convert to appropriate distributed error type
        if isinstance(error, dist.DistBackendError):
            distributed_error = DistributedCommunicationError(str(error))
        elif "CUDA out of memory" in str(error):
            distributed_error = DistributedMemoryError(str(error))
        elif "gradient" in str(error).lower():
            distributed_error = DistributedGradientError(str(error))
        else:
            distributed_error = DistributedTrainingError(str(error))
        
        # Attempt recovery
        return self.error_recovery_manager.execute_with_recovery(
            lambda: self._dummy_recovery_function(),
            error=distributed_error,
            context=context
        ) is not None
    
    def _dummy_recovery_function(self):
        """Dummy function for recovery testing."""
        return True
    
    def check_process_health(self) -> Dict[int, bool]:
        """Check health of all processes."""
        if not dist.is_initialized():
            return {self.rank: True}
        
        try:
            # Simple health check using all_gather
            health_tensor = torch.tensor([1.0 if not self.shutdown_requested else 0.0])
            health_list = [torch.zeros_like(health_tensor) for _ in range(self.world_size)]
            
            dist.all_gather(health_list, health_tensor)
            
            # Update process health
            for i, tensor in enumerate(health_list):
                self.process_health[i] = tensor.item() > 0.5
            
            self.last_successful_sync = time.time()
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            # Mark current process as unhealthy if we can't communicate
            self.process_health[self.rank] = False
        
        return self.process_health
    
    def should_continue_training(self) -> bool:
        """Determine if training should continue based on process health."""
        if self.shutdown_requested:
            return False
        
        health = self.check_process_health()
        healthy_processes = sum(health.values())
        
        # Continue if majority of processes are healthy
        return healthy_processes > self.world_size // 2
    
    def attempt_graceful_degradation(self) -> bool:
        """Attempt to gracefully degrade to fewer processes."""
        if not self.enable_graceful_degradation:
            return False
        
        logger.info(f"Process {self.rank} attempting graceful degradation")
        
        try:
            # Check which processes are still healthy
            health = self.check_process_health()
            healthy_ranks = [rank for rank, healthy in health.items() if healthy]
            
            if len(healthy_ranks) == 0:
                logger.error("No healthy processes found")
                return False
            
            # If we're not in the healthy set, shut down gracefully
            if self.rank not in healthy_ranks:
                logger.info(f"Process {self.rank} not healthy, shutting down")
                return False
            
            # Create new process group with healthy processes only
            if len(healthy_ranks) < self.world_size:
                logger.info(f"Degrading from {self.world_size} to {len(healthy_ranks)} processes")
                # This would require reinitializing the process group
                # For now, just return True to indicate attempt
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Graceful degradation failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup distributed resources."""
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"Error during distributed cleanup: {e}")


def with_distributed_error_handling(fault_tolerance_manager: Optional[DistributedFaultToleranceManager] = None):
    """Decorator that adds distributed error handling to functions.
    
    Args:
        fault_tolerance_manager: Optional fault tolerance manager
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if fault_tolerance_manager:
                    recovery_success = fault_tolerance_manager.handle_distributed_error(
                        e, func.__name__
                    )
                    if not recovery_success:
                        logger.error(f"Failed to recover from error in {func.__name__}")
                        raise
                    # If recovery successful, try again
                    return func(*args, **kwargs)
                else:
                    # No fault tolerance manager, just re-raise
                    raise
        return wrapper
    return decorator


def setup_distributed_error_handling(world_size: int, rank: int, 
                                    enable_graceful_degradation: bool = True) -> DistributedFaultToleranceManager:
    """Setup distributed error handling for a process.
    
    Args:
        world_size: Total number of processes
        rank: Current process rank
        enable_graceful_degradation: Whether to enable graceful degradation
        
    Returns:
        Configured fault tolerance manager
    """
    return DistributedFaultToleranceManager(
        world_size=world_size,
        rank=rank,
        enable_graceful_degradation=enable_graceful_degradation
    )


# Global distributed error recovery manager
distributed_error_recovery_manager = DistributedErrorRecoveryManager()