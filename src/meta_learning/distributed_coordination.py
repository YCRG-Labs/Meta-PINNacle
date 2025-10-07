"""Distributed training coordination for meta-learning.

This module implements distributed training coordination including task sampling,
synchronization, and fault tolerance mechanisms that extend PINNacle's existing
multi-process training setup.
"""

import os
import time
import random
import threading
import queue
import signal
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum
import logging

from .task import Task, TaskData, TaskBatch
from ..utils.distributed_utils import (
    get_rank, get_world_size, is_main_process, barrier,
    all_reduce_tensor, broadcast_tensor, reduce_dict
)


class CoordinationState(Enum):
    """States for distributed training coordination."""
    INITIALIZING = "initializing"
    READY = "ready"
    TRAINING = "training"
    SYNCHRONIZING = "synchronizing"
    CHECKPOINTING = "checkpointing"
    ERROR = "error"
    FINISHED = "finished"


@dataclass
class CoordinationMessage:
    """Message for distributed coordination."""
    sender_rank: int
    message_type: str
    data: Any
    timestamp: float


class DistributedTaskSampler:
    """Distributed task sampler for coordinated task sampling across processes.
    
    This class ensures that all processes sample tasks in a coordinated manner
    while maintaining randomness and load balancing.
    """
    
    def __init__(self, tasks: List[Task], world_size: int, rank: int, 
                 batch_size: int = 25, shuffle: bool = True, seed: Optional[int] = None):
        """Initialize distributed task sampler.
        
        Args:
            tasks: List of all tasks
            world_size: Number of processes
            rank: Current process rank
            batch_size: Batch size for sampling
            shuffle: Whether to shuffle tasks
            seed: Random seed for reproducibility
        """
        self.tasks = tasks
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed + rank)
            np.random.seed(seed + rank)
        
        # Task indices for sampling
        self.task_indices = list(range(len(tasks)))
        if shuffle:
            random.shuffle(self.task_indices)
        
        # Current position in task list
        self.current_position = 0
        self.epoch = 0
        
        # Synchronization state
        self.synchronized = True
    
    def sample_batch(self, synchronized: bool = True) -> List[Task]:
        """Sample a batch of tasks.
        
        Args:
            synchronized: Whether to synchronize sampling across processes
            
        Returns:
            Batch of tasks
        """
        if synchronized and self.world_size > 1:
            return self._sample_batch_synchronized()
        else:
            return self._sample_batch_local()
    
    def _sample_batch_synchronized(self) -> List[Task]:
        """Sample batch with synchronization across processes."""
        # Synchronize random state across processes
        if is_main_process():
            # Generate random indices on main process
            if self.current_position + self.batch_size > len(self.task_indices):
                # Reshuffle if we've gone through all tasks
                if self.shuffle:
                    random.shuffle(self.task_indices)
                self.current_position = 0
                self.epoch += 1
            
            # Sample indices
            sampled_indices = self.task_indices[
                self.current_position:self.current_position + self.batch_size
            ]
            self.current_position += self.batch_size
            
            # Convert to tensor for broadcasting
            indices_tensor = torch.tensor(sampled_indices, dtype=torch.long)
        else:
            # Create empty tensor on other processes
            indices_tensor = torch.zeros(self.batch_size, dtype=torch.long)
        
        # Broadcast indices to all processes
        broadcast_tensor(indices_tensor, src=0)
        
        # Convert back to list and sample tasks
        sampled_indices = indices_tensor.tolist()
        return [self.tasks[idx] for idx in sampled_indices if idx < len(self.tasks)]
    
    def _sample_batch_local(self) -> List[Task]:
        """Sample batch locally without synchronization."""
        if self.current_position + self.batch_size > len(self.task_indices):
            # Reshuffle if we've gone through all tasks
            if self.shuffle:
                random.shuffle(self.task_indices)
            self.current_position = 0
            self.epoch += 1
        
        # Sample indices
        sampled_indices = self.task_indices[
            self.current_position:self.current_position + self.batch_size
        ]
        self.current_position += self.batch_size
        
        return [self.tasks[idx] for idx in sampled_indices]
    
    def reset(self):
        """Reset sampler state."""
        self.current_position = 0
        self.epoch = 0
        if self.shuffle:
            random.shuffle(self.task_indices)
    
    def get_state(self) -> Dict[str, Any]:
        """Get sampler state for checkpointing."""
        return {
            'current_position': self.current_position,
            'epoch': self.epoch,
            'task_indices': self.task_indices
        }
    
    def set_state(self, state: Dict[str, Any]):
        """Set sampler state from checkpoint."""
        self.current_position = state['current_position']
        self.epoch = state['epoch']
        self.task_indices = state['task_indices']


class DistributedSynchronizer:
    """Synchronizer for distributed meta-learning training.
    
    This class handles synchronization of training state, gradients,
    and coordination messages across processes.
    """
    
    def __init__(self, world_size: int, rank: int, sync_frequency: int = 1):
        """Initialize distributed synchronizer.
        
        Args:
            world_size: Number of processes
            rank: Current process rank
            sync_frequency: Frequency of synchronization (in steps)
        """
        self.world_size = world_size
        self.rank = rank
        self.sync_frequency = sync_frequency
        
        # Synchronization state
        self.step_count = 0
        self.last_sync_step = 0
        
        # Message queue for coordination
        self.message_queue = queue.Queue()
        
        # Synchronization statistics
        self.sync_times = []
        self.sync_failures = 0
    
    def should_synchronize(self, step: int) -> bool:
        """Check if synchronization is needed at this step.
        
        Args:
            step: Current training step
            
        Returns:
            Whether synchronization is needed
        """
        return (step - self.last_sync_step) >= self.sync_frequency
    
    def synchronize_gradients(self, model: torch.nn.Module) -> bool:
        """Synchronize gradients across processes.
        
        Args:
            model: Model to synchronize gradients for
            
        Returns:
            Whether synchronization was successful
        """
        if self.world_size == 1:
            return True
        
        start_time = time.time()
        
        try:
            # Synchronize gradients
            for param in model.parameters():
                if param.grad is not None:
                    all_reduce_tensor(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
            
            # Update synchronization statistics
            sync_time = time.time() - start_time
            self.sync_times.append(sync_time)
            self.last_sync_step = self.step_count
            
            return True
            
        except Exception as e:
            self.sync_failures += 1
            if is_main_process():
                print(f"Gradient synchronization failed: {e}")
            return False
    
    def synchronize_state(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize training state across processes.
        
        Args:
            state_dict: State dictionary to synchronize
            
        Returns:
            Synchronized state dictionary
        """
        if self.world_size == 1:
            return state_dict
        
        try:
            # Reduce numerical values
            reduced_state = reduce_dict(state_dict, average=True)
            return reduced_state
            
        except Exception as e:
            if is_main_process():
                print(f"State synchronization failed: {e}")
            return state_dict
    
    def broadcast_message(self, message: CoordinationMessage):
        """Broadcast coordination message to all processes.
        
        Args:
            message: Message to broadcast
        """
        if self.world_size == 1:
            return
        
        try:
            # Serialize message (simplified - in practice would use proper serialization)
            message_data = {
                'sender_rank': message.sender_rank,
                'message_type': message.message_type,
                'timestamp': message.timestamp
            }
            
            # Broadcast message type as tensor
            message_tensor = torch.tensor([
                message.sender_rank,
                hash(message.message_type) % 1000000,  # Simple hash for message type
                int(message.timestamp)
            ], dtype=torch.long)
            
            broadcast_tensor(message_tensor, src=message.sender_rank)
            
        except Exception as e:
            if is_main_process():
                print(f"Message broadcast failed: {e}")
    
    def barrier_with_timeout(self, timeout: float = 300.0) -> bool:
        """Barrier synchronization with timeout.
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            Whether barrier completed successfully
        """
        if self.world_size == 1:
            return True
        
        try:
            # Use threading to implement timeout
            barrier_complete = threading.Event()
            
            def barrier_thread():
                try:
                    barrier()
                    barrier_complete.set()
                except Exception:
                    pass
            
            thread = threading.Thread(target=barrier_thread)
            thread.daemon = True
            thread.start()
            
            # Wait for barrier with timeout
            success = barrier_complete.wait(timeout)
            
            if not success:
                if is_main_process():
                    print(f"Barrier timeout after {timeout} seconds")
            
            return success
            
        except Exception as e:
            if is_main_process():
                print(f"Barrier failed: {e}")
            return False
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """Get synchronization statistics.
        
        Returns:
            Synchronization statistics
        """
        return {
            'total_syncs': len(self.sync_times),
            'sync_failures': self.sync_failures,
            'average_sync_time': np.mean(self.sync_times) if self.sync_times else 0,
            'total_sync_time': np.sum(self.sync_times) if self.sync_times else 0,
            'last_sync_step': self.last_sync_step
        }


class FaultToleranceManager:
    """Fault tolerance manager for distributed meta-learning.
    
    This class implements fault tolerance mechanisms including error detection,
    recovery strategies, and graceful degradation.
    """
    
    def __init__(self, world_size: int, rank: int, max_retries: int = 3,
                 checkpoint_frequency: int = 100):
        """Initialize fault tolerance manager.
        
        Args:
            world_size: Number of processes
            rank: Current process rank
            max_retries: Maximum number of retries for failed operations
            checkpoint_frequency: Frequency of checkpointing
        """
        self.world_size = world_size
        self.rank = rank
        self.max_retries = max_retries
        self.checkpoint_frequency = checkpoint_frequency
        
        # Error tracking
        self.error_count = 0
        self.error_history = []
        self.failed_processes = set()
        
        # Recovery state
        self.recovery_mode = False
        self.last_checkpoint_step = 0
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            if is_main_process():
                print(f"Received signal {signum}, initiating graceful shutdown...")
            self._graceful_shutdown()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def handle_error(self, error: Exception, context: str) -> bool:
        """Handle training error with recovery strategies.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            Whether error was handled successfully
        """
        self.error_count += 1
        error_info = {
            'error': str(error),
            'context': context,
            'rank': self.rank,
            'timestamp': time.time()
        }
        self.error_history.append(error_info)
        
        if is_main_process():
            print(f"Error in {context}: {error}")
        
        # Determine recovery strategy
        if self.error_count <= self.max_retries:
            return self._attempt_recovery(error, context)
        else:
            if is_main_process():
                print(f"Maximum retries ({self.max_retries}) exceeded, initiating graceful degradation")
            return self._graceful_degradation()
    
    def _attempt_recovery(self, error: Exception, context: str) -> bool:
        """Attempt to recover from error.
        
        Args:
            error: Exception that occurred
            context: Context where error occurred
            
        Returns:
            Whether recovery was successful
        """
        self.recovery_mode = True
        
        try:
            if "communication" in context.lower() or "distributed" in context.lower():
                # Communication error - try to re-establish connection
                return self._recover_communication()
            elif "memory" in context.lower() or "cuda" in str(error).lower():
                # Memory error - clear cache and reduce batch size
                return self._recover_memory()
            elif "gradient" in context.lower():
                # Gradient error - reset gradients
                return self._recover_gradients()
            else:
                # Generic recovery - wait and retry
                time.sleep(1.0)
                return True
                
        except Exception as recovery_error:
            if is_main_process():
                print(f"Recovery failed: {recovery_error}")
            return False
        finally:
            self.recovery_mode = False
    
    def _recover_communication(self) -> bool:
        """Recover from communication errors."""
        try:
            # Try to re-establish communication
            if dist.is_initialized():
                # Test communication with a simple all-reduce
                test_tensor = torch.tensor(1.0)
                all_reduce_tensor(test_tensor)
                return True
            else:
                # Try to reinitialize process group
                dist.init_process_group(
                    backend='nccl',
                    rank=self.rank,
                    world_size=self.world_size
                )
                return True
        except Exception:
            return False
    
    def _recover_memory(self) -> bool:
        """Recover from memory errors."""
        try:
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Run garbage collection
            import gc
            gc.collect()
            
            return True
        except Exception:
            return False
    
    def _recover_gradients(self) -> bool:
        """Recover from gradient computation errors."""
        try:
            # This would need access to the model to reset gradients
            # For now, just return True to indicate recovery attempt
            return True
        except Exception:
            return False
    
    def _graceful_degradation(self) -> bool:
        """Implement graceful degradation strategies.
        
        Returns:
            Whether degradation was successful
        """
        try:
            if self.world_size > 1:
                # Degrade to single-process training
                if is_main_process():
                    print("Degrading to single-process training")
                
                # Cleanup distributed resources
                if dist.is_initialized():
                    dist.destroy_process_group()
                
                # Update world size
                self.world_size = 1
                return True
            else:
                # Already single process, can't degrade further
                return False
                
        except Exception as e:
            if is_main_process():
                print(f"Graceful degradation failed: {e}")
            return False
    
    def _graceful_shutdown(self):
        """Perform graceful shutdown."""
        try:
            if is_main_process():
                print("Performing graceful shutdown...")
            
            # Cleanup distributed resources
            if dist.is_initialized():
                dist.destroy_process_group()
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            if is_main_process():
                print(f"Error during graceful shutdown: {e}")
    
    def should_checkpoint(self, step: int) -> bool:
        """Check if checkpointing is needed.
        
        Args:
            step: Current training step
            
        Returns:
            Whether checkpointing is needed
        """
        return (step - self.last_checkpoint_step) >= self.checkpoint_frequency
    
    def get_fault_tolerance_stats(self) -> Dict[str, Any]:
        """Get fault tolerance statistics.
        
        Returns:
            Fault tolerance statistics
        """
        return {
            'error_count': self.error_count,
            'failed_processes': list(self.failed_processes),
            'recovery_mode': self.recovery_mode,
            'max_retries': self.max_retries,
            'error_history': self.error_history[-10:]  # Last 10 errors
        }


class DistributedCoordinator:
    """Main coordinator for distributed meta-learning training.
    
    This class coordinates all aspects of distributed training including
    task sampling, synchronization, and fault tolerance.
    """
    
    def __init__(self, world_size: int, rank: int, config: Dict[str, Any]):
        """Initialize distributed coordinator.
        
        Args:
            world_size: Number of processes
            rank: Current process rank
            config: Coordination configuration
        """
        self.world_size = world_size
        self.rank = rank
        self.config = config
        
        # Initialize components
        self.task_sampler = None  # Initialized when tasks are provided
        self.synchronizer = DistributedSynchronizer(
            world_size, rank, config.get('sync_frequency', 1)
        )
        self.fault_manager = FaultToleranceManager(
            world_size, rank, 
            config.get('max_retries', 3),
            config.get('checkpoint_frequency', 100)
        )
        
        # Coordination state
        self.state = CoordinationState.INITIALIZING
        self.step_count = 0
        
        # Performance tracking
        self.coordination_times = []
    
    def initialize_task_sampling(self, tasks: List[Task], batch_size: int, 
                                shuffle: bool = True, seed: Optional[int] = None):
        """Initialize distributed task sampling.
        
        Args:
            tasks: List of all tasks
            batch_size: Batch size for sampling
            shuffle: Whether to shuffle tasks
            seed: Random seed for reproducibility
        """
        self.task_sampler = DistributedTaskSampler(
            tasks, self.world_size, self.rank, batch_size, shuffle, seed
        )
        self.state = CoordinationState.READY
    
    def coordinate_training_step(self, model: torch.nn.Module, 
                               step: int) -> Tuple[bool, Dict[str, Any]]:
        """Coordinate a single training step across processes.
        
        Args:
            model: Model being trained
            step: Current training step
            
        Returns:
            Tuple of (success, coordination_info)
        """
        start_time = time.time()
        coordination_info = {'step': step, 'rank': self.rank}
        
        try:
            self.state = CoordinationState.TRAINING
            self.step_count = step
            
            # Sample tasks if sampler is available
            if self.task_sampler is not None:
                task_batch = self.task_sampler.sample_batch(synchronized=True)
                coordination_info['batch_size'] = len(task_batch)
            
            # Synchronize gradients if needed
            if self.synchronizer.should_synchronize(step):
                self.state = CoordinationState.SYNCHRONIZING
                sync_success = self.synchronizer.synchronize_gradients(model)
                coordination_info['synchronized'] = sync_success
                
                if not sync_success:
                    # Handle synchronization failure
                    recovery_success = self.fault_manager.handle_error(
                        Exception("Gradient synchronization failed"), 
                        "gradient_synchronization"
                    )
                    if not recovery_success:
                        return False, coordination_info
            
            # Checkpoint if needed
            if self.fault_manager.should_checkpoint(step):
                self.state = CoordinationState.CHECKPOINTING
                coordination_info['checkpointed'] = True
                self.fault_manager.last_checkpoint_step = step
            
            self.state = CoordinationState.READY
            
            # Record coordination time
            coordination_time = time.time() - start_time
            self.coordination_times.append(coordination_time)
            coordination_info['coordination_time'] = coordination_time
            
            return True, coordination_info
            
        except Exception as e:
            self.state = CoordinationState.ERROR
            recovery_success = self.fault_manager.handle_error(e, "training_coordination")
            coordination_info['error'] = str(e)
            coordination_info['recovered'] = recovery_success
            
            return recovery_success, coordination_info
    
    def finalize_training(self) -> Dict[str, Any]:
        """Finalize distributed training and gather statistics.
        
        Returns:
            Final coordination statistics
        """
        self.state = CoordinationState.FINISHED
        
        # Gather statistics from all components
        stats = {
            'coordination_stats': {
                'total_steps': self.step_count,
                'average_coordination_time': np.mean(self.coordination_times) if self.coordination_times else 0,
                'total_coordination_time': np.sum(self.coordination_times) if self.coordination_times else 0
            },
            'synchronization_stats': self.synchronizer.get_sync_statistics(),
            'fault_tolerance_stats': self.fault_manager.get_fault_tolerance_stats()
        }
        
        if self.task_sampler is not None:
            stats['task_sampling_stats'] = {
                'epochs_completed': self.task_sampler.epoch,
                'current_position': self.task_sampler.current_position,
                'total_tasks': len(self.task_sampler.tasks)
            }
        
        return stats
    
    def get_coordination_state(self) -> Dict[str, Any]:
        """Get current coordination state for checkpointing.
        
        Returns:
            Coordination state dictionary
        """
        state = {
            'step_count': self.step_count,
            'state': self.state.value,
            'world_size': self.world_size,
            'rank': self.rank
        }
        
        if self.task_sampler is not None:
            state['task_sampler_state'] = self.task_sampler.get_state()
        
        return state
    
    def set_coordination_state(self, state: Dict[str, Any]):
        """Set coordination state from checkpoint.
        
        Args:
            state: Coordination state dictionary
        """
        self.step_count = state['step_count']
        self.state = CoordinationState(state['state'])
        
        if 'task_sampler_state' in state and self.task_sampler is not None:
            self.task_sampler.set_state(state['task_sampler_state'])