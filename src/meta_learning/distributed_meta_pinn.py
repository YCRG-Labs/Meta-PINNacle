"""DistributedMetaPINN: Distributed meta-learning extending PINNacle's multi-GPU trainer.

This module implements distributed meta-learning capabilities that extend
PINNacle's existing parallel training infrastructure for scalable meta-learning
across multiple GPUs with task-parallel distribution and meta-gradient synchronization.
"""

import os
import time
import copy
import random
import multiprocessing
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union

from .meta_pinn import MetaPINN
from .physics_informed_meta_learner import PhysicsInformedMetaLearner
from .config import DistributedMetaPINNConfig
from .task import Task, TaskData, TaskBatch
from .memory_optimization import (
    GradientCheckpointing, MixedPrecisionTraining, TaskDistributionOptimizer,
    MemoryMonitor, MemoryOptimizedMetaLearning
)
from .distributed_coordination import (
    DistributedCoordinator, DistributedTaskSampler, DistributedSynchronizer,
    FaultToleranceManager, CoordinationState
)
from .distributed_error_handling import (
    safe_distributed_operation,
    safe_gradient_synchronization,
    safe_process_communication,
    with_distributed_error_handling,
    setup_distributed_error_handling,
    DistributedTrainingError,
    DistributedCommunicationError,
    DistributedSynchronizationError,
    DistributedGradientError,
    ProcessFailureError,
    DistributedMemoryError,
    DistributedFaultToleranceManager
)
from ..utils import distributed_utils


class DistributedMetaPINN:
    """Distributed meta-learning model extending PINNacle's multi-GPU trainer.
    
    This class implements distributed meta-learning with:
    1. Task-parallel distribution across GPUs using existing infrastructure
    2. Meta-gradient synchronization using existing communication patterns
    3. Scalable training for large task sets with memory optimization
    """
    
    def __init__(self, config: DistributedMetaPINNConfig, base_model_class=MetaPINN):
        """Initialize DistributedMetaPINN.
        
        Args:
            config: Configuration for distributed meta-learning
            base_model_class: Base meta-learning model class (MetaPINN or PhysicsInformedMetaLearner)
        """
        self.config = config
        self.base_model_class = base_model_class
        
        # Distributed training setup
        self.world_size = config.world_size
        self.rank = None  # Set during initialization
        self.local_rank = None  # Set during initialization
        self.device = None  # Set during initialization
        
        # Task distribution parameters
        self.task_parallel = config.task_parallel
        self.data_parallel = config.data_parallel
        
        # Communication and synchronization
        self.backend = config.backend
        self.sync_frequency = config.sync_frequency
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        
        # Memory optimization components
        self.gradient_checkpointing = GradientCheckpointing(enabled=config.gradient_checkpointing)
        self.mixed_precision = MixedPrecisionTraining(enabled=config.mixed_precision)
        self.task_optimizer = TaskDistributionOptimizer(
            world_size=config.world_size,
            memory_limit_gb=getattr(config, 'memory_limit_gb', 10.0)
        )
        self.memory_monitor = MemoryMonitor(
            alert_threshold=getattr(config, 'memory_alert_threshold', 0.9),
            log_frequency=getattr(config, 'memory_log_frequency', 100)
        )
        
        # Base model instance (created per process)
        self.model = None
        self.ddp_model = None
        
        # Training state
        self.global_step = 0
        self.local_step = 0
        
        # Performance tracking
        self.communication_times = []
        self.computation_times = []
        self.memory_usage = []
        
        # Distributed coordination
        self.coordinator = None  # Initialized during setup
        
        # Fault tolerance
        self.checkpoint_frequency = config.checkpoint_frequency
        self.fault_tolerance = config.fault_tolerance
        
    @safe_distributed_operation
    def setup_distributed(self, rank: int, world_size: int):
        """Setup distributed training environment.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
        """
        self.rank = rank
        self.world_size = world_size
        self.local_rank = rank % torch.cuda.device_count()
        
        # Set device
        if torch.cuda.is_available():
            self.device = torch.device(f'cuda:{self.local_rank}')
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = torch.device('cpu')
        
        # Initialize process group
        os.environ['MASTER_ADDR'] = self.config.master_addr
        os.environ['MASTER_PORT'] = str(self.config.master_port)
        
        try:
            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size,
                timeout=self.config.timeout
            )
            
            # Setup fault tolerance manager
            self.fault_tolerance_manager = setup_distributed_error_handling(
                world_size=world_size,
                rank=rank,
                enable_graceful_degradation=self.fault_tolerance
            )
            
            # Initialize distributed coordinator
            coordination_config = {
                'sync_frequency': self.sync_frequency,
                'max_retries': getattr(self.config, 'max_retries', 3),
                'checkpoint_frequency': self.checkpoint_frequency
            }
            self.coordinator = DistributedCoordinator(world_size, rank, coordination_config)
            
            if self.rank == 0:
                print(f"Distributed training initialized: {world_size} processes, backend: {self.backend}")
                print(f"Coordination enabled with sync frequency: {self.sync_frequency}")
                print(f"Fault tolerance enabled: {self.fault_tolerance}")
                
        except (DistributedCommunicationError, DistributedTrainingError) as e:
            if self.fault_tolerance:
                print(f"Warning: Distributed initialization failed: {e}")
                print("Attempting graceful degradation to single-GPU training")
                return self._fallback_to_single_gpu()
            else:
                raise DistributedTrainingError(f"Distributed initialization failed: {e}")
        except Exception as e:
            if self.fault_tolerance:
                print(f"Warning: Unexpected error during distributed initialization: {e}")
                print("Falling back to single-GPU training")
                return self._fallback_to_single_gpu()
            else:
                raise DistributedTrainingError(f"Unexpected error during distributed initialization: {e}")
    
    def _fallback_to_single_gpu(self):
        """Fallback to single-GPU training when distributed setup fails."""
        print("Initializing single-GPU fallback mode")
        self.world_size = 1
        self.rank = 0
        self.local_rank = 0
        
        # Set device to first available GPU or CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')
            torch.cuda.set_device(0)
        else:
            self.device = torch.device('cpu')
        
        # Initialize single-process coordinator
        self.coordinator = DistributedCoordinator(1, 0, {'sync_frequency': 1})
        self.fault_tolerance_manager = None  # No fault tolerance needed for single GPU
        
        print(f"Single-GPU mode initialized on device: {self.device}")
    
    def create_model(self, model_config) -> Union[MetaPINN, PhysicsInformedMetaLearner]:
        """Create base model instance for this process.
        
        Args:
            model_config: Configuration for the base model
            
        Returns:
            Base model instance
        """
        # Update config with device information
        model_config.device = str(self.device)
        
        # Create base model
        model = self.base_model_class(model_config)
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            # Find unused parameters for complex meta-learning models
            find_unused_parameters = isinstance(model, PhysicsInformedMetaLearner)
            
            self.ddp_model = DDP(
                model.network,
                device_ids=[self.local_rank] if torch.cuda.is_available() else None,
                output_device=self.local_rank if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
                gradient_as_bucket_view=True  # Memory optimization
            )
            
            # Replace model's network with DDP wrapper
            model.network = self.ddp_model
        
        self.model = model
        return model
    
    def distribute_tasks(self, tasks: List[Task]) -> List[Task]:
        """Distribute tasks across processes for task-parallel training with memory optimization.
        
        Args:
            tasks: List of all tasks
            
        Returns:
            List of tasks assigned to this process
        """
        if not self.task_parallel or self.world_size == 1:
            return tasks
        
        # Use memory-optimized task distribution
        if hasattr(self.model, 'network'):
            model_size_mb = sum(p.numel() * 4 for p in self.model.network.parameters()) / 1024**2
            distributed_tasks = self.task_optimizer.distribute_tasks_by_memory(tasks, model_size_mb)
            
            if self.rank < len(distributed_tasks):
                local_tasks = distributed_tasks[self.rank]
            else:
                local_tasks = []
        else:
            # Fallback to simple distribution
            tasks_per_process = len(tasks) // self.world_size
            remainder = len(tasks) % self.world_size
            
            start_idx = self.rank * tasks_per_process
            if self.rank < remainder:
                start_idx += self.rank
                end_idx = start_idx + tasks_per_process + 1
            else:
                start_idx += remainder
                end_idx = start_idx + tasks_per_process
            
            local_tasks = tasks[start_idx:end_idx]
        
        if self.rank == 0:
            print(f"Memory-optimized task distribution: {len(tasks)} total tasks, "
                  f"{len(local_tasks)} tasks assigned to rank {self.rank}")
        
        return local_tasks
    
    @safe_gradient_synchronization
    def synchronize_meta_gradients(self):
        """Synchronize meta-gradients across all processes."""
        if self.world_size == 1:
            return True
        
        start_time = time.time()
        
        try:
            # Check if we should continue training
            if self.fault_tolerance_manager and not self.fault_tolerance_manager.should_continue_training():
                print(f"Process {self.rank}: Training should not continue, skipping gradient sync")
                return False
            
            # Get model parameters
            if hasattr(self.model, 'network'):
                if isinstance(self.model.network, DDP):
                    # DDP handles gradient synchronization automatically
                    # But we still need to check for NaN/Inf gradients
                    self._check_gradient_health()
                else:
                    # Manual gradient synchronization for non-DDP models
                    success = self._manual_gradient_sync()
                    if not success:
                        raise DistributedGradientError("Manual gradient synchronization failed")
            
            # Synchronize meta-optimizer state if needed
            if hasattr(self.model, 'meta_optimizer') and self.global_step % self.sync_frequency == 0:
                self._synchronize_optimizer_state()
            
            sync_time = time.time() - start_time
            self.communication_times.append(sync_time)
            return True
            
        except (DistributedGradientError, DistributedCommunicationError) as e:
            if self.fault_tolerance_manager:
                recovery_success = self.fault_tolerance_manager.handle_distributed_error(e, "gradient_synchronization")
                if recovery_success:
                    print(f"Process {self.rank}: Recovered from gradient synchronization error")
                    return True
                else:
                    print(f"Process {self.rank}: Failed to recover from gradient synchronization error")
                    return False
            else:
                raise
        except Exception as e:
            error = DistributedGradientError(f"Unexpected error in gradient synchronization: {e}")
            if self.fault_tolerance_manager:
                recovery_success = self.fault_tolerance_manager.handle_distributed_error(error, "gradient_synchronization")
                return recovery_success
            else:
                raise error
    
    def _check_gradient_health(self):
        """Check for NaN/Inf in gradients."""
        if not hasattr(self.model, 'network'):
            return
        
        for name, param in self.model.network.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    raise DistributedGradientError(f"NaN detected in gradients for parameter: {name}")
                if torch.isinf(param.grad).any():
                    raise DistributedGradientError(f"Inf detected in gradients for parameter: {name}")
    
    def _manual_gradient_sync(self) -> bool:
        """Perform manual gradient synchronization."""
        try:
            for param in self.model.network.parameters():
                if param.grad is not None:
                    # Check for NaN/Inf before synchronization
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise DistributedGradientError("NaN/Inf detected in gradients before synchronization")
                    
                    # Synchronize gradients
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size
                    
                    # Check for NaN/Inf after synchronization
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        raise DistributedGradientError("NaN/Inf detected in gradients after synchronization")
            
            return True
            
        except dist.DistBackendError as e:
            raise DistributedCommunicationError(f"Communication error during gradient sync: {e}")
        except Exception as e:
            raise DistributedGradientError(f"Error during manual gradient sync: {e}")
    
    def _synchronize_optimizer_state(self):
        """Synchronize optimizer state across processes."""
        if not hasattr(self.model, 'meta_optimizer'):
            return
        
        # Synchronize optimizer state dict
        optimizer_state = self.model.meta_optimizer.state_dict()
        
        # Convert state to tensors for communication
        state_tensors = []
        for group in optimizer_state['state'].values():
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    state_tensors.append(value)
        
        # All-reduce optimizer state tensors
        for tensor in state_tensors:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            tensor.div_(self.world_size)
    
    def distributed_meta_train_step(self, local_tasks: List[Task]) -> Dict[str, float]:
        """Perform distributed meta-training step.
        
        Args:
            local_tasks: Tasks assigned to this process
            
        Returns:
            Training metrics
        """
        computation_start = time.time()
        
        # Sample local task batch
        if len(local_tasks) >= self.model.meta_batch_size:
            task_indices = np.random.choice(len(local_tasks), self.model.meta_batch_size, replace=False)
            local_task_batch = [local_tasks[i] for i in task_indices]
        else:
            local_task_batch = local_tasks
        
        # Compute local meta-gradients
        local_meta_loss = 0.0
        local_task_losses = []
        
        # Enable gradient accumulation for memory efficiency
        effective_batch_size = len(local_task_batch)
        accumulation_steps = max(1, effective_batch_size // self.gradient_accumulation_steps)
        
        self.model.meta_optimizer.zero_grad()
        
        for i, task in enumerate(local_task_batch):
            # Inner loop: adapt to task
            adapted_params = self.model.adapt_to_task(task)
            
            # Outer loop: evaluate on query set
            query_loss = self.model.compute_total_loss(task.query_data, task, adapted_params)
            
            # Scale loss for gradient accumulation
            scaled_loss = query_loss / accumulation_steps
            local_meta_loss += scaled_loss
            local_task_losses.append(query_loss.item())
            
            # Backward pass with memory optimization
            with self.mixed_precision.autocast_context():
                if self.gradient_checkpointing.enabled:
                    # Use gradient checkpointing for memory efficiency
                    self.gradient_checkpointing.checkpoint_function(lambda: scaled_loss.backward())
                else:
                    self.mixed_precision.backward(scaled_loss)
            
            # Gradient accumulation step
            if (i + 1) % accumulation_steps == 0 or i == len(local_task_batch) - 1:
                # Clip gradients if specified
                if self.model.config.grad_clip_norm is not None:
                    if isinstance(self.model.network, DDP):
                        torch.nn.utils.clip_grad_norm_(
                            self.model.network.module.parameters(),
                            self.model.config.grad_clip_norm
                        )
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.network.parameters(),
                            self.model.config.grad_clip_norm
                        )
                
                # Synchronize gradients across processes
                self.synchronize_meta_gradients()
                
                # Meta-optimizer step with mixed precision
                self.mixed_precision.step(self.model.meta_optimizer)
                self.model.meta_optimizer.zero_grad()
                
                # Monitor memory usage
                self.memory_monitor.log_memory_usage(
                    self.global_step,
                    {'task_batch_size': len(local_task_batch), 'accumulation_step': i}
                )
        
        computation_time = time.time() - computation_start
        self.computation_times.append(computation_time)
        
        # Update step counters
        self.local_step += 1
        self.global_step += 1
        
        # Gather metrics from all processes
        metrics = self._gather_training_metrics(local_meta_loss.item(), local_task_losses)
        
        return metrics
    
    def _gather_training_metrics(self, local_loss: float, local_task_losses: List[float]) -> Dict[str, float]:
        """Gather training metrics from all processes.
        
        Args:
            local_loss: Local meta-loss
            local_task_losses: Local task losses
            
        Returns:
            Aggregated metrics
        """
        if self.world_size == 1:
            return {
                'meta_loss': local_loss,
                'mean_task_loss': np.mean(local_task_losses),
                'std_task_loss': np.std(local_task_losses),
                'n_tasks': len(local_task_losses),
                'global_step': self.global_step
            }
        
        # Convert to tensors for communication
        local_metrics = torch.tensor([
            local_loss,
            np.mean(local_task_losses),
            np.std(local_task_losses),
            len(local_task_losses)
        ], device=self.device)
        
        # All-reduce metrics
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        
        # Compute global metrics
        global_loss = local_metrics[0].item() / self.world_size
        global_mean_task_loss = local_metrics[1].item() / self.world_size
        global_std_task_loss = local_metrics[2].item() / self.world_size
        total_tasks = int(local_metrics[3].item())
        
        return {
            'meta_loss': global_loss,
            'mean_task_loss': global_mean_task_loss,
            'std_task_loss': global_std_task_loss,
            'n_tasks': total_tasks,
            'global_step': self.global_step,
            'world_size': self.world_size
        }
    
    def distributed_meta_train(self, train_tasks: List[Task], val_tasks: Optional[List[Task]] = None,
                              meta_iterations: Optional[int] = None, **train_args) -> Dict[str, Any]:
        """Distributed meta-training pipeline.
        
        Args:
            train_tasks: Training tasks
            val_tasks: Validation tasks
            meta_iterations: Number of meta-iterations
            **train_args: Additional training arguments
            
        Returns:
            Training results
        """
        if meta_iterations is None:
            meta_iterations = self.model.config.meta_iterations
        
        # Initialize coordinated task sampling
        if self.coordinator:
            self.coordinator.initialize_task_sampling(
                train_tasks, 
                self.model.meta_batch_size,
                shuffle=True,
                seed=getattr(self.model.config, 'seed', None)
            )
        
        # Distribute tasks across processes
        local_train_tasks = self.distribute_tasks(train_tasks)
        local_val_tasks = self.distribute_tasks(val_tasks) if val_tasks else None
        
        if self.rank == 0:
            print(f"Starting distributed meta-training for {meta_iterations} iterations...")
            print(f"Training on {len(train_tasks)} total tasks ({len(local_train_tasks)} local)")
            if val_tasks:
                print(f"Validating on {len(val_tasks)} total tasks ({len(local_val_tasks)} local)")
        
        # Initialize training state
        training_history = []
        best_val_loss = float('inf')
        start_time = time.time()
        
        for iteration in range(meta_iterations):
            # Coordinate training step
            if self.coordinator:
                coord_success, coord_info = self.coordinator.coordinate_training_step(
                    self.model.network, iteration
                )
                if not coord_success:
                    if self.rank == 0:
                        print(f"Coordination failed at iteration {iteration}, attempting recovery...")
                    continue
            
            # Training step
            train_metrics = self.distributed_meta_train_step(local_train_tasks)
            
            # Validation step
            if local_val_tasks and iteration % self.model.config.validation_frequency == 0:
                val_metrics = self.distributed_meta_validate(local_val_tasks)
                
                # Update best model (only on rank 0)
                if self.rank == 0 and val_metrics['meta_loss'] < best_val_loss:
                    best_val_loss = val_metrics['meta_loss']
                    self.save_checkpoint(iteration, val_metrics['meta_loss'])
                
                # Log progress (only on rank 0)
                if self.rank == 0:
                    print(f"Iteration {iteration}: train_loss={train_metrics['meta_loss']:.6f}, "
                          f"val_loss={val_metrics['meta_loss']:.6f}")
                
                # Store training history
                training_history.append({
                    'iteration': iteration,
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                })
            
            elif self.rank == 0 and iteration % 100 == 0:
                print(f"Iteration {iteration}: train_loss={train_metrics['meta_loss']:.6f}")
            
            # Checkpoint periodically
            if self.rank == 0 and iteration % self.checkpoint_frequency == 0:
                self.save_checkpoint(iteration, train_metrics['meta_loss'])
            
            # Memory cleanup
            if iteration % 100 == 0:
                torch.cuda.empty_cache()
        
        total_time = time.time() - start_time
        
        # Finalize coordination and gather statistics
        coordination_stats = {}
        if self.coordinator:
            coordination_stats = self.coordinator.finalize_training()
        
        if self.rank == 0:
            print(f"Distributed meta-training completed in {total_time:.2f} seconds")
        
        results = {
            'training_history': training_history,
            'best_val_loss': best_val_loss,
            'total_time': total_time,
            'final_iteration': meta_iterations,
            'world_size': self.world_size,
            'performance_stats': self.get_performance_stats(),
            'coordination_stats': coordination_stats
        }
        
        return results
    
    def distributed_meta_validate(self, local_val_tasks: List[Task]) -> Dict[str, float]:
        """Distributed meta-validation.
        
        Args:
            local_val_tasks: Local validation tasks
            
        Returns:
            Validation metrics
        """
        self.model.network.eval()
        local_val_losses = []
        
        with torch.no_grad():
            for task in local_val_tasks:
                # Adapt to task
                adapted_params = self.model.adapt_to_task(task)
                
                # Evaluate on query set
                query_loss = self.model.compute_total_loss(task.query_data, task, adapted_params)
                local_val_losses.append(query_loss.item())
        
        self.model.network.train()
        
        # Gather validation metrics
        if self.world_size == 1:
            return {
                'meta_loss': np.mean(local_val_losses),
                'std_loss': np.std(local_val_losses),
                'n_tasks': len(local_val_losses)
            }
        
        # Convert to tensors for communication
        local_metrics = torch.tensor([
            np.mean(local_val_losses),
            np.std(local_val_losses),
            len(local_val_losses)
        ], device=self.device)
        
        # All-reduce metrics
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)
        
        return {
            'meta_loss': local_metrics[0].item() / self.world_size,
            'std_loss': local_metrics[1].item() / self.world_size,
            'n_tasks': int(local_metrics[2].item())
        }
    
    def save_checkpoint(self, iteration: int, loss: float):
        """Save training checkpoint.
        
        Args:
            iteration: Current iteration
            loss: Current loss value
        """
        if self.rank != 0:
            return
        
        checkpoint = {
            'iteration': iteration,
            'loss': loss,
            'global_step': self.global_step,
            'model_state_dict': self.model.network.module.state_dict() if isinstance(self.model.network, DDP) else self.model.network.state_dict(),
            'optimizer_state_dict': self.model.meta_optimizer.state_dict(),
            'config': self.config,
            'world_size': self.world_size
        }
        
        # Add coordination state if available
        if self.coordinator:
            checkpoint['coordination_state'] = self.coordinator.get_coordination_state()
        
        checkpoint_path = f"checkpoint_iter_{iteration}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only recent checkpoints
        if iteration > self.checkpoint_frequency * 5:
            old_checkpoint = f"checkpoint_iter_{iteration - self.checkpoint_frequency * 5}.pt"
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if isinstance(self.model.network, DDP):
            self.model.network.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.network.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.model.meta_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.global_step = checkpoint.get('global_step', 0)
        
        # Load coordination state if available
        if 'coordination_state' in checkpoint and self.coordinator:
            self.coordinator.set_coordination_state(checkpoint['coordination_state'])
        
        if self.rank == 0:
            print(f"Loaded checkpoint from iteration {checkpoint['iteration']}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Performance statistics
        """
        stats = {
            'communication_times': {
                'mean': np.mean(self.communication_times) if self.communication_times else 0,
                'std': np.std(self.communication_times) if self.communication_times else 0,
                'total': np.sum(self.communication_times) if self.communication_times else 0
            },
            'computation_times': {
                'mean': np.mean(self.computation_times) if self.computation_times else 0,
                'std': np.std(self.computation_times) if self.computation_times else 0,
                'total': np.sum(self.computation_times) if self.computation_times else 0
            },
            'memory_usage': {
                'peak': max(self.memory_usage) if self.memory_usage else 0,
                'mean': np.mean(self.memory_usage) if self.memory_usage else 0
            },
            'world_size': self.world_size,
            'rank': self.rank,
            'device': str(self.device)
        }
        
        # Add GPU memory stats if available
        if torch.cuda.is_available():
            stats['gpu_memory'] = {
                'allocated': torch.cuda.memory_allocated(self.device),
                'cached': torch.cuda.memory_reserved(self.device),
                'max_allocated': torch.cuda.max_memory_allocated(self.device)
            }
        
        return stats
    
    def cleanup(self):
        """Cleanup distributed training resources."""
        if self.world_size > 1:
            dist.destroy_process_group()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def distributed_meta_train_worker(rank: int, world_size: int, config: DistributedMetaPINNConfig,
                                 model_config, train_tasks: List[Task], val_tasks: Optional[List[Task]],
                                 meta_iterations: int, results_queue: mp.Queue):
    """Worker function for distributed meta-training.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Distributed training configuration
        model_config: Base model configuration
        train_tasks: Training tasks
        val_tasks: Validation tasks
        meta_iterations: Number of meta-iterations
        results_queue: Queue for returning results
    """
    try:
        # Setup distributed training
        distributed_trainer = DistributedMetaPINN(config)
        distributed_trainer.setup_distributed(rank, world_size)
        
        # Create model
        model = distributed_trainer.create_model(model_config)
        
        # Run distributed training
        results = distributed_trainer.distributed_meta_train(
            train_tasks, val_tasks, meta_iterations
        )
        
        # Return results (only from rank 0)
        if rank == 0:
            results_queue.put(results)
        
        # Cleanup
        distributed_trainer.cleanup()
        
    except Exception as e:
        if rank == 0:
            results_queue.put({'error': str(e)})
        raise


def launch_distributed_meta_training(config: DistributedMetaPINNConfig, model_config,
                                    train_tasks: List[Task], val_tasks: Optional[List[Task]] = None,
                                    meta_iterations: int = 10000) -> Dict[str, Any]:
    """Launch distributed meta-training using multiprocessing.
    
    Args:
        config: Distributed training configuration
        model_config: Base model configuration
        train_tasks: Training tasks
        val_tasks: Validation tasks
        meta_iterations: Number of meta-iterations
        
    Returns:
        Training results
    """
    world_size = config.world_size
    
    if world_size == 1:
        # Single process training
        distributed_trainer = DistributedMetaPINN(config)
        distributed_trainer.setup_distributed(0, 1)
        model = distributed_trainer.create_model(model_config)
        return distributed_trainer.distributed_meta_train(train_tasks, val_tasks, meta_iterations)
    
    # Multi-process training
    mp.set_start_method('spawn', force=True)
    results_queue = mp.Queue()
    
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=distributed_meta_train_worker,
            args=(rank, world_size, config, model_config, train_tasks, val_tasks, meta_iterations, results_queue)
        )
        p.start()
        processes.append(p)
    
    # Wait for results
    results = results_queue.get()
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    if 'error' in results:
        raise RuntimeError(f"Distributed training failed: {results['error']}")
    
    return results