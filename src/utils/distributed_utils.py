"""Distributed training utilities extending PINNacle's existing infrastructure.

This module provides utilities for distributed meta-learning that integrate
with PINNacle's existing parallel training capabilities.
"""

import os
import time
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List, Optional, Any, Union


def setup_distributed_environment(rank: int, world_size: int, backend: str = 'nccl',
                                 master_addr: str = 'localhost', master_port: int = 12355):
    """Setup distributed training environment.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: Communication backend
        master_addr: Master node address
        master_port: Master node port
    """
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Initialize process group
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size
    )
    
    # Set device for this process
    if torch.cuda.is_available() and backend == 'nccl':
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return device


def cleanup_distributed():
    """Cleanup distributed training resources."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_rank() -> int:
    """Get current process rank."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get total number of processes."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def all_reduce_tensor(tensor: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """All-reduce tensor across all processes.
    
    Args:
        tensor: Input tensor
        op: Reduction operation
        
    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor
    
    # Clone tensor to avoid in-place operations
    reduced_tensor = tensor.clone()
    dist.all_reduce(reduced_tensor, op=op)
    
    return reduced_tensor


def all_gather_tensors(tensor: torch.Tensor) -> List[torch.Tensor]:
    """All-gather tensors from all processes.
    
    Args:
        tensor: Input tensor
        
    Returns:
        List of tensors from all processes
    """
    if not dist.is_initialized():
        return [tensor]
    
    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    
    return tensor_list


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source process to all processes.
    
    Args:
        tensor: Input tensor
        src: Source process rank
        
    Returns:
        Broadcasted tensor
    """
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def reduce_dict(input_dict: Dict[str, Union[torch.Tensor, float]], 
                average: bool = True) -> Dict[str, float]:
    """Reduce dictionary of values across all processes.
    
    Args:
        input_dict: Dictionary of values to reduce
        average: Whether to average the values
        
    Returns:
        Dictionary of reduced values
    """
    if not dist.is_initialized():
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in input_dict.items()}
    
    world_size = get_world_size()
    reduced_dict = {}
    
    for key, value in input_dict.items():
        if isinstance(value, torch.Tensor):
            reduced_tensor = all_reduce_tensor(value)
            if average:
                reduced_tensor /= world_size
            reduced_dict[key] = reduced_tensor.item()
        else:
            # Convert to tensor for reduction
            tensor_value = torch.tensor(float(value), device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
            reduced_tensor = all_reduce_tensor(tensor_value)
            if average:
                reduced_tensor /= world_size
            reduced_dict[key] = reduced_tensor.item()
    
    return reduced_dict


def synchronize_model_parameters(model: torch.nn.Module):
    """Synchronize model parameters across all processes.
    
    Args:
        model: PyTorch model
    """
    if not dist.is_initialized():
        return
    
    for param in model.parameters():
        dist.broadcast(param.data, src=0)


def distribute_tasks_across_processes(tasks: List[Any], rank: int, world_size: int) -> List[Any]:
    """Distribute tasks across processes for task-parallel training.
    
    Args:
        tasks: List of tasks
        rank: Current process rank
        world_size: Total number of processes
        
    Returns:
        List of tasks assigned to this process
    """
    if world_size == 1:
        return tasks
    
    # Calculate tasks per process with load balancing
    tasks_per_process = len(tasks) // world_size
    remainder = len(tasks) % world_size
    
    # Distribute tasks
    start_idx = rank * tasks_per_process
    if rank < remainder:
        start_idx += rank
        end_idx = start_idx + tasks_per_process + 1
    else:
        start_idx += remainder
        end_idx = start_idx + tasks_per_process
    
    return tasks[start_idx:end_idx]


def gather_results_from_all_processes(local_results: Dict[str, Any]) -> Dict[str, Any]:
    """Gather results from all processes to rank 0.
    
    Args:
        local_results: Results from current process
        
    Returns:
        Aggregated results (only valid on rank 0)
    """
    if not dist.is_initialized():
        return local_results
    
    world_size = get_world_size()
    rank = get_rank()
    
    # Convert results to tensors for communication
    tensor_results = {}
    for key, value in local_results.items():
        if isinstance(value, (int, float)):
            tensor_results[key] = torch.tensor(float(value), device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
        elif isinstance(value, torch.Tensor):
            tensor_results[key] = value
        elif isinstance(value, (list, np.ndarray)):
            tensor_results[key] = torch.tensor(value, device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
    
    # Gather tensors from all processes
    gathered_results = {}
    for key, tensor in tensor_results.items():
        gathered_tensors = all_gather_tensors(tensor)
        if rank == 0:
            # Aggregate results on rank 0
            if tensor.numel() == 1:  # Scalar value
                gathered_results[key] = [t.item() for t in gathered_tensors]
            else:  # Vector/matrix
                gathered_results[key] = [t.cpu().numpy() for t in gathered_tensors]
    
    return gathered_results if rank == 0 else {}


class DistributedTimer:
    """Timer for measuring distributed training performance."""
    
    def __init__(self):
        self.start_times = {}
        self.elapsed_times = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        barrier()  # Synchronize before starting
        self.start_times[name] = time.time()
    
    def end(self, name: str):
        """End timing an operation."""
        barrier()  # Synchronize before ending
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.elapsed_times:
                self.elapsed_times[name] = []
            self.elapsed_times[name].append(elapsed)
    
    def get_average_time(self, name: str) -> float:
        """Get average time for an operation."""
        if name not in self.elapsed_times:
            return 0.0
        return np.mean(self.elapsed_times[name])
    
    def get_total_time(self, name: str) -> float:
        """Get total time for an operation."""
        if name not in self.elapsed_times:
            return 0.0
        return np.sum(self.elapsed_times[name])
    
    def reset(self):
        """Reset all timers."""
        self.start_times.clear()
        self.elapsed_times.clear()


class DistributedMemoryTracker:
    """Memory usage tracker for distributed training."""
    
    def __init__(self):
        self.memory_snapshots = {}
    
    def snapshot(self, name: str):
        """Take a memory snapshot."""
        if torch.cuda.is_available():
            memory_info = {
                'allocated': torch.cuda.memory_allocated(),
                'cached': torch.cuda.memory_reserved(),
                'max_allocated': torch.cuda.max_memory_allocated()
            }
        else:
            # For CPU, we can't easily track memory usage
            memory_info = {'allocated': 0, 'cached': 0, 'max_allocated': 0}
        
        if name not in self.memory_snapshots:
            self.memory_snapshots[name] = []
        self.memory_snapshots[name].append(memory_info)
    
    def get_peak_memory(self, name: str) -> Dict[str, float]:
        """Get peak memory usage for an operation."""
        if name not in self.memory_snapshots:
            return {'allocated': 0, 'cached': 0, 'max_allocated': 0}
        
        snapshots = self.memory_snapshots[name]
        return {
            'allocated': max(s['allocated'] for s in snapshots),
            'cached': max(s['cached'] for s in snapshots),
            'max_allocated': max(s['max_allocated'] for s in snapshots)
        }
    
    def reset(self):
        """Reset memory tracking."""
        self.memory_snapshots.clear()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def log_distributed_info():
    """Log distributed training information."""
    if is_main_process():
        print(f"Distributed training info:")
        print(f"  World size: {get_world_size()}")
        print(f"  Backend: {dist.get_backend() if dist.is_initialized() else 'Not initialized'}")
        if torch.cuda.is_available():
            print(f"  CUDA devices: {torch.cuda.device_count()}")
            print(f"  Current device: {torch.cuda.current_device()}")


def check_distributed_setup():
    """Check if distributed training is properly set up."""
    if not dist.is_initialized():
        return False
    
    try:
        # Test communication
        test_tensor = torch.tensor(1.0, device=torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
        all_reduce_tensor(test_tensor)
        return True
    except Exception as e:
        print(f"Distributed setup check failed: {e}")
        return False


def get_distributed_stats() -> Dict[str, Any]:
    """Get distributed training statistics."""
    stats = {
        'initialized': dist.is_initialized(),
        'world_size': get_world_size(),
        'rank': get_rank(),
        'is_main_process': is_main_process()
    }
    
    if dist.is_initialized():
        stats['backend'] = dist.get_backend()
    
    if torch.cuda.is_available():
        stats['cuda_device_count'] = torch.cuda.device_count()
        stats['current_cuda_device'] = torch.cuda.current_device()
        stats['cuda_memory_allocated'] = torch.cuda.memory_allocated()
        stats['cuda_memory_cached'] = torch.cuda.memory_reserved()
    
    return stats