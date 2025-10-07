"""Extension to PINNacle's trainer.py for distributed meta-learning.

This module extends the existing Trainer class to support distributed meta-learning
while maintaining compatibility with PINNacle's existing training infrastructure.
"""

import os
import sys
import time
import dill
import json
import multiprocessing
import torch
import torch.distributed as dist
from typing import Dict, List, Optional, Any, Union, Callable

from .distributed_meta_pinn import DistributedMetaPINN, launch_distributed_meta_training
from .config import DistributedMetaPINNConfig
from .task import Task
from ..utils.distributed_utils import (
    setup_distributed_environment, cleanup_distributed, is_main_process,
    log_distributed_info, DistributedTimer, DistributedMemoryTracker
)


class DistributedMetaTrainer:
    """Distributed meta-learning trainer extending PINNacle's existing trainer infrastructure.
    
    This class extends PINNacle's Trainer to support distributed meta-learning
    while maintaining compatibility with existing parallel training capabilities.
    """
    
    def __init__(self, exp_name: str, device: str, config: DistributedMetaPINNConfig):
        """Initialize distributed meta-learning trainer.
        
        Args:
            exp_name: Experiment name
            device: Device specification (e.g., "0,1,2,3" for multi-GPU)
            config: Distributed training configuration
        """
        self.exp_name = exp_name
        self.device_list = device.split(",")
        self.config = config
        
        # Update world size based on available devices
        if config.world_size == 1 and len(self.device_list) > 1:
            self.config.world_size = len(self.device_list)
        
        # Meta-learning specific attributes
        self.meta_tasks = []
        self.repeat = 1
        
        # Performance tracking
        self.timer = DistributedTimer()
        self.memory_tracker = DistributedMemoryTracker()
        
        # Training state
        self.training_results = {}
        
    def set_repeat(self, repeat: int):
        """Set number of repetitions for experiments."""
        self.repeat = repeat
    
    def add_meta_task(self, get_model_func: Callable, train_tasks: List[Task], 
                     val_tasks: Optional[List[Task]], meta_train_args: Dict[str, Any]):
        """Add meta-learning task to the trainer.
        
        Args:
            get_model_func: Function that returns a meta-learning model
            train_tasks: Training tasks
            val_tasks: Validation tasks
            meta_train_args: Meta-training arguments
        """
        # Serialize the task data for multiprocessing
        task_data = dill.dumps((get_model_func, train_tasks, val_tasks, meta_train_args))
        self.meta_tasks.append((task_data, meta_train_args))
    
    def setup_experiment(self, filename: str, seed: int):
        """Setup experiment directory and configuration.
        
        Args:
            filename: Script filename
            seed: Random seed
        """
        # Create experiment directory
        os.makedirs(f"runs/{self.exp_name}", exist_ok=True)
        
        # Copy script file
        if os.path.exists(filename):
            import shutil
            shutil.copy(filename, f"runs/{self.exp_name}/script.py.bak")
        
        # Save configuration
        config_data = {
            "seed": seed,
            "distributed_config": self.config.__dict__,
            "n_meta_tasks": len(self.meta_tasks),
            "world_size": self.config.world_size,
            "device_list": self.device_list
        }
        
        with open(f"runs/{self.exp_name}/config.json", 'w') as f:
            json.dump(config_data, f, indent=4, default=lambda x: str(x))
    
    def train_all_distributed(self) -> Dict[str, Any]:
        """Train all meta-learning tasks using distributed training.
        
        Returns:
            Training results
        """
        if self.config.world_size > 1:
            return self._train_all_distributed_multiprocess()
        else:
            return self._train_all_single_process()
    
    def _train_all_single_process(self) -> Dict[str, Any]:
        """Train all tasks on single process (fallback)."""
        print("Running single-process meta-learning training...")
        
        all_results = {}
        
        for repeat_idx in range(self.repeat):
            for task_idx, (task_data, meta_train_args) in enumerate(self.meta_tasks):
                print(f"***** Begin meta-task #{task_idx}-{repeat_idx} *****")
                
                # Setup logging
                save_path = f"runs/{self.exp_name}/{task_idx}-{repeat_idx}"
                os.makedirs(save_path, exist_ok=True)
                
                # Deserialize task data
                get_model_func, train_tasks, val_tasks, args = dill.loads(task_data)
                
                # Create model
                model = get_model_func()
                
                # Set device
                if self.device_list[0] != 'cpu':
                    device = torch.device(f"cuda:{self.device_list[0]}")
                    model = model.to(device)
                
                # Train model
                self.timer.start(f"task_{task_idx}_{repeat_idx}")
                self.memory_tracker.snapshot(f"task_{task_idx}_{repeat_idx}_start")
                
                results = model.meta_train(train_tasks, val_tasks, **args)
                
                self.timer.end(f"task_{task_idx}_{repeat_idx}")
                self.memory_tracker.snapshot(f"task_{task_idx}_{repeat_idx}_end")
                
                # Save results
                results['save_path'] = save_path
                results['task_idx'] = task_idx
                results['repeat_idx'] = repeat_idx
                all_results[f"{task_idx}-{repeat_idx}"] = results
                
                # Save model
                model.save_model(f"{save_path}/model.pt")
                
                print(f"*****  End meta-task #{task_idx}-{repeat_idx}  *****")
        
        return all_results
    
    def _train_all_distributed_multiprocess(self) -> Dict[str, Any]:
        """Train all tasks using distributed multiprocessing."""
        print(f"Running distributed meta-learning training with {self.config.world_size} processes...")
        
        # Log distributed info
        log_distributed_info()
        
        all_results = {}
        
        for repeat_idx in range(self.repeat):
            for task_idx, (task_data, meta_train_args) in enumerate(self.meta_tasks):
                print(f"***** Begin distributed meta-task #{task_idx}-{repeat_idx} *****")
                
                # Setup save path
                save_path = f"runs/{self.exp_name}/{task_idx}-{repeat_idx}"
                os.makedirs(save_path, exist_ok=True)
                
                # Deserialize task data
                get_model_func, train_tasks, val_tasks, args = dill.loads(task_data)
                
                # Create model config
                model = get_model_func()
                model_config = model.config
                
                # Launch distributed training
                self.timer.start(f"distributed_task_{task_idx}_{repeat_idx}")
                
                try:
                    results = launch_distributed_meta_training(
                        config=self.config,
                        model_config=model_config,
                        train_tasks=train_tasks,
                        val_tasks=val_tasks,
                        meta_iterations=args.get('meta_iterations', model_config.meta_iterations)
                    )
                    
                    # Add metadata
                    results['save_path'] = save_path
                    results['task_idx'] = task_idx
                    results['repeat_idx'] = repeat_idx
                    results['distributed'] = True
                    results['world_size'] = self.config.world_size
                    
                    all_results[f"{task_idx}-{repeat_idx}"] = results
                    
                except Exception as e:
                    print(f"Distributed training failed for task {task_idx}-{repeat_idx}: {e}")
                    if self.config.fault_tolerance:
                        print("Falling back to single-process training...")
                        # Fallback to single process
                        model = get_model_func()
                        if self.device_list[0] != 'cpu':
                            device = torch.device(f"cuda:{self.device_list[0]}")
                            model = model.to(device)
                        
                        results = model.meta_train(train_tasks, val_tasks, **args)
                        results['save_path'] = save_path
                        results['task_idx'] = task_idx
                        results['repeat_idx'] = repeat_idx
                        results['distributed'] = False
                        results['fallback'] = True
                        all_results[f"{task_idx}-{repeat_idx}"] = results
                    else:
                        raise
                
                self.timer.end(f"distributed_task_{task_idx}_{repeat_idx}")
                
                print(f"*****  End distributed meta-task #{task_idx}-{repeat_idx}  *****")
        
        return all_results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of distributed training.
        
        Returns:
            Performance summary
        """
        summary = {
            'timing': {},
            'memory': {},
            'distributed_config': self.config.__dict__
        }
        
        # Timing summary
        for name in self.timer.elapsed_times:
            summary['timing'][name] = {
                'total_time': self.timer.get_total_time(name),
                'average_time': self.timer.get_average_time(name),
                'n_measurements': len(self.timer.elapsed_times[name])
            }
        
        # Memory summary
        for name in self.memory_tracker.memory_snapshots:
            summary['memory'][name] = self.memory_tracker.get_peak_memory(name)
        
        return summary
    
    def save_results(self, results: Dict[str, Any]):
        """Save training results to file.
        
        Args:
            results: Training results
        """
        results_path = f"runs/{self.exp_name}/results.json"
        
        # Add performance summary
        results['performance_summary'] = self.get_performance_summary()
        
        # Save results
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4, default=lambda x: str(x))
        
        print(f"Results saved to {results_path}")
    
    def summary(self):
        """Generate summary of meta-learning experiments."""
        from ..utils import summary
        
        # Use existing PINNacle summary functionality
        summary.summary(
            f"runs/{self.exp_name}", 
            len(self.meta_tasks), 
            self.repeat, 
            [args.get('meta_iterations', 10000) for _, args in self.meta_tasks]
        )
        
        # Add distributed-specific summary
        print("\n" + "="*50)
        print("DISTRIBUTED META-LEARNING SUMMARY")
        print("="*50)
        
        perf_summary = self.get_performance_summary()
        
        print(f"World size: {self.config.world_size}")
        print(f"Backend: {self.config.backend}")
        print(f"Task parallel: {self.config.task_parallel}")
        print(f"Data parallel: {self.config.data_parallel}")
        
        if perf_summary['timing']:
            print("\nTiming Summary:")
            for name, timing in perf_summary['timing'].items():
                print(f"  {name}: {timing['total_time']:.2f}s total, "
                      f"{timing['average_time']:.2f}s average")
        
        if perf_summary['memory']:
            print("\nMemory Summary:")
            for name, memory in perf_summary['memory'].items():
                if memory['allocated'] > 0:
                    print(f"  {name}: {memory['allocated']/1e9:.2f}GB peak allocated")


def distributed_meta_train_process(rank: int, world_size: int, config: DistributedMetaPINNConfig,
                                  task_data: bytes, save_path: str, seed: int):
    """Process function for distributed meta-learning training.
    
    This function is called by each process in distributed training and follows
    PINNacle's existing process-based training pattern.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        config: Distributed training configuration
        task_data: Serialized task data
        save_path: Path to save results
        seed: Random seed
    """
    # Setup logging (following PINNacle's pattern)
    from trainer import HookedStdout
    
    hooked = HookedStdout(f"{save_path}/log_{rank}.txt")
    sys.stdout = hooked
    sys.stderr = HookedStdout(f"{save_path}/logerr_{rank}.txt", sys.stderr)
    
    try:
        # Setup distributed environment
        device = setup_distributed_environment(rank, world_size, config.backend,
                                             config.master_addr, config.master_port)
        
        # Set random seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Setup DeepXDE (following PINNacle's pattern)
        import deepxde as dde
        if device.type == 'cuda':
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            torch.set_default_tensor_type(torch.FloatTensor)
        dde.config.set_default_float('float32')
        dde.config.set_random_seed(seed)
        
        # Deserialize task data
        get_model_func, train_tasks, val_tasks, meta_train_args = dill.loads(task_data)
        
        # Create distributed trainer
        distributed_trainer = DistributedMetaPINN(config)
        distributed_trainer.setup_distributed(rank, world_size)
        
        # Create model
        model = get_model_func()
        model = distributed_trainer.create_model(model.config)
        
        # Run distributed training
        results = distributed_trainer.distributed_meta_train(
            train_tasks, val_tasks, **meta_train_args
        )
        
        # Save results (only on rank 0)
        if rank == 0:
            torch.save(results, f"{save_path}/distributed_results.pt")
        
        # Cleanup
        distributed_trainer.cleanup()
        
    except Exception as e:
        print(f"Process {rank} failed: {e}")
        raise
    finally:
        cleanup_distributed()


class DistributedMetaTrainerFactory:
    """Factory for creating distributed meta-learning trainers."""
    
    @staticmethod
    def create_trainer(exp_name: str, device: str, 
                      distributed_config: Optional[DistributedMetaPINNConfig] = None) -> DistributedMetaTrainer:
        """Create distributed meta-learning trainer.
        
        Args:
            exp_name: Experiment name
            device: Device specification
            distributed_config: Distributed training configuration
            
        Returns:
            Distributed meta-learning trainer
        """
        if distributed_config is None:
            # Create default configuration
            device_list = device.split(",")
            distributed_config = DistributedMetaPINNConfig(
                world_size=len(device_list) if len(device_list) > 1 else 1
            )
        
        return DistributedMetaTrainer(exp_name, device, distributed_config)
    
    @staticmethod
    def extend_existing_trainer(trainer, distributed_config: DistributedMetaPINNConfig) -> DistributedMetaTrainer:
        """Extend existing PINNacle trainer with distributed meta-learning capabilities.
        
        Args:
            trainer: Existing PINNacle trainer
            distributed_config: Distributed training configuration
            
        Returns:
            Extended distributed meta-learning trainer
        """
        # Create distributed trainer with same configuration
        distributed_trainer = DistributedMetaTrainer(
            trainer.exp_name, 
            ",".join(trainer.device), 
            distributed_config
        )
        
        # Copy existing configuration
        distributed_trainer.repeat = trainer.repeat
        
        return distributed_trainer