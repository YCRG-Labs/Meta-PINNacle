"""Configuration classes for meta-learning that extend PINNacle's existing patterns."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import torch


@dataclass
class MetaLearningConfig:
    """Base configuration class for meta-learning models.
    
    This extends PINNacle's configuration patterns to include meta-learning
    specific parameters while maintaining compatibility with existing systems.
    """
    # Network architecture (following PINNacle patterns)
    layers: List[int] = field(default_factory=lambda: [2, 64, 64, 64, 1])
    activation: str = 'tanh'
    initializer: str = 'Glorot uniform'
    
    # Meta-learning specific parameters
    meta_lr: float = 0.001  # Meta-learning rate (outer loop)
    adapt_lr: float = 0.01  # Adaptation learning rate (inner loop)
    adaptation_steps: int = 5  # Number of gradient steps for adaptation
    meta_batch_size: int = 25  # Number of tasks per meta-batch
    
    # Physics loss configuration (extending PINNacle's loss system)
    physics_loss_weight: float = 1.0
    data_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    initial_loss_weight: float = 1.0
    
    # Training configuration
    meta_iterations: int = 10000
    validation_frequency: int = 100
    save_frequency: int = 1000
    
    # Device and precision
    device: str = 'cuda'
    dtype: torch.dtype = torch.float32
    
    # Reproducibility
    seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'layers': self.layers,
            'activation': self.activation,
            'initializer': self.initializer,
            'meta_lr': self.meta_lr,
            'adapt_lr': self.adapt_lr,
            'adaptation_steps': self.adaptation_steps,
            'meta_batch_size': self.meta_batch_size,
            'physics_loss_weight': self.physics_loss_weight,
            'data_loss_weight': self.data_loss_weight,
            'boundary_loss_weight': self.boundary_loss_weight,
            'initial_loss_weight': self.initial_loss_weight,
            'meta_iterations': self.meta_iterations,
            'validation_frequency': self.validation_frequency,
            'save_frequency': self.save_frequency,
            'device': self.device,
            'dtype': str(self.dtype),
            'seed': self.seed
        }


@dataclass
class MetaPINNConfig(MetaLearningConfig):
    """Configuration for basic MetaPINN (MAML-based) model."""
    
    # MAML specific parameters
    first_order: bool = True  # Use first-order MAML for efficiency
    allow_unused: bool = True  # Allow unused parameters in gradient computation
    allow_nograd: bool = True  # Allow parameters without gradients
    
    # Gradient clipping
    grad_clip_norm: Optional[float] = None
    meta_grad_clip_norm: Optional[float] = None


@dataclass
class PhysicsInformedMetaLearnerConfig(MetaPINNConfig):
    """Configuration for enhanced PhysicsInformedMetaLearner model."""
    
    # Adaptive constraint weighting
    adaptive_constraint_weighting: bool = True
    constraint_balancer_type: str = 'dynamic'  # 'static', 'dynamic', 'learned'
    constraint_update_frequency: int = 10
    
    # Physics regularization
    physics_regularization_weight: float = 0.01
    physics_consistency_weight: float = 0.1
    
    # Multi-scale handling
    multi_scale_handling: bool = False
    scale_factors: List[float] = field(default_factory=lambda: [1.0, 0.1, 0.01])
    
    # Uncertainty estimation
    uncertainty_estimation: bool = False
    uncertainty_method: str = 'ensemble'  # 'ensemble', 'dropout', 'bayesian'
    n_uncertainty_samples: int = 10


@dataclass
class TransferLearningPINNConfig(MetaLearningConfig):
    """Configuration for transfer learning baseline model."""
    
    # Pre-training configuration
    pretrain_tasks: int = 100
    pretrain_epochs_per_task: int = 1000
    
    # Fine-tuning strategies
    fine_tune_strategy: str = 'full_fine_tuning'  # 'full_fine_tuning', 'feature_extraction', 'gradual_unfreezing'
    fine_tune_epochs: int = 100
    fine_tune_lr: float = 0.001
    
    # Layer freezing for gradual unfreezing
    unfreeze_schedule: List[int] = field(default_factory=lambda: [25, 50, 75])  # Epochs to unfreeze layers
    
    # Domain adaptation
    domain_adaptation: bool = False
    domain_adaptation_weight: float = 0.1
    source_distribution_cache_size: int = 1000


@dataclass
class DistributedMetaPINNConfig(MetaPINNConfig):
    """Configuration for distributed meta-learning training extending PINNacle's multi-GPU trainer."""
    
    # Distributed training parameters
    world_size: int = 1
    backend: str = 'nccl'  # 'nccl', 'gloo', 'mpi'
    master_addr: str = 'localhost'
    master_port: int = 12355
    timeout: int = 1800  # 30 minutes timeout
    
    # Task and data parallelism
    task_parallel: bool = True  # Distribute tasks across GPUs
    data_parallel: bool = True  # Use DDP for data parallelism
    
    # Memory optimization
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    gradient_accumulation_steps: int = 1
    
    # Communication and synchronization
    sync_frequency: int = 1  # Synchronize every N steps
    bucket_size_mb: int = 25
    find_unused_parameters: bool = True  # For complex meta-learning models
    
    # Fault tolerance
    fault_tolerance: bool = True
    checkpoint_frequency: int = 100
    max_retries: int = 3


@dataclass
class EvaluationConfig:
    """Configuration for few-shot evaluation framework."""
    
    # Few-shot evaluation parameters
    evaluation_shots: List[int] = field(default_factory=lambda: [1, 5, 10, 25])
    n_test_tasks: int = 50
    n_evaluation_runs: int = 5  # Multiple runs for statistical significance
    
    # Metrics configuration
    compute_l2_error: bool = True
    compute_relative_error: bool = True
    compute_physics_residual: bool = True
    
    # Timing measurements
    measure_adaptation_time: bool = True
    measure_inference_time: bool = True
    measure_memory_usage: bool = True
    
    # Statistical analysis
    confidence_level: float = 0.95
    statistical_tests: List[str] = field(default_factory=lambda: ['t_test', 'wilcoxon'])
    effect_size_metrics: List[str] = field(default_factory=lambda: ['cohens_d'])
    
    # Visualization
    generate_plots: bool = True
    plot_format: str = 'png'
    plot_dpi: int = 300


@dataclass
class ExperimentConfig:
    """Complete experiment configuration combining all components."""
    
    # Model configurations
    models: Dict[str, MetaLearningConfig] = field(default_factory=dict)
    
    # Problem configuration
    problems: List[str] = field(default_factory=lambda: [
        'heat_1d', 'burgers_2d', 'poisson_2d', 'navier_stokes', 'reaction_diffusion'
    ])
    
    # Task distribution
    n_train_tasks: int = 100
    n_val_tasks: int = 20
    n_test_tasks: int = 50
    
    # Evaluation configuration
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment metadata
    experiment_name: str = 'meta_learning_experiment'
    description: str = ''
    tags: List[str] = field(default_factory=list)
    
    # Output configuration
    output_dir: str = 'runs'
    save_models: bool = True
    save_results: bool = True
    
    def get_model_config(self, model_name: str) -> MetaLearningConfig:
        """Get configuration for specific model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in configuration")
        return self.models[model_name]
    
    def add_model_config(self, model_name: str, config: MetaLearningConfig):
        """Add model configuration."""
        self.models[model_name] = config
    
    @classmethod
    def create_default_experiment(cls) -> 'ExperimentConfig':
        """Create default experiment configuration with all models."""
        config = cls()
        
        # Add default model configurations
        config.add_model_config('StandardPINN', MetaLearningConfig())
        config.add_model_config('MetaPINN', MetaPINNConfig())
        config.add_model_config('PhysicsInformedMetaLearner', PhysicsInformedMetaLearnerConfig())
        config.add_model_config('TransferLearningPINN', TransferLearningPINNConfig())
        config.add_model_config('DistributedMetaPINN', DistributedMetaPINNConfig())
        
        return config


def parse_meta_learning_args(args) -> ExperimentConfig:
    """Parse command line arguments into meta-learning configuration.
    
    This function extends PINNacle's existing argument parsing patterns
    to handle meta-learning specific parameters.
    """
    config = ExperimentConfig()
    
    # Parse basic experiment parameters
    if hasattr(args, 'experiment_name'):
        config.experiment_name = args.experiment_name
    if hasattr(args, 'output_dir'):
        config.output_dir = args.output_dir
    
    # Parse model-specific parameters
    if hasattr(args, 'meta_lr'):
        for model_config in config.models.values():
            if hasattr(model_config, 'meta_lr'):
                model_config.meta_lr = args.meta_lr
    
    if hasattr(args, 'adapt_lr'):
        for model_config in config.models.values():
            if hasattr(model_config, 'adapt_lr'):
                model_config.adapt_lr = args.adapt_lr
    
    # Parse task distribution parameters
    if hasattr(args, 'n_train_tasks'):
        config.n_train_tasks = args.n_train_tasks
    if hasattr(args, 'n_test_tasks'):
        config.n_test_tasks = args.n_test_tasks
    
    return config