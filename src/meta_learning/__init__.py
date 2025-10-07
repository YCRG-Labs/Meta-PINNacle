"""Meta-learning extensions for PINNacle framework."""

__all__ = [
    "task", "config", "utils", "meta_pinn", 
    "physics_informed_meta_learner", "constraint_balancer", 
    "multi_scale_handler", "uncertainty_estimator", "transfer_learning_pinn",
    "distributed_meta_pinn", "memory_optimization", "distributed_coordination",
    "distributed_trainer_extension"
]

from . import task
from . import config
from . import utils
from . import meta_pinn
from . import physics_informed_meta_learner
from . import constraint_balancer
from . import multi_scale_handler
from . import uncertainty_estimator
from . import transfer_learning_pinn
from . import distributed_meta_pinn
from . import memory_optimization
from . import distributed_coordination
from . import distributed_trainer_extension

# Import main classes for convenience
from .meta_pinn import MetaPINN
from .physics_informed_meta_learner import PhysicsInformedMetaLearner
from .transfer_learning_pinn import TransferLearningPINN, TransferLearningPINNWithDomainAdaptation
from .constraint_balancer import (
    ConstraintBalancer, 
    StaticConstraintBalancer,
    DynamicConstraintBalancer, 
    LearnedConstraintBalancer,
    PhysicsRegularizer,
    create_constraint_balancer
)
from .multi_scale_handler import (
    MultiScaleHandler,
    ScaleAwarePDEExtension,
    extend_pde_with_multi_scale
)
from .uncertainty_estimator import (
    UncertaintyEstimator,
    EnsembleUncertaintyEstimator,
    DropoutUncertaintyEstimator,
    BayesianUncertaintyEstimator,
    UncertaintyGuidedSampler,
    UncertaintyAwareConstraintWeighting,
    UncertaintyAnalyzer,
    create_uncertainty_estimator
)