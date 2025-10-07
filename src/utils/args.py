import re
import argparse


def parse_hidden_layers(command_args):
    layers = []
    for s in re.split(r"[,_-]", command_args.hidden_layers):
        if '*' in s:
            siz, num = s.split('*')
            layers += [int(siz)] * int(num)
        else:
            layers += [int(s)]
    return layers


def parse_loss_weight(command_args):
    if command_args.loss_weight == '': return None
    weights = []
    for s in re.split(r"[,_-]", command_args.loss_weight):
        weights.append(float(s))
    return weights


def parse_evaluation_shots(command_args):
    """Parse comma-separated evaluation shots string"""
    if hasattr(command_args, 'evaluation_shots') and command_args.evaluation_shots:
        return [int(k.strip()) for k in command_args.evaluation_shots.split(",")]
    return [1, 5, 10, 25]  # Default shots


def parse_model_list(command_args, available_models):
    """Parse comma-separated model list or return all models"""
    if hasattr(command_args, 'models') and command_args.models != "all":
        model_names = [name.strip() for name in command_args.models.split(",")]
        return [(name, config) for name, config in available_models if name in model_names]
    return available_models


def parse_problem_list(command_args, available_problems):
    """Parse comma-separated problem list or return all problems"""
    if hasattr(command_args, 'problems') and command_args.problems != "all":
        problem_names = [name.strip() for name in command_args.problems.split(",")]
        problem_map = {cls.__name__: cls for cls in available_problems}
        return [problem_map[name] for name in problem_names if name in problem_map]
    return available_problems


def add_meta_learning_arguments(parser):
    """Add meta-learning specific arguments to argument parser"""
    meta_group = parser.add_argument_group('Meta-Learning Parameters')
    
    # Core meta-learning parameters
    meta_group.add_argument('--meta-lr', type=float, default=1e-3, 
                           help='Meta-learning rate for outer loop optimization')
    meta_group.add_argument('--adapt-lr', type=float, default=1e-2, 
                           help='Adaptation learning rate for inner loop optimization')
    meta_group.add_argument('--adaptation-steps', type=int, default=5, 
                           help='Number of gradient steps for task adaptation')
    meta_group.add_argument('--meta-batch-size', type=int, default=25, 
                           help='Number of tasks per meta-batch')
    meta_group.add_argument('--physics-weight', type=float, default=1.0, 
                           help='Weight for physics loss in meta-learning')
    meta_group.add_argument('--first-order', action='store_true', 
                           help='Use first-order MAML approximation')
    
    # Task generation parameters
    task_group = parser.add_argument_group('Task Generation Parameters')
    task_group.add_argument('--n-train-tasks', type=int, default=100, 
                           help='Number of training tasks to generate')
    task_group.add_argument('--n-val-tasks', type=int, default=20, 
                           help='Number of validation tasks to generate')
    task_group.add_argument('--n-test-tasks', type=int, default=50, 
                           help='Number of test tasks to generate')
    
    # Meta-training parameters
    training_group = parser.add_argument_group('Meta-Training Parameters')
    training_group.add_argument('--meta-iterations', type=int, default=10000, 
                               help='Number of meta-training iterations')
    training_group.add_argument('--val-frequency', type=int, default=100, 
                               help='Frequency of validation during meta-training')
    training_group.add_argument('--checkpoint-frequency', type=int, default=1000, 
                               help='Frequency of checkpointing during meta-training')
    
    # Evaluation parameters
    eval_group = parser.add_argument_group('Evaluation Parameters')
    eval_group.add_argument('--evaluation-shots', type=str, default="1,5,10,25", 
                           help='Comma-separated list of K-shot evaluations')
    eval_group.add_argument('--run-evaluation', action='store_true', 
                           help='Run few-shot evaluation after training')
    eval_group.add_argument('--evaluation-steps', type=int, default=5, 
                           help='Number of adaptation steps during evaluation')
    
    # Model and problem selection
    selection_group = parser.add_argument_group('Model and Problem Selection')
    selection_group.add_argument('--models', type=str, default="all", 
                                help='Comma-separated list of models to run or "all"')
    selection_group.add_argument('--problems', type=str, default="all", 
                                help='Comma-separated list of problems to run or "all"')
    
    # Advanced meta-learning options
    advanced_group = parser.add_argument_group('Advanced Meta-Learning Options')
    advanced_group.add_argument('--adaptive-weighting', action='store_true', 
                               help='Use adaptive constraint weighting in PhysicsInformedMetaLearner')
    advanced_group.add_argument('--physics-regularization', type=float, default=0.01, 
                               help='Physics regularization weight for meta-parameters')
    advanced_group.add_argument('--uncertainty-estimation', action='store_true', 
                               help='Enable uncertainty estimation during adaptation')
    advanced_group.add_argument('--multi-scale-handling', action='store_true', 
                               help='Enable multi-scale physics handling')
    
    # Transfer learning specific parameters
    transfer_group = parser.add_argument_group('Transfer Learning Parameters')
    transfer_group.add_argument('--pretrain-epochs', type=int, default=5000, 
                               help='Number of pre-training epochs for transfer learning')
    transfer_group.add_argument('--finetune-strategy', type=str, default='full', 
                               choices=['full', 'feature_extraction', 'gradual_unfreezing'],
                               help='Fine-tuning strategy for transfer learning')
    transfer_group.add_argument('--finetune-epochs', type=int, default=1000, 
                               help='Number of fine-tuning epochs')
    
    # Distributed training parameters
    distributed_group = parser.add_argument_group('Distributed Training Parameters')
    distributed_group.add_argument('--distributed', action='store_true', 
                                  help='Enable distributed meta-learning training')
    distributed_group.add_argument('--world-size', type=int, default=1, 
                                  help='Number of processes for distributed training')
    distributed_group.add_argument('--rank', type=int, default=0, 
                                  help='Rank of current process in distributed training')
    distributed_group.add_argument('--dist-backend', type=str, default='nccl', 
                                  help='Distributed training backend')
    distributed_group.add_argument('--dist-url', type=str, default='env://', 
                                  help='URL for distributed training initialization')
    
    return parser


def validate_meta_learning_args(args):
    """Validate meta-learning arguments and set defaults"""
    # Validate evaluation shots
    try:
        shots = parse_evaluation_shots(args)
        if not all(k > 0 for k in shots):
            raise ValueError("All evaluation shots must be positive integers")
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid evaluation shots: {e}")
    
    # Validate learning rates
    if args.meta_lr <= 0:
        raise argparse.ArgumentTypeError("Meta learning rate must be positive")
    if args.adapt_lr <= 0:
        raise argparse.ArgumentTypeError("Adaptation learning rate must be positive")
    
    # Validate task counts
    if args.n_train_tasks <= 0:
        raise argparse.ArgumentTypeError("Number of training tasks must be positive")
    if args.n_val_tasks <= 0:
        raise argparse.ArgumentTypeError("Number of validation tasks must be positive")
    if args.n_test_tasks <= 0:
        raise argparse.ArgumentTypeError("Number of test tasks must be positive")
    
    # Validate adaptation steps
    if args.adaptation_steps <= 0:
        raise argparse.ArgumentTypeError("Adaptation steps must be positive")
    
    # Validate meta-batch size
    if args.meta_batch_size <= 0:
        raise argparse.ArgumentTypeError("Meta-batch size must be positive")
    if args.meta_batch_size > args.n_train_tasks:
        raise argparse.ArgumentTypeError("Meta-batch size cannot exceed number of training tasks")
    
    return args
