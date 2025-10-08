#!/usr/bin/env python3
"""
Comprehensive Meta-Learning Benchmark Execution Script

This script executes the comprehensive meta-learning benchmark as specified in task 11.1:
- Execute all 5 models on all parametric PDE problems using extended PINNacle
- Generate comprehensive performance comparison results
- Validate statistical significance of meta-learning advantages

Requirements: 8.1, 8.2, 8.3, 8.4
"""

import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append("src")

# Set environment for PyTorch backend
os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

# Import PINNacle infrastructure
from trainer import Trainer
from meta_benchmark import (
    meta_pde_list,
    meta_model_list,
    create_meta_learning_task,
    add_meta_learning_arguments,
)

# Import meta-learning evaluation components
from src.meta_learning.evaluation_framework import MetaLearningEvaluationFramework
from src.meta_learning.statistical_analyzer import StatisticalAnalyzer
from src.meta_learning.computational_analyzer import ComputationalAnalyzer
from src.utils.meta_logging import MetaLearningLogger


class ComprehensiveBenchmarkExecutor:
    """Executes comprehensive meta-learning benchmark across all models and problems"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiment_name = config.get("experiment_name", "comprehensive_benchmark")
        self.results_dir = Path(
            config.get("results_dir", f"runs/{self.experiment_name}")
        )
        self.device = config.get("device", "0")
        self.seed = config.get("seed", 42)

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "benchmark.log"),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Initialize evaluation framework
        self.evaluation_framework = MetaLearningEvaluationFramework(
            output_dir=str(self.results_dir / "evaluation"),
            evaluation_shots=[1, 5, 10, 25],
            confidence_level=0.95,
            device=self.device,
            verbose=True,
        )

        # Results storage
        self.benchmark_results = {}
        self.trained_models = {}
        self.experiment_metadata = {}

    def setup_experiment(self, args):
        """Setup experiment configuration and environment"""
        self.logger.info("Setting up comprehensive benchmark experiment")

        # Set random seeds for reproducibility
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
            dde.config.set_random_seed(self.seed)

        # Save experiment configuration
        config_dict = vars(args)
        config_path = self.results_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        # Log system information
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"PyTorch version: {torch.__version__}")
        self.logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            self.logger.info(f"CUDA device count: {torch.cuda.device_count()}")

        self.experiment_metadata = {
            "start_time": time.time(),
            "config": config_dict,
            "system_info": {
                "pytorch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "device": self.device,
            },
        }

    def execute_single_experiment(
        self, model_name: str, model_config_class, pde_class, args
    ) -> Dict[str, Any]:
        """Execute a single model-problem combination experiment"""
        experiment_id = f"{model_name}_{pde_class.__name__}"
        self.logger.info(f"Starting experiment: {experiment_id}")

        start_time = time.time()

        try:
            # Create model task
            get_model = create_meta_learning_task(
                pde_class, model_name, model_config_class, args
            )

            # Setup trainer for this specific experiment
            date_str = time.strftime("%m.%d-%H.%M.%S", time.localtime())
            trainer = Trainer(f"{date_str}-{experiment_id}", self.device)

            # Determine training arguments based on model type
            if model_name == "StandardPINN":
                train_args = {
                    "iterations": args.iter,
                    "display_every": args.log_every,
                    "callbacks": [],
                }
            else:
                train_args = {
                    "meta_iterations": args.meta_iterations,
                    "display_every": args.log_every,
                    "evaluation_shots": [
                        int(k) for k in args.evaluation_shots.split(",")
                    ],
                    "run_evaluation": False,  # We'll do comprehensive evaluation separately
                    "callbacks": [],
                }

            # Add task and run training
            trainer.add_task(get_model, train_args)
            trainer.setup(__file__, self.seed)
            trainer.set_repeat(1)  # Single run for comprehensive evaluation

            # Run training
            self.logger.info(f"Training {experiment_id}...")
            trainer.train_all()

            # Get trained model for evaluation
            model = get_model()

            # Store trained model
            self.trained_models[experiment_id] = model

            training_time = time.time() - start_time

            # Store experiment results
            experiment_results = {
                "model_name": model_name,
                "problem_name": pde_class.__name__,
                "training_time": training_time,
                "status": "completed",
                "config": vars(args),
            }

            self.logger.info(
                f"Completed experiment: {experiment_id} in {training_time:.2f}s"
            )
            return experiment_results

        except Exception as e:
            self.logger.error(f"Failed experiment {experiment_id}: {str(e)}")
            return {
                "model_name": model_name,
                "problem_name": pde_class.__name__,
                "training_time": time.time() - start_time,
                "status": "failed",
                "error": str(e),
            }

    def execute_all_experiments(self, args) -> Dict[str, Any]:
        """Execute all model-problem combinations"""
        self.logger.info("Starting execution of all experiments")

        # Parse model and problem selections (use all by default)
        selected_models = meta_model_list
        selected_problems = meta_pde_list

        total_experiments = len(selected_models) * len(selected_problems)
        self.logger.info(f"Total experiments to execute: {total_experiments}")

        experiment_results = []
        completed_experiments = 0

        # Execute all model-problem combinations
        for pde_class in selected_problems:
            for model_name, model_config_class in selected_models:
                self.logger.info(
                    f"Progress: {completed_experiments + 1}/{total_experiments}"
                )

                result = self.execute_single_experiment(
                    model_name, model_config_class, pde_class, args
                )
                experiment_results.append(result)
                completed_experiments += 1

                # Save intermediate results
                self._save_intermediate_results(experiment_results)

        # Organize results by model and problem
        organized_results = {}
        for result in experiment_results:
            if result["status"] == "completed":
                model_name = result["model_name"]
                problem_name = result["problem_name"]

                if model_name not in organized_results:
                    organized_results[model_name] = {}
                organized_results[model_name][problem_name] = result

        self.benchmark_results = organized_results

        self.logger.info(f"Completed {completed_experiments} experiments")
        self.logger.info(
            f"Successful experiments: {len([r for r in experiment_results if r['status'] == 'completed'])}"
        )
        self.logger.info(
            f"Failed experiments: {len([r for r in experiment_results if r['status'] == 'failed'])}"
        )

        return {
            "experiment_results": experiment_results,
            "organized_results": organized_results,
            "summary": {
                "total_experiments": total_experiments,
                "completed_experiments": len(
                    [r for r in experiment_results if r["status"] == "completed"]
                ),
                "failed_experiments": len(
                    [r for r in experiment_results if r["status"] == "failed"]
                ),
            },
        }

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive evaluation on all trained models"""
        self.logger.info("Starting comprehensive evaluation of all trained models")

        if not self.trained_models:
            self.logger.error("No trained models available for evaluation")
            return {}

        # Organize models by problem for evaluation
        evaluation_results = {}

        # Group models by problem
        problems_models = {}
        for experiment_id, model in self.trained_models.items():
            model_name, problem_name = experiment_id.split("_", 1)

            if problem_name not in problems_models:
                problems_models[problem_name] = {}
            problems_models[problem_name][model_name] = model

        # Evaluate each problem separately
        for problem_name, models in problems_models.items():
            self.logger.info(
                f"Evaluating problem: {problem_name} with {len(models)} models"
            )

            # Get test tasks from any model (they should all have the same test tasks)
            test_tasks = None
            for model in models.values():
                if hasattr(model, "test_tasks") and model.test_tasks:
                    test_tasks = model.test_tasks
                    break

            if test_tasks is None:
                self.logger.warning(f"No test tasks found for problem: {problem_name}")
                continue

            # Run comprehensive evaluation
            problem_evaluation = self.evaluation_framework.evaluate_comprehensive(
                models=models, test_tasks=test_tasks
            )

            evaluation_results[problem_name] = problem_evaluation

        self.logger.info("Comprehensive evaluation completed")
        return evaluation_results

    def validate_statistical_significance(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate statistical significance of meta-learning advantages"""
        self.logger.info(
            "Validating statistical significance of meta-learning advantages"
        )

        significance_validation = {}

        for problem_name, problem_results in evaluation_results.items():
            if "statistical_comparisons" not in problem_results:
                continue

            statistical_comparisons = problem_results["statistical_comparisons"]

            # Analyze meta-learning vs standard comparisons
            meta_vs_standard_comparisons = {}
            meta_learning_advantages = []

            for comparison_name, comparison in statistical_comparisons.items():
                # Check if this is a meta-learning vs standard comparison
                if (
                    any(
                        meta_term in comparison_name.lower()
                        for meta_term in ["meta", "maml", "physics"]
                    )
                    and "standard" in comparison_name.lower()
                ):

                    meta_vs_standard_comparisons[comparison_name] = comparison

                    # Check for significant advantages
                    for K, shot_comparison in comparison.shot_comparisons.items():
                        if shot_comparison.get("paired_t_test", {}).get(
                            "significant", False
                        ):
                            # Check if meta-learning is better (negative effect size means first model is better)
                            effect_size = shot_comparison.get("effect_size", {}).get(
                                "cohens_d", 0
                            )
                            if (
                                effect_size < 0
                            ):  # Meta-learning model is first in comparison
                                meta_learning_advantages.append(
                                    {
                                        "comparison": comparison_name,
                                        "K_shots": K,
                                        "p_value": shot_comparison["paired_t_test"][
                                            "p_value"
                                        ],
                                        "effect_size": abs(effect_size),
                                        "relative_improvement": shot_comparison[
                                            "descriptive_statistics"
                                        ]["relative_improvement_percent"],
                                    }
                                )

            significance_validation[problem_name] = {
                "meta_vs_standard_comparisons": meta_vs_standard_comparisons,
                "significant_advantages": meta_learning_advantages,
                "total_significant_advantages": len(meta_learning_advantages),
                "validation_summary": {
                    "meta_learning_shows_advantage": len(meta_learning_advantages) > 0,
                    "strong_advantages": len(
                        [
                            adv
                            for adv in meta_learning_advantages
                            if adv["effect_size"] >= 0.8
                        ]
                    ),
                    "medium_advantages": len(
                        [
                            adv
                            for adv in meta_learning_advantages
                            if 0.5 <= adv["effect_size"] < 0.8
                        ]
                    ),
                    "small_advantages": len(
                        [
                            adv
                            for adv in meta_learning_advantages
                            if 0.2 <= adv["effect_size"] < 0.5
                        ]
                    ),
                },
            }

        # Overall validation summary
        total_advantages = sum(
            len(v["significant_advantages"]) for v in significance_validation.values()
        )
        problems_with_advantages = len(
            [
                v
                for v in significance_validation.values()
                if v["validation_summary"]["meta_learning_shows_advantage"]
            ]
        )

        overall_validation = {
            "total_significant_advantages": total_advantages,
            "problems_with_meta_learning_advantages": problems_with_advantages,
            "total_problems_evaluated": len(significance_validation),
            "meta_learning_advantage_rate": (
                problems_with_advantages / len(significance_validation)
                if significance_validation
                else 0
            ),
            "validation_conclusion": (
                "Meta-learning shows significant advantages"
                if problems_with_advantages > len(significance_validation) / 2
                else "Mixed or no clear advantages"
            ),
        }

        self.logger.info(
            f"Statistical validation completed: {total_advantages} significant advantages found"
        )
        self.logger.info(
            f"Meta-learning advantage rate: {overall_validation['meta_learning_advantage_rate']:.2%}"
        )

        return {
            "problem_validations": significance_validation,
            "overall_validation": overall_validation,
        }

    def generate_comprehensive_performance_comparison(
        self, evaluation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive performance comparison results"""
        self.logger.info("Generating comprehensive performance comparison results")

        performance_comparison = {
            "model_rankings": {},
            "problem_summaries": {},
            "cross_problem_analysis": {},
            "performance_matrices": {},
        }

        # Analyze performance for each problem
        for problem_name, problem_results in evaluation_results.items():
            if "few_shot_results" not in problem_results:
                continue

            few_shot_results = problem_results["few_shot_results"]

            # Create performance matrix for this problem
            performance_matrix = {}
            model_rankings = {}

            for K in [1, 5, 10, 25]:
                K_performance = {}
                for model_name, model_results in few_shot_results.items():
                    if K in model_results.shot_results:
                        error = model_results.shot_results[K].get(
                            "mean_l2_error", np.inf
                        )
                        K_performance[model_name] = error

                # Rank models for this K
                if K_performance:
                    sorted_models = sorted(K_performance.items(), key=lambda x: x[1])
                    model_rankings[f"K_{K}"] = [model for model, _ in sorted_models]
                    performance_matrix[f"K_{K}"] = K_performance

            performance_comparison["model_rankings"][problem_name] = model_rankings
            performance_comparison["performance_matrices"][
                problem_name
            ] = performance_matrix

            # Problem summary
            if performance_matrix:
                best_models = {}
                for K, rankings in model_rankings.items():
                    if rankings:
                        best_models[K] = rankings[0]

                performance_comparison["problem_summaries"][problem_name] = {
                    "best_models_by_shot": best_models,
                    "models_evaluated": list(few_shot_results.keys()),
                    "evaluation_shots": list(performance_matrix.keys()),
                }

        # Cross-problem analysis
        all_models = set()
        for problem_results in performance_comparison["model_rankings"].values():
            for shot_rankings in problem_results.values():
                all_models.update(shot_rankings)

        # Calculate average rankings across problems
        model_avg_rankings = {}
        for model in all_models:
            rankings = []
            for problem_rankings in performance_comparison["model_rankings"].values():
                for shot_rankings in problem_rankings.values():
                    if model in shot_rankings:
                        rankings.append(
                            shot_rankings.index(model) + 1
                        )  # 1-based ranking

            if rankings:
                model_avg_rankings[model] = np.mean(rankings)

        # Overall best performers
        if model_avg_rankings:
            sorted_overall = sorted(model_avg_rankings.items(), key=lambda x: x[1])
            performance_comparison["cross_problem_analysis"] = {
                "overall_rankings": sorted_overall,
                "best_overall_model": sorted_overall[0][0] if sorted_overall else None,
                "average_rankings": model_avg_rankings,
            }

        self.logger.info("Performance comparison results generated")
        return performance_comparison

    def save_all_results(
        self,
        execution_results: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        significance_validation: Dict[str, Any],
        performance_comparison: Dict[str, Any],
    ):
        """Save all benchmark results to files"""
        self.logger.info("Saving all benchmark results")

        # Complete results
        complete_results = {
            "experiment_metadata": self.experiment_metadata,
            "execution_results": execution_results,
            "evaluation_results": evaluation_results,
            "significance_validation": significance_validation,
            "performance_comparison": performance_comparison,
            "benchmark_completion_time": time.time(),
        }

        # Save complete results
        with open(self.results_dir / "complete_benchmark_results.json", "w") as f:
            json.dump(complete_results, f, indent=2, default=str)

        # Save summary report
        self._generate_summary_report(complete_results)

        # Save individual components
        with open(self.results_dir / "execution_results.json", "w") as f:
            json.dump(execution_results, f, indent=2, default=str)

        with open(self.results_dir / "significance_validation.json", "w") as f:
            json.dump(significance_validation, f, indent=2, default=str)

        with open(self.results_dir / "performance_comparison.json", "w") as f:
            json.dump(performance_comparison, f, indent=2, default=str)

        self.logger.info(f"All results saved to: {self.results_dir}")

    def _generate_summary_report(self, complete_results: Dict[str, Any]):
        """Generate human-readable summary report"""
        report_path = self.results_dir / "benchmark_summary_report.md"

        with open(report_path, "w") as f:
            f.write("# Comprehensive Meta-Learning Benchmark Results\n\n")

            # Experiment overview
            metadata = complete_results["experiment_metadata"]
            f.write("## Experiment Overview\n")
            f.write(f"- Start Time: {time.ctime(metadata['start_time'])}\n")
            f.write(
                f"- Duration: {(complete_results['benchmark_completion_time'] - metadata['start_time']) / 3600:.2f} hours\n"
            )
            f.write(f"- Device: {metadata['system_info']['device']}\n")
            f.write(
                f"- PyTorch Version: {metadata['system_info']['pytorch_version']}\n\n"
            )

            # Execution summary
            exec_summary = complete_results["execution_results"]["summary"]
            f.write("## Execution Summary\n")
            f.write(f"- Total Experiments: {exec_summary['total_experiments']}\n")
            f.write(
                f"- Completed Successfully: {exec_summary['completed_experiments']}\n"
            )
            f.write(f"- Failed Experiments: {exec_summary['failed_experiments']}\n")
            f.write(
                f"- Success Rate: {exec_summary['completed_experiments'] / exec_summary['total_experiments'] * 100:.1f}%\n\n"
            )

            # Performance comparison
            perf_comparison = complete_results["performance_comparison"]
            if (
                "cross_problem_analysis" in perf_comparison
                and perf_comparison["cross_problem_analysis"]
            ):
                f.write("## Overall Performance Rankings\n")
                overall_rankings = perf_comparison["cross_problem_analysis"][
                    "overall_rankings"
                ]
                for i, (model, avg_rank) in enumerate(overall_rankings[:5]):  # Top 5
                    f.write(f"{i+1}. **{model}** (Average Rank: {avg_rank:.2f})\n")
                f.write("\n")

            # Statistical significance validation
            sig_validation = complete_results["significance_validation"][
                "overall_validation"
            ]
            f.write("## Statistical Significance Validation\n")
            f.write(
                f"- Total Significant Advantages: {sig_validation['total_significant_advantages']}\n"
            )
            f.write(
                f"- Problems with Meta-Learning Advantages: {sig_validation['problems_with_meta_learning_advantages']}\n"
            )
            f.write(
                f"- Meta-Learning Advantage Rate: {sig_validation['meta_learning_advantage_rate']:.1%}\n"
            )
            f.write(f"- **Conclusion**: {sig_validation['validation_conclusion']}\n\n")

            # Problem-specific results
            f.write("## Problem-Specific Results\n")
            for problem_name, problem_summary in perf_comparison.get(
                "problem_summaries", {}
            ).items():
                f.write(f"### {problem_name}\n")
                f.write(
                    f"- Models Evaluated: {len(problem_summary['models_evaluated'])}\n"
                )

                best_models = problem_summary.get("best_models_by_shot", {})
                for shot, best_model in best_models.items():
                    f.write(f"- Best Model ({shot}): {best_model}\n")
                f.write("\n")

            # Detailed results location
            f.write("## Detailed Results\n")
            f.write("- Complete Results: `complete_benchmark_results.json`\n")
            f.write("- Evaluation Details: `evaluation/`\n")
            f.write("- Individual Experiment Logs: Check trainer logs\n")

    def _save_intermediate_results(self, experiment_results: List[Dict]):
        """Save intermediate results during execution"""
        with open(self.results_dir / "intermediate_results.json", "w") as f:
            json.dump(experiment_results, f, indent=2, default=str)


def main():
    """Main execution function for comprehensive benchmark"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Meta-Learning Benchmark"
    )

    # Basic arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="comprehensive_meta_benchmark",
        help="Name of the benchmark experiment",
    )
    parser.add_argument(
        "--results-dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default="0", help="Device to use for training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    # Standard PINNacle arguments
    parser.add_argument("--hidden-layers", type=str, default="100*5")
    parser.add_argument("--loss-weight", type=str, default="")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--iter", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--plot-every", type=int, default=5000)
    parser.add_argument("--method", type=str, default="adam")

    # Add meta-learning arguments
    add_meta_learning_arguments(parser)

    # Override some defaults for comprehensive benchmark
    parser.set_defaults(
        n_train_tasks=100,
        n_val_tasks=20,
        n_test_tasks=50,
        meta_iterations=10000,
        evaluation_shots="1,5,10,25",
    )

    args = parser.parse_args()

    # Set default results directory
    if args.results_dir is None:
        args.results_dir = f"runs/{args.experiment_name}"

    print("=" * 80)
    print("COMPREHENSIVE META-LEARNING BENCHMARK")
    print("=" * 80)
    print(f"Experiment: {args.experiment_name}")
    print(f"Results Directory: {args.results_dir}")
    print(f"Device: {args.device}")
    print(f"Models: All 5 meta-learning models")
    print(f"Problems: All parametric PDE problems")
    print(f"Training Tasks: {args.n_train_tasks}")
    print(f"Test Tasks: {args.n_test_tasks}")
    print(f"Meta Iterations: {args.meta_iterations}")
    print(f"Evaluation Shots: {args.evaluation_shots}")
    print("=" * 80)

    # Initialize benchmark executor
    config = {
        "experiment_name": args.experiment_name,
        "results_dir": args.results_dir,
        "device": args.device,
        "seed": args.seed,
    }

    executor = ComprehensiveBenchmarkExecutor(config)

    try:
        # Setup experiment
        executor.setup_experiment(args)

        # Execute all experiments
        print("\nPhase 1: Executing all model-problem combinations...")
        execution_results = executor.execute_all_experiments(args)

        # Run comprehensive evaluation
        print("\nPhase 2: Running comprehensive evaluation...")
        evaluation_results = executor.run_comprehensive_evaluation()

        # Validate statistical significance
        print("\nPhase 3: Validating statistical significance...")
        significance_validation = executor.validate_statistical_significance(
            evaluation_results
        )

        # Generate performance comparison
        print("\nPhase 4: Generating performance comparison...")
        performance_comparison = executor.generate_comprehensive_performance_comparison(
            evaluation_results
        )

        # Save all results
        print("\nPhase 5: Saving results...")
        executor.save_all_results(
            execution_results,
            evaluation_results,
            significance_validation,
            performance_comparison,
        )

        print("\n" + "=" * 80)
        print("BENCHMARK COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Results saved to: {executor.results_dir}")
        print(f"Summary report: {executor.results_dir}/benchmark_summary_report.md")

        # Print key findings
        overall_validation = significance_validation["overall_validation"]
        print(f"\nKey Findings:")
        print(
            f"- Total Significant Advantages: {overall_validation['total_significant_advantages']}"
        )
        print(
            f"- Meta-Learning Advantage Rate: {overall_validation['meta_learning_advantage_rate']:.1%}"
        )
        print(f"- Conclusion: {overall_validation['validation_conclusion']}")

        if "cross_problem_analysis" in performance_comparison:
            best_model = performance_comparison["cross_problem_analysis"].get(
                "best_overall_model"
            )
            if best_model:
                print(f"- Best Overall Model: {best_model}")

    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        executor.logger.warning("Benchmark interrupted by user")
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        executor.logger.error(f"Benchmark failed: {str(e)}")
        raise
    finally:
        print(f"\nBenchmark logs saved to: {executor.results_dir}/benchmark.log")


if __name__ == "__main__":
    main()
