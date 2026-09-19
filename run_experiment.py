#!/usr/bin/env python3
"""Run distributionally robust feature-selection experiments.

This version allows stopping and resuming at any stage:
1. Data loading
2. Our method variable selection
3. Baseline method selection
4. Downstream evaluation

Usage:
  python run_experiment.py --populations linear_regression linear_regression
  python run_experiment.py --resume /path/to/run
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import torch
from typing import List, Dict, Any, Optional

# Import modular components
from dro_feature_selection.baseline_methods import BaselineMethods
from dro_feature_selection.checkpoint import CheckpointManager
from dro_feature_selection.data_loader import DataManager
from dro_feature_selection.downstream_eval import DownstreamEvaluator
from dro_feature_selection.variable_selector import VariableSelector

from visualize_training import plot_training_metrics, create_multi_series_horizontal_bars


def get_latest_run_number(save_path: str) -> int:
    if not os.path.exists(save_path): os.makedirs(save_path); return 0
    existing = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d)) and d.startswith('run_')]
    run_nums = [int(d.split('_')[1]) for d in existing if d.split('_')[1].isdigit()]
    return max(run_nums) + 1 if run_nums else 0

def convert_numpy_to_python(obj: Any) -> Any:
    """Convert numpy types to standard Python types for JSON compatibility"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_to_python(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(value) for value in obj]
    return obj

def run_pipeline(args, resume_path=None):
    """Run the full pipeline with modular stages and checkpointing"""
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Setup save path and checkpoint manager
    if resume_path:
        save_path = resume_path
        print(f"Resuming from: {save_path}")
    else:
        # Create a new run folder
        if args.populations[0].lower().startswith('baseline_failure'):
            base_save_path = os.path.join(args.save_path, f'{args.populations[0]}/')
        elif args.populations[0].lower().startswith('linear'):
            base_save_path = os.path.join(args.save_path, 'linear_regression/')
        elif args.populations[0].lower().startswith('sinusoidal'):  
            base_save_path = os.path.join(args.save_path, 'sinusoidal_regression/')
        elif args.populations[0].lower().startswith('cubic_regression'):
            base_save_path = os.path.join(args.save_path, 'cubic_regression/')
        else:
            base_save_path = args.save_path
        
        os.makedirs(base_save_path, exist_ok=True)
        run_no = get_latest_run_number(base_save_path)
        save_path = os.path.join(base_save_path, f'run_{run_no}/')
        os.makedirs(save_path, exist_ok=True)
        
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(save_path)
    
    # Build population configs
    pop_configs = [{'pop_id': i, 'dataset_type': dt} for i, dt in enumerate(args.populations)]
    
    # Save/load experiment parameters
    if not resume_path:
        # New run - save parameters
        experiment_params = vars(args).copy()
        experiment_params['pop_configs'] = pop_configs
        experiment_params['script_version'] = 'unified_checkpointed_pipeline'
        experiment_params['final_save_path'] = save_path
        
        print("\n--- Running distributionally robust feature-selection experiment ---")
        print(json.dumps(convert_numpy_to_python(experiment_params), indent=2))
        
        with open(os.path.join(save_path, 'experiment_params.json'), 'w') as f:
            json.dump(convert_numpy_to_python(experiment_params), f, indent=2)
            
        checkpoint_manager.store_run_params(experiment_params)
    else:
        # Resuming - load parameters
        params_path = os.path.join(save_path, 'experiment_params.json')
        legacy_params_path = os.path.join(save_path, 'params_v10.json')
        if not os.path.exists(params_path) and os.path.exists(legacy_params_path):
            params_path = legacy_params_path
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                experiment_params = json.load(f)
                print(f"Loaded parameters from: {params_path}")
        else:
            experiment_params = vars(args).copy()
            print(f"Warning: No parameters file found at {params_path}. Using command line args.")
    
    # ===== STAGE 1: DATA LOADING =====
    data_manager = DataManager(cache_dir="./data_cache")
    
    # Check if data is already loaded
    if checkpoint_manager.is_data_loaded():
        print("Data already loaded. Loading from checkpoint...")
        data_path = checkpoint_manager.get_data_path()
        try:
            with open(data_path, 'rb') as f:
                data_checkpoint = pickle.load(f)
                pop_data = data_checkpoint.get('pop_data')
                pop_data_test_val = data_checkpoint.get('pop_data_test_val')
                is_classification = data_checkpoint.get('is_classification', False)
            print(f"Successfully loaded data from: {data_path}")
        except Exception as e:
            print(f"Error loading data checkpoint: {e}. Regenerating data...")
            # Fall through to data generation
    
    if not checkpoint_manager.is_data_loaded() or 'pop_data' not in locals():
        # Load or generate dataset
        print("Generating or loading dataset from cache...")
        
        # Determine if this is a classification task
        is_classification = any('uci' in config['dataset_type'].lower() for config in pop_configs)
        
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        effective_dataset_size = (
            args.baseline_data_size
            if any('baseline' in config['dataset_type'].lower() for config in pop_configs)
            else args.dataset_size
        )
        pop_data, pop_data_test_val = data_manager.load_or_generate_data(
            pop_configs=pop_configs,
            m1=args.m1,
            m=args.m,
            dataset_size=effective_dataset_size,
            noise_scale=args.noise_scale,
            corr_strength=args.corr_strength,
            estimator_type=args.estimator_type,
            device=device,
            base_model_type=args.base_model_type,
            seed=args.seed,
            acs_data_fraction=args.acs_data_fraction,
            force_regenerate=args.force_regenerate_data,
            uci_populations=args.uci_populations,
            is_classification=is_classification
        )
        
        # Save data checkpoint
        data_checkpoint = {
            'pop_data': pop_data,
            'pop_data_test_val': pop_data_test_val,
            'is_classification': is_classification
        }
        data_path = os.path.join(save_path, 'data_checkpoint.pkl')
        with open(data_path, 'wb') as f:
            pickle.dump(data_checkpoint, f)
        
        checkpoint_manager.mark_data_loaded(data_path)
        print(f"Data generated and saved to: {data_path}")
    
    # Set the correct dimension for UCI data
    if is_classification and len(pop_data) > 0:
        args.m = pop_data[0]['X_raw'].shape[1]
        print(f"Data dimension set to {args.m}")
    
    # Determine budget
    budget = args.budget
    if budget is None:
        # Calculate budget based on meaningful indices if available
        meaningful_indices_list = [pop.get('meaningful_indices') for pop in pop_data]
        if all(isinstance(indices, (list, tuple)) for indices in meaningful_indices_list if indices is not None):
            all_meaningful_indices = set()
            for indices in meaningful_indices_list:
                if indices is not None:
                    all_meaningful_indices.update(indices)
            budget = len(all_meaningful_indices)
            print(f"Budget set to the size of union of meaningful indices: {budget}")
        else:
            budget = min(args.m, max(1, args.m1 // 2) + len(pop_configs) * (args.m1 - max(1, args.m1 // 2)))
            print(f"Budget set to: {budget}")
    
    # ===== STAGE 2: OUR METHOD VARIABLE SELECTION =====
    variable_selector = VariableSelector(checkpoint_manager)
    
    if checkpoint_manager.is_our_method_complete():
        print("Our method already completed. Loading results...")
        our_method_results = checkpoint_manager.load_checkpoint('our_method')
        if not our_method_results:
            print("Warning: Could not load our method results. Re-running.")
    
    if not checkpoint_manager.is_our_method_complete() or 'our_method_results' not in locals():
        print("Running our variable selection method...")
        scheduler_kwargs = {}
        if args.scheduler_type == 'StepLR':
            scheduler_kwargs = {'step_size': args.scheduler_step_size, 'gamma': args.scheduler_gamma}
        elif args.scheduler_type == 'MultiStepLR':
            scheduler_kwargs = {'milestones': args.scheduler_milestones, 'gamma': args.scheduler_gamma}
        elif args.scheduler_type == 'ExponentialLR':
            scheduler_kwargs = {'gamma': args.scheduler_gamma}
        elif args.scheduler_type == 'CosineAnnealingLR':
            scheduler_kwargs = {
                'T_max': args.num_epochs if args.scheduler_t_max < 0 else args.scheduler_t_max,
                'eta_min': args.scheduler_min_lr,
            }
        elif args.scheduler_type == 'ReduceLROnPlateau':
            scheduler_kwargs = {
                'factor': args.scheduler_gamma,
                'patience': args.scheduler_patience,
                'min_lr': args.scheduler_min_lr,
            }

        our_method_params = {
            'save_path': save_path,
            'parameterization': args.parameterization,
            'learning_rate': args.learning_rate,
            'optimizer_type': args.optimizer_type,
            'alpha_init': args.alpha_init,
            'num_epochs': args.num_epochs,
            'penalty_type': args.penalty_type,
            'penalty_lambda': args.penalty_lambda,
            'early_stopping_patience': args.patience,
            'param_freezing': args.param_freezing,
            'smooth_minmax': args.smooth_minmax,
            'gradient_mode': args.gradient_mode,
            't2_estimator_type': args.t2_estimator_type,
            'N_grad_samples': args.N_grad_samples,
            'use_baseline': args.use_baseline,
            'estimator_type': args.estimator_type,
            'base_model_type': args.base_model_type,
            'objective_value_estimator': args.objective_value_estimator,
            'k_kernel': args.k_kernel,
            'seed': args.seed,
            'scheduler_type': args.scheduler_type,
            'scheduler_kwargs': scheduler_kwargs,
        }
        
        our_method_results = variable_selector.run_selection(
            pop_data=pop_data,
            m1=args.m1,
            m=args.m,
            budget=budget,
            params=our_method_params
        )
        print("Our method variable selection complete.")
    
    # ===== STAGE 3: BASELINE METHODS =====
    baseline_methods = BaselineMethods(checkpoint_manager)
    
    if checkpoint_manager.is_baselines_complete():
        print("Baseline methods already completed. Loading results...")
        baseline_results = checkpoint_manager.load_checkpoint('baselines')
        if not baseline_results:
            print("Warning: Could not load baseline results. Re-running.")
    
    if not checkpoint_manager.is_baselines_complete() or 'baseline_results' not in locals():
        print("Running baseline selection methods...")
        baseline_params = {
            'alpha_lasso': args.lasso_alpha,
            'seed': args.seed
        }
        
        baseline_results = baseline_methods.run_all_baselines(
            pop_data=pop_data,
            budget=budget,
            params=baseline_params,
            is_classification=is_classification
        )
        print("Baseline methods complete.")
    
    # ===== STAGE 4: DOWNSTREAM EVALUATION =====
    downstream_evaluator = DownstreamEvaluator(checkpoint_manager)
    
    if checkpoint_manager.is_downstream_eval_complete():
        print("Downstream evaluation already completed. Loading results...")
        downstream_results = checkpoint_manager.load_checkpoint('downstream_eval')
        if not downstream_results:
            print("Warning: Could not load downstream evaluation results. Re-running.")
    
    if not checkpoint_manager.is_downstream_eval_complete() or 'downstream_results' not in locals():
        print("Running downstream evaluation...")
        downstream_results = downstream_evaluator.evaluate_selections(
            pop_data_test_val=pop_data_test_val,
            our_method_results=our_method_results,
            baseline_results=baseline_results,
            budget=budget,
            seed=args.seed,
            is_classification=is_classification,
            save_path=save_path
        )
        print("Downstream evaluation complete.")
    
    print(f"\n=== All stages completed successfully! ===")
    print(f"Results stored in: {save_path}")
    
    # # see if results_comparison_budget_...csv exits, if yes, create_multi_series_horizontal_bars(df, output_dir, file_prefix)
    # if os.path.exists(os.path.join(save_path, 'results_comparison_budget_*.csv')):
    #     # create_multi_series_horizontal_bars(df, output_dir, file_prefix)
    #     df= pd.read_csv(os.path.join(save_path, 'results_comparison_budget_*.csv'))
    #     create_multi_series_horizontal_bars(save_path, file_prefix='results_comparison_budget_')
    # visualize training metrics: plot_training_metrics(results, save_dir)
    plot_training_metrics(our_method_results, save_dir=save_path)

    return {
        'save_path': save_path,
        'our_method_results': our_method_results,
        'baseline_results': baseline_results,
        'downstream_results': downstream_results
    }

def parse_args():
    parser = argparse.ArgumentParser(description='Distributionally robust feature-selection experiments')
    # Add resume functionality
    parser.add_argument('--resume', type=str, default=None, help='Resume from a specific run path')
    parser.add_argument('--save-path', '--save_path', dest='save_path', type=str, default='./results/', help='Path to save results')
    parser.add_argument('--populations', type=str, nargs='+', help='List of population types')
    parser.add_argument('--m1', type=int, default=10, help='Number of variables to select')
    parser.add_argument('--m', type=int, default=100, help='Total number of variables')
    parser.add_argument('--dataset-size', '--dataset_size', dest='dataset_size', type=int, default=1000, help='Size of the dataset')
    parser.add_argument('--baseline-data-size', '--baseline_data_size', dest='baseline_data_size', type=int, default=15000, help='Dataset size for baseline-failure experiments')
    parser.add_argument('--noise-scale', '--noise_scale', dest='noise_scale', type=float, default=0.1, help='Noise scale for data generation')
    parser.add_argument('--corr-strength', '--corr_strength', dest='corr_strength', type=float, default=0.0, help='Correlation strength for data generation')
    parser.add_argument('--estimator-type', '--estimator_type', dest='estimator_type', type=str, default='if', choices=['plugin', 'if'], help='Estimator used for E[Y|X]')
    parser.add_argument('--base-model-type', '--base_model_type', dest='base_model_type', type=str, default='rf', choices=['rf', 'krr', 'xgb'], help='Base model used by the estimator')
    parser.add_argument('--acs-data-fraction', '--acs_data_fraction', dest='acs_data_fraction', type=float, default=0.2, help='Fraction of ACS data')
    parser.add_argument('--force-regenerate-data', '--force_regenerate_data', dest='force_regenerate_data', action='store_true', help='Force data regeneration')
    parser.add_argument('--budget', type=int, default=None, help='Budget for variable selection')
    parser.add_argument('--parameterization', type=str, default='alpha', choices=['alpha', 'theta'], help='Parameter to optimize (alpha or theta)')
    parser.add_argument('--learning-rate', '--learning_rate', dest='learning_rate', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--optimizer-type', '--optimizer_type', dest='optimizer_type', type=str, default='adam', choices=['sgd', 'adam'], help='Optimizer type')
    parser.add_argument('--alpha-init', '--alpha_init', dest='alpha_init', type=str, default='random_1', help='Config for initial alpha value (random_1, random_2, ones, adaptive)')
    parser.add_argument('--num-epochs', '--num_epochs', dest='num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--penalty-type', '--penalty_type', dest='penalty_type', type=str, default='Reciprocal_L1', choices=['Reciprocal_L1', 'Neg_L1', 'Max_Dev', 'Quadratic_Barrier', 'Exponential', 'None'], help='Type of penalty to use')
    parser.add_argument('--penalty-lambda', '--penalty_lambda', dest='penalty_lambda', type=float, default=0.001, help='Lambda for penalty')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--param-freezing', '--param_freezing', dest='param_freezing', action=argparse.BooleanOptionalAction, default=True, help='Enable parameter freezing')
    parser.add_argument('--smooth-minmax', '--smooth_minmax', dest='smooth_minmax', type=float, default=float('inf'), help='Beta param for SmoothMax objective (inf for hard max)')
    parser.add_argument('--gradient-mode', '--gradient_mode', dest='gradient_mode', type=str, default='autograd', choices=['autograd', 'reinforce'], help='Gradient mode')
    parser.add_argument('--t2-estimator-type', '--t2_estimator_type', dest='t2_estimator_type', type=str, default='kernel_if_like', choices=['mc_plugin', 'kernel_if_like'], help='T2 estimator type for gradient')
    parser.add_argument('--N-grad-samples', '--N_grad_samples', dest='N_grad_samples', type=int, default=25, help='MC samples for gradient estimation')
    parser.add_argument('--use-baseline', '--use_baseline', dest='use_baseline', action=argparse.BooleanOptionalAction, default=True, help='Use baseline for REINFORCE optimization')
    parser.add_argument('--objective-value-estimator', '--objective_value_estimator', dest='objective_value_estimator', type=str, default='if', choices=['if', 'mc'], help='Estimator for tracking objective value')
    parser.add_argument('--k-kernel', '--k_kernel', dest='k_kernel', type=int, default=500, help='k for kernel estimators')
    parser.add_argument('--lasso-alpha', '--lasso_alpha', dest='lasso_alpha', type=float, default=0.01, help='Alpha for Lasso regression')
    parser.add_argument('--scheduler-type', type=str, default=None, choices=['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--scheduler-step-size', type=int, default=30)
    parser.add_argument('--scheduler-gamma', type=float, default=0.1)
    parser.add_argument('--scheduler-milestones', type=int, nargs='+', default=[50, 80])
    parser.add_argument('--scheduler-t-max', type=int, default=-1)
    parser.add_argument('--scheduler-min-lr', type=float, default=1e-6)
    parser.add_argument('--scheduler-patience', type=int, default=10)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--uci-populations', '--uci_populations', dest='uci_populations', type=str, nargs='+', default=None,
                   help='Filter UCI data to specific populations (Male, Female, Young, etc.)')
    parser.add_argument('--exclude-population-features', '--exclude_population_features', dest='exclude_population_features', action='store_true',
                   help='Exclude features used to define populations (sex, race, age_group)')
    return parser.parse_args()


def restore_saved_args_for_resume(args):
    """Populate CLI defaults from a run while preserving its resume target."""
    resume_path = args.resume
    params_candidates = (
        os.path.join(resume_path, 'experiment_params.json'),
        os.path.join(resume_path, 'params_v10.json'),
    )
    for params_path in params_candidates:
        if os.path.exists(params_path):
            with open(params_path, 'r') as params_file:
                saved_params = json.load(params_file)
            for key, value in saved_params.items():
                if key != 'resume' and hasattr(args, key):
                    setattr(args, key, value)
            break
    args.resume = resume_path
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.resume and not args.populations:
        raise SystemExit("--populations is required when starting a new experiment")

    if args.resume and not args.populations:
        args = restore_saved_args_for_resume(args)
        if not args.populations:
            raise SystemExit(
                "Could not recover --populations from the resumed run; pass it explicitly"
            )
    
    # print the uci_populations and populations argument
    if args.uci_populations:
        print(f"UCI populations: {args.uci_populations}")
        print(f"Populations: {args.populations}")
    else:
        print(f"Populations: {args.populations}")
    # Check if resume path is provided
    if args.resume:
        run_pipeline(args, resume_path=args.resume)
    else:
        run_pipeline(args)
