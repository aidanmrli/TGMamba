import os
import json
import argparse
import lightning as L
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from optuna.visualization import plot_param_importances
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader
import joblib
import wandb
import logging
import sys
# Import your model and dataset
from model import LightGTMamba  # Make sure this import works

def load_data(data_dir):
    print("Loading FFT data from", data_dir)
    train_dataset = torch.load(os.path.join(data_dir, 'train_dataset_fft.pt'))
    val_dataset = torch.load(os.path.join(data_dir, 'test_dataset_fft.pt'))

    return train_dataset, val_dataset

def objective(trial, args, train_dataloader, val_dataloader):
    # Suggest hyperparameters
    trial_params = {
        'num_tgmamba_layers': trial.suggest_int('num_tgmamba_layers', 1, 4),
        'model_dim': trial.suggest_categorical('model_dim', [16, 32, 50]),
        'state_expansion_factor': trial.suggest_categorical('state_expansion_factor', [16, 32, 48, 64, 128]),
        'conv_type': trial.suggest_categorical('conv_type', ['gcnconv', 'graphconv', 'chebconv', 'gatv2conv']),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
        'lr_init': trial.suggest_float('lr_init', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.5, log=True),
        'seq_pool_type': trial.suggest_categorical('seq_pool_type', ['last', 'mean', 'max']),
        'vertex_pool_type': trial.suggest_categorical('vertex_pool_type', ['mean', 'max']),
        'edge_learner_layers': trial.suggest_int('edge_learner_layers', 1, 3),
    }
    
    # Initialize WandbLogger
    with wandb.init(project="tuhz-no-edge-attention", name=f"trial_{trial.number}", config=trial_params, reinit=True) as run:
        # Initialize WandbLogger with the current run
        wandb_logger = WandbLogger(experiment=run)    
        # first_batch = next(iter(train_dataloader))
        try:
            # Create model
            model = LightGTMamba(
                num_vertices=19,
                conv_type=trial_params['conv_type'],
                seq_pool_type=trial_params['seq_pool_type'],
                vertex_pool_type=trial_params['vertex_pool_type'],
                input_dim=100,
                d_model=trial_params['model_dim'],
                d_state=trial_params['state_expansion_factor'],
                d_conv=4,
                num_tgmamba_layers=trial_params['num_tgmamba_layers'],
                optimizer_name=trial_params['optimizer_name'],
                lr=trial_params['lr_init'],
                weight_decay=trial_params['weight_decay'],
                rmsnorm=True,
                edge_learner_attention=False,
                edge_learner_layers=trial_params['edge_learner_layers'],
                attn_softmax_temp=1.0,
                attn_threshold=0.1,
                edge_learner_time_varying=True,
            )

            # Early stopping callback
            early_stop_callback = EarlyStopping(
                monitor="val/auroc",
                min_delta=0.00,
                patience=5,
                verbose=False,
                mode="max"
            )

            # Optuna pruning callback
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val/auroc")

            # Trainer
            trainer = L.Trainer(
                logger=wandb_logger,
                callbacks=[early_stop_callback, pruning_callback],
                max_epochs=30,
                accelerator="gpu",
                devices=1,  # Use GPUs 3 and 4 (index 2 and 3)
                # strategy="ddp",  # Enable DDP for multi-GPU training
                accumulate_grad_batches=args.accumulate_grad_batches,
                enable_progress_bar=True,
                enable_checkpointing=False,
            )

            # Train the model
            trainer.logger.log_hyperparams(trial_params)

            trainer.fit(model, train_dataloader, val_dataloader)
            pruning_callback.check_pruned()

            # Get the best validation AUROC
            best_val_auroc = trainer.callback_metrics["val/auroc"].item()

            # Log the best score to Optuna
            trial.set_user_attr("best_val_auroc", best_val_auroc)
            print(f"Trial {trial.number} finished with best validation AUROC: {best_val_auroc}")

            return best_val_auroc
        except Exception as e:
            print(f"Trial {trial.number} failed due to error: {str(e)}")
            # Return a very low score to indicate failure
            return float('-inf')

def main(args):
    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_dataset')
    train_dataset, val_dataset = load_data(data_dir)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)

    # Create and run Optuna study
    wandb_kwargs = {"project": "tuhz-no-edge-attention"}
    # wandbc = WeightsAndBiasesCallback(metric_name="val/auroc", wandb_kwargs=wandb_kwargs)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study = joblib.load("tuhz_study_no_attn.pkl")

    # study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner(), load_if_exists=True)
    joblib.dump(study, "tuhz_study_no_attn.pkl")
    study.optimize(lambda trial: objective(trial, args, train_dataloader, val_dataloader), 
                   n_trials=args.n_trials, 
                   n_jobs=1,
                   gc_after_trial=True
                #    callbacks=[wandbc]
                   )

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    # Generate and save hyperparameter importance plot
    param_importance_fig = plot_param_importances(study)
    param_importance_fig.write_image(os.path.join(args.save_dir, "hyperparameter_importance.png"))
    # Save the study for later analysis
    optuna.save_study(study, os.path.join(args.save_dir, "tuhz_study_no_attn.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TUHZ Hyperparameter Search (no attention)")
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='optuna_results/tuhz')
    parser.add_argument('--num_vertices', type=int, default=19)
    parser.add_argument('--train_batch_size', type=int, default=64)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0], help="GPU IDs to use for training")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=300, help="Number of trials for Optuna")
    
    args = parser.parse_args()

    main(args)