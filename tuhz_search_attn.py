import os
import json
import argparse
import lightning as L
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from optuna.integration.wandb import WeightsAndBiasesCallback
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.loggers import WandbLogger
from torch_geometric.loader import DataLoader

# Import your model and dataset
from model import LightGTMamba  # Make sure this import works

def load_data(data_dir, dataset_has_fft):
    print("Loading FFT data from", data_dir)
    train_dataset = torch.load(os.path.join(data_dir, 'train_dataset_fft.pt'))
    val_dataset = torch.load(os.path.join(data_dir, 'test_dataset_fft.pt'))

    return train_dataset, val_dataset

def objective(trial, args, train_dataloader, val_dataloader):
    # Suggest hyperparameters
    trial_params = {
        'num_tgmamba_layers': trial.suggest_int('num_tgmamba_layers', 1, 8),
        'model_dim': trial.suggest_categorical('model_dim', [16, 32, 50]),
        'state_expansion_factor': trial.suggest_categorical('state_expansion_factor', [16, 32, 48, 64, 128]),
        'conv_type': trial.suggest_categorical('conv_type', ['gcnconv', 'graphconv', 'chebconv', 'gatv2conv']),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
        'lr_init': trial.suggest_float('lr_init', 1e-5, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.5, log=True),
        'edge_learner_attention': trial.suggest_categorical('edge_learner_attention', [True, False]),
        'attn_threshold': trial.suggest_float('attn_threshold', 0.03, 0.25),
        'attn_softmax_temp': trial.suggest_float('attn_softmax_temp', 0.001, 1.0, log=True),
        'seq_pool_type': trial.suggest_categorical('seq_pool_type', ['last', 'mean', 'max']),
        'vertex_pool_type': trial.suggest_categorical('vertex_pool_type', ['mean', 'max']),
        'edge_learner_time_varying': trial.suggest_categorical('edge_learner_time_varying', [True, False]),
    }
    
    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="tgmamba_optuna", name=f"trial_{trial.number}", config=vars(args))
    first_batch = next(iter(train_dataloader))
    # Create model
    model = LightGTMamba(
        num_vertices=19,
        conv_type=trial_params['conv_type'],
        seq_pool_type=trial_params['seq_pool_type'],
        vertex_pool_type=trial_params['vertex_pool_type'],
        input_dim=first_batch.x.shape[-1],
        d_model=trial_params['model_dim'],
        d_state=trial_params['state_expansion_factor'],
        d_conv=4,
        num_tgmamba_layers=trial_params['num_tgmamba_layers'],
        optimizer_name=trial_params['optimizer_name'],
        lr=trial_params['lr_init'],
        weight_decay=trial_params['weight_decay'],
        rmsnorm=True,
        edge_learner_attention=True,
        edge_learner_layers=1,
        attn_softmax_temp=trial_params['attn_softmax_temp'],
        attn_threshold=trial_params['attn_threshold'],
        edge_learner_time_varying=trial_params['edge_learner_time_varying'],
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
        max_epochs=20,
        accelerator="gpu",
        devices=args.gpu_id,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
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

def main(args):
    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Load data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_dataset')
    train_dataset, val_dataset = load_data(data_dir, args.dataset_has_fft)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)

    # Create and run Optuna study
    wandb_kwargs = {"project": "tgmamba-optuna-wandb-example"}
    wandbc = WeightsAndBiasesCallback(metric_name="accuracy", wandb_kwargs=wandb_kwargs)

    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    study.optimize(lambda trial: objective(trial, args, train_dataloader, val_dataloader), 
                   n_trials=args.n_trials, 
                   n_jobs=len(args.gpu_id),
                   callbacks=[wandbc])

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Save the study for later analysis
    optuna.save_study(study, os.path.join(args.save_dir, "tgmamba_optuna_study.pkl"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGMamba Hyperparameter Search")
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='optuna_results/tuhz')
    parser.add_argument('--dataset_has_fft', action='store_true', help="Enable FFT-processed data in the model")
    parser.add_argument('--num_vertices', type=int, default=19)
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=100, help="Number of trials for Optuna")
    
    args = parser.parse_args()
    main(args)