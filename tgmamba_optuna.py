import os
import json
import argparse
import lightning as L
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
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

def objective(trial, args, train_dataset, val_dataset):
    # Suggest hyperparameters
    params = {
        'num_tgmamba_layers': trial.suggest_int('num_tgmamba_layers', 1, 4),
        'model_dim': trial.suggest_categorical('model_dim', [16, 32, 50]),
        'state_expansion_factor': trial.suggest_categorical('state_expansion_factor', [16, 32, 64, 128]),
        'lr_init': trial.suggest_float('lr_init', 1e-4, 5e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.5, log=True),
        'edge_learner_attention': trial.suggest_categorical('edge_learner_attention', [True, False]),
        'attn_threshold': trial.suggest_float('attn_threshold', 0.05, 0.2),
        'attn_softmax_temp': trial.suggest_float('attn_softmax_temp', 0.001, 1.0, log=True),
        'local_conv_width': trial.suggest_int('local_conv_width', 2, 4),
        'seq_pool_type': trial.suggest_categorical('seq_pool_type', ['last', 'mean', 'max']),
        'vertex_pool_type': trial.suggest_categorical('vertex_pool_type', ['mean', 'max']),
        'rmsnorm': trial.suggest_categorical('rmsnorm', [True, False]),
        'edge_learner_time_varying': trial.suggest_categorical('edge_learner_time_varying', [True, False]),
        'edge_learner_layers': trial.suggest_int('edge_learner_layers', 1, 3),
        'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
        'conv_type': trial.suggest_categorical('conv_type', ['gcnconv', 'graphconv', 'gatv2conv', 'chebconv']),
    }

    # Update args with suggested parameters
    for key, value in params.items():
        setattr(args, key, value)

    # Initialize WandbLogger
    wandb_logger = WandbLogger(project="tgmamba_optuna", name=f"trial_{trial.number}", config=vars(args))

    # Create model
    model = LightGTMamba(
        num_vertices=args.num_vertices,
        conv_type=args.conv_type.lower(),
        seq_pool_type=args.seq_pool_type,
        vertex_pool_type=args.vertex_pool_type,
        input_dim=train_dataset[0].x.shape[-1],
        d_model=args.model_dim,
        d_state=args.state_expansion_factor,
        d_conv=args.local_conv_width,
        num_tgmamba_layers=args.num_tgmamba_layers,
        optimizer_name=args.optimizer_name,
        lr=args.lr_init,
        weight_decay=args.weight_decay,
        rmsnorm=args.rmsnorm,
        edge_learner_attention=args.edge_learner_attention,
        edge_learner_layers=args.edge_learner_layers,
        attn_softmax_temp=args.attn_softmax_temp,
        attn_threshold=args.attn_threshold,
        edge_learner_time_varying=args.edge_learner_time_varying,
    )

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=args.num_workers)

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
    trainer.fit(model, train_dataloader, val_dataloader)

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

    # Create and run Optuna study
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(lambda trial: objective(trial, args, train_dataset, val_dataset), 
                   n_trials=args.n_trials, 
                   n_jobs=len(args.gpu_id))

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
    parser.add_argument('--save_dir', type=str, default='TGMamba/results')
    parser.add_argument('--dataset_has_fft', action='store_true', help="Enable FFT-processed data in the model")
    parser.add_argument('--conv_type', type=str, default='graphconv')
    parser.add_argument('--num_vertices', type=int, default=19)
    parser.add_argument('--test_batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=100, help="Number of trials for Optuna")
    
    args = parser.parse_args()
    main(args)