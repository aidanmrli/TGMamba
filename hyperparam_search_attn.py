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
from data import TUHZDataModule, DODHDataModule, BCIchaDataModule, BCIchaDataset
import joblib
import wandb
import logging
import sys

# Import your model and dataset
from model import LightGTMamba  # Make sure this import works
DODH_PROCESSED_DATA_DIR='/h/liaidan/TGMamba/data/'
TUHZ_PROCESSED_DATA_DIR='/h/liaidan/TGMamba/data/tuhz/new/'
BCICHA_DATA_DIR='/h/liaidan/TGMamba/data/BCIcha/'

def load_data(args):
    if args.dataset == 'tuhz':
        datamodule = TUHZDataModule(
            data_dir=TUHZ_PROCESSED_DATA_DIR,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            dataset_has_fft=True,
        )
        input_dim = 100
        stopping_metric = "val/auroc"
    elif args.dataset == 'dodh':
        datamodule = DODHDataModule(
            preprocessed_data_dir=DODH_PROCESSED_DATA_DIR,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            balanced_sampling=True,
            use_class_weight=False,
            pin_memory=True,
        )
        input_dim = 129
        stopping_metric = "val/macro_f1"
    elif args.dataset == 'bcicha':
        SUBJECT_LIST = {2, 6, 7, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 24, 26}
        assert args.subject in SUBJECT_LIST, f"Invalid subject number for BCI Competition IV Dataset 2a. Must be one of {SUBJECT_LIST}"
        datamodule = BCIchaDataModule(
            data_dir=BCICHA_DATA_DIR,
            subject=args.subject,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            dataset_has_fft=True,
        )
        stopping_metric = "val/auroc"
        input_dim = 11
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    datamodule.prepare_data()
    datamodule.setup()
    return datamodule, input_dim, stopping_metric

def objective(trial, args, datamodule, input_dim, stopping_metric):
    # Suggest hyperparameters
    trial_params = {
        'num_tgmamba_layers': trial.suggest_int('num_tgmamba_layers', 2, 3),
        'model_dim': trial.suggest_categorical('model_dim', [32, 50, 100]),
        'state_expansion_factor': trial.suggest_categorical('state_expansion_factor', [16, 32, 48, 64]),
        'conv_type': 'graphconv', # trial.suggest_categorical('conv_type', ['gcnconv', 'graphconv', 'chebconv', 'gatv2conv']),
        'optimizer_name': 'adamw', # trial.suggest_categorical('optimizer_name', ['adam', 'adamw']),
        'lr_init': trial.suggest_float('lr_init', 1e-7, 2e-3, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 0.01, 0.5, log=True),
        'dropout': trial.suggest_float('dropout', 0.1, 0.5),
        'edge_learner_attention': trial.suggest_categorical('edge_learner_attention', [True, False]),
        'attn_threshold': trial.suggest_float('attn_threshold', 0.03, 0.3),
        'attn_softmax_temp': trial.suggest_float('attn_softmax_temp', 0.001, 1.0, log=True),
        'seq_pool_type': trial.suggest_categorical('seq_pool_type', ['last', 'mean', 'max']),
        'vertex_pool_type': trial.suggest_categorical('vertex_pool_type', ['mean', 'max']),
        'edge_learner_time_varying': True, # trial.suggest_categorical('edge_learner_time_varying', [True, False]),
        'edge_learner_layers': 1, # trial.suggest_int('edge_learner_layers', 1, 3),
        'train_batch_size': args.train_batch_size,
        'test_batch_size': args.test_batch_size,
    }
    
    # Initialize WandbLogger
    with wandb.init(project=f"{args.dataset}-moredata-hyperparameter-search", name=f"vec_dropout3_trial_{trial.number}", config=trial_params, tags=[f"dropout"], reinit=True) as run:
        if args.dataset == 'bcicha':
            wandb_logger = WandbLogger(experiment=run, tags=[f"subject_{args.subject}"])    
        else:
            wandb_logger = WandbLogger(experiment=run)
        try:
            # Create model
            model = LightGTMamba(
                dataset=args.dataset,
                conv_type=trial_params['conv_type'],
                seq_pool_type=trial_params['seq_pool_type'],
                vertex_pool_type=trial_params['vertex_pool_type'],
                input_dim=input_dim,
                d_model=trial_params['model_dim'],
                d_state=trial_params['state_expansion_factor'],
                d_conv=4,
                num_tgmamba_layers=trial_params['num_tgmamba_layers'],
                optimizer_name='adamw', # haven't tried this yet,
                lr=trial_params['lr_init'],
                weight_decay=trial_params['weight_decay'],
                dropout=trial_params['dropout'],
                rmsnorm=True,
                edge_learner_attention=trial_params['edge_learner_attention'],
                edge_learner_layers=trial_params['edge_learner_layers'],
                edge_learner_time_varying=trial_params['edge_learner_time_varying'],
                attn_time_varying=False,
                attn_softmax_temp=trial_params['attn_softmax_temp'],
                attn_threshold=trial_params['attn_threshold'],
            )

            # Early stopping callback
            early_stop_callback = EarlyStopping(
                monitor=stopping_metric,
                min_delta=0.00,
                patience=10,
                verbose=False,
                mode="max"
            )

            # Optuna pruning callback
            pruning_callback = PyTorchLightningPruningCallback(trial, monitor=stopping_metric)

            # Trainer
            trainer = L.Trainer(
                logger=wandb_logger,
                callbacks=[early_stop_callback, pruning_callback],
                max_epochs=50,
                accelerator="gpu",
                devices=1,  # Use GPUs 3 and 4 (index 2 and 3)
                # strategy="ddp",  # Enable DDP for multi-GPU training
                accumulate_grad_batches=args.accumulate_grad_batches,
                enable_progress_bar=True,
                enable_checkpointing=False,
            )

            # Train the model
            trainer.logger.log_hyperparams(trial_params)
            trainer.fit(model, datamodule=datamodule)
            pruning_callback.check_pruned()

            best_val_score = trainer.callback_metrics[stopping_metric].item()

            # Log the best score to Optuna
            trial.set_user_attr(f"best_{stopping_metric}", best_val_score)
            print(f"Trial {trial.number} finished with best {stopping_metric}: {best_val_score}")

            return best_val_score
        except Exception as e:
            print(f"Trial {trial.number} encountered an error: {str(e)}")
            
            # Try to get the best score achieved before the error
            try:
                best_val_score = trainer.callback_metrics[stopping_metric].item()
                print(f"Best {stopping_metric} before error: {best_val_score}")
                return best_val_score
            except:
                print("Could not retrieve a valid score. Returning -inf.")
                return float('-inf') 

def main(args):
    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    # Load data
    datamodule, input_dim, stopping_metric = load_data(args)
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    study = joblib.load(f"{args.dataset}_study.pkl") if os.path.exists(f"{args.dataset}_study.pkl") else optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    # study = optuna.create_study(direction='maximize', pruner=optuna.pruners.HyperbandPruner())
    joblib.dump(study, "tuhz_study_attn.pkl")
    
    study.optimize(lambda trial: objective(trial, args, datamodule, input_dim, stopping_metric), 
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
    joblib.dump(study, f"{args.dataset}_study.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGMamba Hyperparameter Search")
    parser.add_argument('--dataset', type=str, choices=['tuhz', 'dodh', 'bcicha'], required=True, help="Dataset to use for hyperparameter search")
    parser.add_argument('--subject', type=int, default=2)
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='optuna_results/')
    parser.add_argument('--train_batch_size', type=int, default=40)
    parser.add_argument('--test_batch_size', type=int, default=40)
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0], help="GPU IDs to use for training")
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--n_trials', type=int, default=300, help="Number of trials for Optuna")
    
    args = parser.parse_args()

    main(args)