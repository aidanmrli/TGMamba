import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch
import os
import json
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import LightGTMamba
import argparse
from pytorch_lightning.loggers import WandbLogger
from data import DODHDataModule, TUHZDataModule

DODH_RAW_DATA_DIR='/home/amli/dreem-learning-open/data/h5/dodh'


def main(args):
    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'processed_dataset')
    
    # TODO: instead of loading the train dataset that way, use this datamodule
    project_name_suffix = ""
    stopping_callback_metric = ""
    if args.dataset == 'dodh':
        datamodule = DODHDataModule(
                raw_data_path=DODH_RAW_DATA_DIR, 
                dataset_name="dodh",  # should be 'dodh'
                freq=250,    # args.sampling_freq = 250 Hz or if we make it lower then we can make the seq shorter
                train_batch_size=args.train_batch_size,
                test_batch_size=args.test_batch_size,
                num_workers=args.num_workers,
                standardize=True,
                balanced_sampling=True,  # True
                pin_memory=True,
            )
        stopping_callback_metric = "val/macro_f1"
        project_name_suffix = "_dodh"
    elif args.dataset == 'tuhz':
        datamodule = TUHZDataModule(
            data_dir=data_dir,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            dataset_has_fft=args.dataset_has_fft,
        )
        stopping_callback_metric = "val/auroc"
    # Prepare the data
    datamodule.prepare_data()
    datamodule.setup()
    print("Model parameters: ", args.state_expansion_factor, args.seq_pool_type, args.vertex_pool_type, args.conv_type, args.model_dim, args.local_conv_width, args.num_tgmamba_layers)
    
    model = LightGTMamba(dataset=args.dataset, 
                         conv_type=args.conv_type.lower(), 
                         seq_pool_type=args.seq_pool_type, 
                         vertex_pool_type=args.vertex_pool_type,
                         input_dim=datamodule.train_dataset[0].x.shape[-1], 
                         d_model=args.model_dim, 
                         d_state=args.state_expansion_factor, 
                         d_conv=args.local_conv_width,
                         num_tgmamba_layers=args.num_tgmamba_layers, 
                         optimizer_name=args.optimizer_name,
                         lr=args.lr_init,
                         weight_decay=args.weight_decay,
                         rmsnorm=args.rmsnorm,
                         edge_learner_attention=True,  # args.edge_learner_attention,
                         attn_threshold=args.attn_threshold,
                         edge_learner_time_varying=False,) # args.edge_learner_time_varying,)

    # Callbacks
    checkpoint_filename = f"depth-{args.num_tgmamba_layers}-state-{args.state_expansion_factor}_conv-{args.conv_type}_seq-{args.seq_pool_type}_vp-{args.vertex_pool_type}_fft-{str(args.dataset_has_fft)}_{{epoch:02d}}"
    checkpoint_callback = ModelCheckpoint(
        monitor=stopping_callback_metric,
        mode="max",
        dirpath=args.save_dir,
        filename=checkpoint_filename,
        save_last=True,
        save_top_k=1,  # Save top 3 models
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor=stopping_callback_metric, mode="max", patience=args.patience
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    project_name = "tgmamba" + project_name_suffix
    wandb_logger = WandbLogger(project=project_name, log_model="all", config=vars(args))

    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        logger=wandb_logger,
        devices=args.gpu_id,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model, datamodule=datamodule)
    print("Training complete.")

    print("Testing best model...")
    best_results = trainer.test(
        datamodule=datamodule,
        ckpt_path="best",
    )

    # Print or process the results as needed
    print("Best model results:", best_results)

    # # Save ROC curve data
    # from scipy.io import savemat
    # mdic = {"roc_fpr": fpr, "roc_tpr": tpr}
    # savemat("TGMamba/results/test_roc.mat", mdic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGMamba Training Script")
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='TGMamba/results')
    parser.add_argument('--dataset', type=str, default='tuhz')
    parser.add_argument('--dataset_has_fft', action='store_true', help="Enable FFT-processed data in the model")
    parser.add_argument('--conv_type', type=str, default='graphconv')
    parser.add_argument('--seq_pool_type', type=str, default='mean')
    parser.add_argument('--vertex_pool_type', type=str, default='max')
    parser.add_argument('--model_dim', type=int, default=1)
    parser.add_argument('--state_expansion_factor', type=int, default=32)
    parser.add_argument('--local_conv_width', type=int, default=4)
    parser.add_argument('--num_tgmamba_layers', type=int, default=2)
    parser.add_argument('--num_vertices', type=int, default=19)
    parser.add_argument('--rmsnorm', action='store_true', help="Enable RMSNorm in the model")
    parser.add_argument('--edge_learner_layers', type=int, default=1)
    parser.add_argument('--edge_learner_attention', action='store_true', help="Enable attention in the edge learner")
    parser.add_argument('--attn_threshold', type=float, default=0.2)
    parser.add_argument('--attn_softmax_temp', type=float, default=0.01)
    parser.add_argument('--edge_learner_time_varying', action='store_true', help="Enable time-varying edge weights in the edge learner")
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr_init', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--optimizer_name', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[0])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    args = parser.parse_args()
    main(args)