import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch
import os
import json
import torch.optim as optim
from torch_geometric.loader import DataLoader
from model import LightGTMamba
import wandb
import argparse

def main(args):
    wandb.init(project="tgmamba", config=vars(args))

    # Set random seed
    L.seed_everything(args.rand_seed, workers=True)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)
    
    print("Loading datasets...")
    if args.dataset_has_fft:
        train_dataset = torch.load('data/processed_dataset/train_dataset_fft.pt')
        val_dataset = torch.load('data/processed_dataset/val_dataset_fft.pt')
        test_dataset = torch.load('data/processed_dataset/test_dataset_fft.pt')
    else:
        train_dataset = torch.load('data/processed_dataset/train_dataset_raw.pt')
        val_dataset = torch.load('data/processed_dataset/val_dataset_raw.pt')
        test_dataset = torch.load('data/processed_dataset/test_dataset_raw.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers)

    model = LightGTMamba(num_vertices=args.num_vertices, 
                         conv_type=args.conv_type.lower(), 
                         seq_pool_type=args.seq_pool_type, 
                         vertex_pool_type=args.vertex_pool_type,
                         input_dim=100 if args.dataset_has_fft else 200, 
                         d_model=args.model_dim, 
                         d_state=args.state_expansion_factor, 
                         d_conv=args.local_conv_width, 
                         lr=args.lr_init)

    optimizer = optim.Adam(params=model.parameters(), lr=args.lr_init)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)

    # Callbacks
    checkpoint_filename = f"state-{args.state_expansion_factor}_seq-{args.seq_pool_type}_vp-{args.vertex_pool_type}_fft-{str(args.dataset_has_fft)}_{{epoch:02d}}.ckpt"
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=args.save_dir,
        filename=checkpoint_filename,
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss", mode="min", patience=args.patience
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=args.num_epochs,
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        devices=args.gpu_id,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=True,
        strategy=DDPStrategy(find_unused_parameters=False),
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
    print("Training complete.")

    print("Running test...")
    trainer.test(
        model=model,
        dataloaders=test_dataloader,
    )

    # # Save ROC curve data
    # from scipy.io import savemat
    # mdic = {"roc_fpr": fpr, "roc_tpr": tpr}
    # savemat("TGMamba/results/test_roc.mat", mdic)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TGMamba Training Script")
    parser.add_argument('--rand_seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='TGMamba/results')
    parser.add_argument('--dataset_has_fft', type=bool, default=True)
    parser.add_argument('--conv_type', type=str, default='gcnconv')
    parser.add_argument('--seq_pool_type', type=str, default='last')
    parser.add_argument('--vertex_pool_type', type=str, default='mean')
    parser.add_argument('--model_dim', type=int, default=32)
    parser.add_argument('--state_expansion_factor', type=int, default=16)
    parser.add_argument('--local_conv_width', type=int, default=4)
    parser.add_argument('--num_vertices', type=int, default=19)
    parser.add_argument('--train_batch_size', type=int, default=1024)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--lr_init', type=float, default=5e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu_id', nargs='+', type=int, default=[7])
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    
    args = parser.parse_args()
    main(args)