import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch
import os
import json
import torch.nn.functional as F
from sklearn import metrics

import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from model import LightGTMamba
import wandb
    

def main():

    # Hardcoded arguments
    args = {
        'rand_seed': 42,
        'save_dir': 'TGMamba/results',
        # 'dataset': 'tuh',
        'max_seq_len': 1000,
        'num_vertices': 19,
        'train_batch_size': 1024,
        'val_batch_size': 256,
        'test_batch_size': 256,
        'num_workers': 4,
        'balanced_sampling': True,
        'lr_init': 1e-3,
        'l2_wd': 1e-5,
        'optimizer': 'adam',
        'scheduler': 'cosine',
        'num_epochs': 100,
        'patience': 5,
        'gpu_id': [7],
        'accumulate_grad_batches': 1,
    }
    wandb.init(project="tgmamba", config=args)

    # Set random seed
    L.seed_everything(args['rand_seed'], workers=True)
    os.makedirs(args['save_dir'], exist_ok=True)
    with open(os.path.join(args['save_dir'], "args.json"), "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)
    
    print("Loading datasets...")
    
    train_dataset = torch.load('TGMamba/data/processed_dataset/train_dataset.pt')
    val_dataset = torch.load('TGMamba/data/processed_dataset/val_dataset.pt')
    test_dataset = torch.load('TGMamba/data/processed_dataset/test_dataset.pt')

    train_dataloader = DataLoader(train_dataset, batch_size=args['train_batch_size'], num_workers=255)
    val_dataloader = DataLoader(val_dataset, batch_size=args['test_batch_size'], num_workers=255)
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], num_workers=255)

    model = LightGTMamba(args, fire_rate=0.5, conv_type='gconv')

    optimizer = optim.Adam(params=model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=args['save_dir'],
        save_last=True,
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    early_stopping_callback = EarlyStopping(
        monitor="val/loss", mode="min", patience=args['patience']
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        max_epochs=args['num_epochs'],
        callbacks=[checkpoint_callback, early_stopping_callback, lr_monitor],
        devices=args['gpu_id'],
        accumulate_grad_batches=args['accumulate_grad_batches'],
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
    
    print("Evaluating model on test set...")
    model.eval()
    y_true = []
    y_pred = []
    for data in test_dataloader:
        data = data.to("cuda")
        output = model(data)
        output = torch.sigmoid(output).squeeze()
        y_true.extend(data.y.cpu().numpy())
        y_pred.extend((output > 0.5).cpu().numpy())

    accuracy = metrics.accuracy_score(y_true, y_pred)
    f1 = metrics.f1_score(y_true, y_pred)
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    auc = metrics.auc(fpr, tpr)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    wandb.log({
        "test/accuracy": accuracy,
        "test/f1_score": f1,
        "test/auc_roc": auc
    })
    # wandb.log({"test/roc_curve": wandb.plot.roc_curve(y_true, y_pred)})


    # Save ROC curve data
    from scipy.io import savemat
    mdic = {"roc_fpr": fpr, "roc_tpr": tpr}
    savemat("TGMamba/results/test_roc.mat", mdic)


if __name__ == "__main__":
    main()