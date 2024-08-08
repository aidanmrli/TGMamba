import lightning as L
import torch
from torch_geometric.loader import DataLoader
from sklearn import metrics

from mamba_ssm import TGMamba
import os
import json
from model import LightGTMamba
import wandb


def main():
    # Load the hardcoded arguments from the training run
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
        'patience': 3,
        'gpu_id': [6],
        'accumulate_grad_batches': 1,
    }
    wandb.init(project="tgmamba", config=args)

    # Set random seed
    L.seed_everything(args['rand_seed'], workers=True)

    print("Loading test dataset...")
    test_dataset = torch.load('TGMamba/data/processed_dataset/test_dataset.pt')
    test_dataloader = DataLoader(test_dataset, batch_size=args['test_batch_size'], num_workers=args['num_workers'])

    print("Loading trained model...")
    model = LightGTMamba.load_from_checkpoint("TGMamba/results/last.ckpt", args=args, fire_rate=0.5, conv_type='gconv')
    # model.load_state_dict(torch.load('TGMamba/results/checkpoint_abcd.pt'))
    # model.to("cuda")
    model.to("cuda")

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

    # Save ROC curve data
    from scipy.io import savemat
    mdic = {"roc_fpr": fpr, "roc_tpr": tpr}
    savemat("TGMamba/results/test_roc.mat", mdic)

    print("Running test...")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=args['gpu_id'],
        enable_progress_bar=True,
    )

    trainer.test(
        model=model,
        dataloaders=test_dataloader,
    )

if __name__ == "__main__":
    main()