import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.strategies import DDPStrategy
import torch
import os
import json
import torch.nn.functional as F
from torchmetrics import AUROC

import torch.optim as optim
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from model import LightGTMamba
import wandb

# class LightGTMamba(L.LightningModule):
#     def __init__(self, args, fire_rate, conv_type):
#         super().__init__()
#         self.args = args
#         self.fire_rate = fire_rate  # probability determining how often neurons are updated.
#         self.conv_type = conv_type  # gconv or chebconv

#         self.in_proj = torch.nn.Linear(100, 32)     # from arshia's code
#         self.block = TGMamba(
#             d_model=32,
#             d_state=16,
#             d_conv=4,
#             expand=2,
#             num_vertices=19
#         ).to("cuda")
#         self.fc = torch.nn.Linear(32, 1)
#         torch.set_float32_matmul_precision('high') 

#     def forward(self, data):
#         # Normalize input data
#         clip = (data.x.float() - data.x.float().mean(2, keepdim=True)) / (data.x.float().std(2, keepdim=True) + 1e-10)
#         batch, seqlen, _ = data.x.size()
#         num_vertices = self.args['num_vertices']
#         batch = batch // num_vertices
#         data.x = self.in_proj(clip)  # (B*V, L, d_model)

#         out = self.block(data) # (B*V, L, d_model)
#         out = out.view(
#                 batch, num_vertices, seqlen, -1
#             )  
        
#         # TODO: pooling over the sequence length?
#         out = out.mean(dim=2)
#         # out = out[:, :, -1, :]  # (B, V, d_model)

#         # TODO: pool over the vertices?
#         out = out.mean(dim=1)  # (B, d_model)

#         out = self.fc(out)  # (B, 1)        # apply fully connected linear layer from 32 features to 1 output
#         return out

#     def training_step(self, data, batch_idx):
#         out = self(data)
#         assert out.size(0) == data.y.size(0), "Batch size mismatch"
#         assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

#         loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
#         self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
#         wandb.log({"train/loss": loss})
#         return loss

#     def validation_step(self, data, batch_idx):
#         out = self(data)
#         assert out.size(0) == data.y.size(0), "Batch size mismatch"
#         assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

#         loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
#         self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
#         wandb.log({"val/loss": loss})
#         return loss
    
#     def test_step(self, data, batch_idx):
#         out = self(data)
#         assert out.size(0) == data.y.size(0), "Batch size mismatch"
#         assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

#         loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
#         self.log("test/loss", loss, 
#                  on_step=False, 
#                  on_epoch=True, 
#                  prog_bar=True, 
#                  logger=True, 
#                  batch_size=data.num_graphs,
#                  sync_dist=True)
#         wandb.log({"test/loss": loss})
#         return loss
    
#     def configure_optimizers(self):
#         optimizer = optim.Adam(params=self.parameters(), lr=5e-4)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)
#         return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
    

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
        'patience': 3,
        'gpu_id': [5],
        'accumulate_grad_batches': 1,
    }
    wandb.init(project="tgmamba", config=args)

    # Set random seed
    L.seed_everything(args['rand_seed'], workers=True)
    os.makedirs(args['save_dir'], exist_ok=True)
    with open(os.path.join(args['save_dir'], "args.json"), "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)
    
    print("Building dataset...")
    
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
    # torch.save(model.state_dict(), 'TGMamba/results/checkpoint_abcd.pt' )
    # print("Model checkpoint saved.")
    # # Test
    # trainer.test(
    #     model=model,
    #     ckpt_path="best",
    #     test_dataloaders=test_dataloader,
    # )

if __name__ == "__main__":
    main()