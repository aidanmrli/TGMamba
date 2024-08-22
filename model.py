import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
import wandb
from torchmetrics import Accuracy, F1Score, AUROC

class LightGTMamba(L.LightningModule):
    def __init__(self, 
                 num_vertices=19, 
                 conv_type='gconv', 
                 seq_pool_type='last', 
                 vertex_pool_type='mean',
                 input_dim=100, 
                 d_model=32, 
                 d_state=16, 
                 d_conv=4,
                 num_tgmamba_layers=2, 
                 lr=5e-4,
                 rmsnorm=True):
        super().__init__()
        self.num_vertices = num_vertices
        self.seq_pool_type = seq_pool_type
        self.vertex_pool_type = vertex_pool_type
        self.lr = lr
        self.rmsnorm = rmsnorm

        # if FFT, input_dim = 100. Else, input_dim = 200.
        self.in_proj = torch.nn.Linear(input_dim, d_model)     # from arshia's code
        self.blocks = torch.nn.ModuleList([
            TGMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
                num_vertices=self.num_vertices,
                conv_type=conv_type,
                rmsnorm=self.rmsnorm,
            ) for _ in range(num_tgmamba_layers)
        ])
        self.fc = torch.nn.Linear(d_model, 1)
        torch.set_float32_matmul_precision('high') 

        # Initialize metrics
        self.train_accuracy = Accuracy(task="binary")
        self.train_f1 = F1Score(task="binary")
        self.train_auroc = AUROC(task="binary")

        self.val_accuracy = Accuracy(task="binary")
        self.val_f1 = F1Score(task="binary")
        self.val_auroc = AUROC(task="binary")

        self.test_accuracy = Accuracy(task="binary")
        self.test_f1 = F1Score(task="binary")
        self.test_auroc = AUROC(task="binary")

    def forward(self, data: Data):
        """
        input data: torch_geometric.data.Data object

        Output: torch.Tensor of shape (B, 1) representing predictions
        """
        # Normalize input data
        clip = (data.x.float() - data.x.float().mean(2, keepdim=True)) / (data.x.float().std(2, keepdim=True) + 1e-10)
        batch, seqlen, _ = data.x.size()
        num_vertices = self.num_vertices
        batch = batch // num_vertices

        out = self.in_proj(clip)  # (B*V, L, d_model)
        for block in self.blocks:
            out = block(out, data.edge_index, data.edge_weight)  # (B*V, L, d_model)
        
        out = out.view(
            batch, num_vertices, seqlen, -1
        )  # (B, V, L, d_model)
        
        # pooling over the sequence length. consider mean vs max pooling
        if self.seq_pool_type == 'last':
            out = out[:, :, -1, :]  # (B, V, d_model)
        elif self.seq_pool_type == 'mean':
            out = out.mean(dim=2)
        elif self.seq_pool_type == 'max':
            out, _ = out.max(dim=2)
        else:
            raise ValueError("Invalid sequence pooling type")

        # pool over the vertices. consider mean vs max pooling
        if self.vertex_pool_type == 'mean':
            out = out.mean(dim=1)  # (B, d_model)
        elif self.vertex_pool_type == 'max':
            out, _ = out.max(dim=1)
        else:
            raise ValueError("Invalid vertex pooling type")

        out = self.fc(out)  # (B, 1)
        return out

    def training_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"
        
        # Compute metrics
        preds = torch.sigmoid(out)
        assert preds.size(0) == data.y.size(0), "Batch size mismatch"
        targets = data.y.type(torch.float32).reshape(-1, 1)
        loss = F.binary_cross_entropy_with_logits(out, targets)
        self.train_accuracy(preds, targets)
        self.train_f1(preds, targets)
        self.train_auroc(preds, targets)

        # Log metrics
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
        self.log("train/accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        wandb.log({
            "train/loss": loss,
            "train/accuracy": self.train_accuracy.compute(),
            "train/f1": self.train_f1.compute(),
            "train/auroc": self.train_auroc.compute()
        })

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

        # Compute metrics
        preds = torch.sigmoid(out)
        assert preds.size(0) == data.y.size(0), "Batch size mismatch"
        targets = data.y.type(torch.float32).reshape(-1, 1)
        loss = F.binary_cross_entropy_with_logits(out, targets)
        self.val_accuracy(preds, targets)
        self.val_f1(preds, targets)
        self.val_auroc(preds, targets)

        # Log metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
        self.log("val/accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        wandb.log({
            "val/loss": loss,
            "val/accuracy": self.val_accuracy.compute(),
            "val/f1": self.val_f1.compute(),
            "val/auroc": self.val_auroc.compute()
        })
        return loss

    def test_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"
        
        preds = torch.sigmoid(out)
        targets = data.y.type(torch.float32).reshape(-1, 1)
        loss = F.binary_cross_entropy_with_logits(out, targets)
        self.test_accuracy(preds, targets)
        self.test_f1(preds, targets)
        self.test_auroc(preds, targets)

        # Log metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
        self.log("test/accuracy", self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("test/auroc", self.test_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        wandb.log({
            "test/loss": loss,
            "test/accuracy": self.test_accuracy.compute(),
            "test/f1": self.test_f1.compute(),
            "test/auroc": self.test_auroc.compute()
        })        
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]