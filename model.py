import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
import wandb
from torchmetrics import Accuracy, F1Score, AUROC

class LightGTMamba(L.LightningModule):
    def __init__(self, args, fire_rate, conv_type):
        super().__init__()
        self.args = args

        # NOTE: these two are not used.
        self.fire_rate = fire_rate  # probability determining how often neurons are updated.
        self.conv_type = conv_type  # gconv or chebconv

        self.in_proj = torch.nn.Linear(100, 32)     # from arshia's code
        self.block = TGMamba(
            d_model=32,
            d_state=16,
            d_conv=4,
            expand=2,
            num_vertices=19
        ).to("cuda")
        self.fc = torch.nn.Linear(32, 1)
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
        num_vertices = self.args['num_vertices']
        batch = batch // num_vertices
        data.x = self.in_proj(clip)  # (B*V, L, d_model)

        out = self.block(data) # (B*V, L, d_model)
        out = out.view(
                batch, num_vertices, seqlen, -1
            )  
        
        # pooling over the sequence length. consider mean vs max pooling
        # out = out.mean(dim=2)
        out, _ = out.max(dim=2)
        # out = out[:, :, -1, :]  # (B, V, d_model)

        # pool over the vertices. consider mean vs max pooling
        # out = out.mean(dim=1)  # (B, d_model)
        out, _ = out.max(dim=1)  # (B, d_model)

        out = self.fc(out)  # (B, 1)        # apply fully connected linear layer from 32 features to 1 output
        return out

    def training_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"
        
        # Compute metrics
        preds = torch.sigmoid(out)
        targets = data.y.type(torch.float32).reshape(-1, 1)
        loss = F.mse_loss(preds, targets)
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
        targets = data.y.type(torch.float32).reshape(-1, 1)

        loss = F.mse_loss(preds, targets)
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

        loss = F.mse_loss(preds, targets)
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
        optimizer = optim.Adam(params=self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]