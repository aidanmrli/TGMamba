import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
import wandb

class LightGTMamba(L.LightningModule):
    def __init__(self, args, fire_rate, conv_type):
        super().__init__()
        self.args = args
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
        
        # TODO: pooling over the sequence length?
        out = out.mean(dim=2)
        # out = out[:, :, -1, :]  # (B, V, d_model)

        # TODO: pool over the vertices?
        out = out.mean(dim=1)  # (B, d_model)

        out = self.fc(out)  # (B, 1)        # apply fully connected linear layer from 32 features to 1 output
        return out

    def training_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

        loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
        wandb.log({"train/loss": loss})
        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

        loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)
        wandb.log({"val/loss": loss})
        return loss
    
    def test_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"

        loss = F.mse_loss(torch.sigmoid(out), data.y.type(torch.float32).reshape(-1, 1))
        self.log("test/loss", loss, 
                 on_step=False, 
                 on_epoch=True, 
                 prog_bar=True, 
                 logger=True, 
                 batch_size=data.num_graphs,
                 sync_dist=True)
        wandb.log({"test/loss": loss})
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(params=self.parameters(), lr=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 2000], 0.3)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]