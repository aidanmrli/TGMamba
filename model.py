import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
from torchmetrics import Accuracy, F1Score, AUROC
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR


class LightGTMamba(L.LightningModule):
    def __init__(self, 
                 num_vertices=19, 
                 conv_type='graphconv', 
                 seq_pool_type='last', 
                 vertex_pool_type='mean',
                 input_dim=100, 
                 d_model=32, 
                 d_state=16, 
                 d_conv=4,
                 num_tgmamba_layers=2,
                 optimizer_name='adamw', 
                 lr=5e-4,
                 weight_decay=1e-4,
                 rmsnorm=True,
                 edge_learner_layers=1,
                 edge_learner_attention=True,
                 edge_learner_time_varying=True,
                 attn_threshold=0.1,
                 attn_softmax_temp=0.01,
                 pass_edges_to_next_layer=False,
                 **kwargs):
        super().__init__()
        self.num_vertices = num_vertices
        self.seq_pool_type = seq_pool_type
        self.vertex_pool_type = vertex_pool_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.rmsnorm = rmsnorm
        self.edge_learner_layers = edge_learner_layers
        self.edge_learner_attention = edge_learner_attention
        self.edge_learner_time_varying = edge_learner_time_varying
        self.pass_edges_to_next_layer = pass_edges_to_next_layer

        # if FFT, input_dim = 100. Else, input_dim = 200.
        print("Input dim: ", input_dim)
        print("d_model: ", d_model)
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
                edge_learner_layers=edge_learner_layers,
                edge_learner_attention=edge_learner_attention,
                edge_learner_time_varying=edge_learner_time_varying,
                attn_threshold=attn_threshold,
                attn_softmax_temp=attn_softmax_temp,
            ) for _ in range(num_tgmamba_layers)
        ])
        self.fc = torch.nn.Linear(d_model, 1)
        torch.set_float32_matmul_precision('high') 

        # Initialize metrics
        self.accuracy = Accuracy(task="binary")
        self.f1 = F1Score(task="binary")
        self.auroc = AUROC(task="binary")

    def forward(self, data: Data):
        """
        input data: torch_geometric.data.Data object

        Output: torch.Tensor of shape (B, 1) representing predictions
        """
        # Normalize input data
        # clip = (data.x.float() - data.x.float().mean(2, keepdim=True)) / (data.x.float().std(2, keepdim=True) + 1e-10)
        batch, seqlen, _ = data.x.size()
        num_vertices = self.num_vertices
        batch = batch // num_vertices
        # print("data.x.size(): ", data.x.size())
        out = self.in_proj(data.x)  # (B*V, L, d_model)
        assert not torch.isnan(out).any(), "NaN in input data"
        # out = data.x
        edge_index, edge_weight = data.edge_index, data.edge_weight
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            if self.pass_edges_to_next_layer:
                out, edge_index, edge_weight = block(out, edge_index, edge_weight)  # (B*V, L, d_model)
            else:
                out, _, _ = block(out, edge_index, edge_weight)
            assert not torch.isnan(out).any(), "NaN in block output at layer {}".format(i)
        
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
        accuracy = self.accuracy(preds, targets)
        f1 = self.f1(preds, targets)
        auroc = self.auroc(preds, targets)

        # Create log dictionary
        log_dict = {
            "train/loss": loss,
            "train/accuracy": accuracy,
            "train/f1": f1,
            "train/auroc": auroc,
        }

        # Log all metrics at once
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

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
        accuracy = self.accuracy(preds, targets)
        f1 = self.f1(preds, targets)
        auroc = self.auroc(preds, targets)

        log_dict = {
            "val/loss": loss,
            "val/accuracy": accuracy,
            "val/f1": f1,
            "val/auroc": auroc,
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

        return {"loss": loss, "auroc": auroc}

    def test_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        assert out.size(1) == 1, "Output size mismatch: output has more than 1 feature"
        
        preds = torch.sigmoid(out)
        targets = data.y.type(torch.float32).reshape(-1, 1)
        loss = F.binary_cross_entropy_with_logits(out, targets)
        accuracy = self.accuracy(preds, targets)
        f1 = self.f1(preds, targets)
        auroc = self.auroc(preds, targets)

        log_dict = {
            "test/loss": loss,
            "test/accuracy": accuracy,
            "test/f1": f1,
            "test/auroc": auroc,
        }

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        # Reset metrics at the end of each training epoch
        self.accuracy.reset()
        self.f1.reset()
        self.auroc.reset()

    def on_validation_epoch_end(self):
        # Reset metrics at the end of each validation epoch
        self.accuracy.reset()
        self.f1.reset()
        self.auroc.reset()

    def on_test_epoch_end(self):
        # Reset metrics at the end of each test epoch
        self.accuracy.reset()
        self.f1.reset()
        self.auroc.reset()
    
    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(
                params=self.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = optim.AdamW(
                params=self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        else:
            raise NotImplementedError
            # Warm-up scheduler
        # warmup_scheduler = LinearLR(
        #     optimizer, 
        #     start_factor=0.5, 
        #     end_factor=1.0, 
        #     total_iters=1
        # )

        # Cosine annealing scheduler
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=100,  # Total epochs - warmup epochs
            eta_min=self.lr * 1e-2  # Minimum LR is 1% of initial LR
        )

        # # Combine schedulers
        # scheduler = SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, main_scheduler],
        #     milestones=[1]  # Switch to main scheduler after 5 epochs
        # )
        # scheduler = CosineAnnealingWarmRestarts(
        #     optimizer,
        #     T_0=10,
        #     T_mult=2,
        #     eta_min=1e-6,
        #     last_epoch=-1
        # )
        return {"optimizer": optimizer, "lr_scheduler": main_scheduler, "monitor": "val/loss"}
        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]