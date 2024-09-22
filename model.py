import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
from torchmetrics import Accuracy, F1Score, AUROC, CohenKappa
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR


class LightGTMamba(L.LightningModule):
    def __init__(self, 
                 dataset='tuhz',
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
                 attn_time_varying=False,
                 attn_threshold=0.1,
                 attn_softmax_temp=0.01,
                 pass_edges_to_next_layer=False,
                 **kwargs):
        super().__init__()
        torch.set_float32_matmul_precision('high') 

        self.dataset = dataset
        if dataset == 'tuhz':
            num_classes = 1
            self.num_vertices = 19
            self.accuracy = Accuracy(task="binary")
            self.f1 = F1Score(task="binary")
            self.auroc = AUROC(task="binary")
        elif dataset == 'dodh':
            num_classes = 5
            self.num_vertices = 16
            # for DOD-H
            self.macro_f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)
            self.cohen_kappa = CohenKappa(task="multiclass", num_classes=num_classes)

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
        self.in_proj = torch.nn.Linear(input_dim, d_model)
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
                edge_learner_time_varying=edge_learner_time_varying, # True
                attn_time_varying=attn_time_varying, # False
                attn_threshold=attn_threshold,
                attn_softmax_temp=attn_softmax_temp,
            ) for _ in range(num_tgmamba_layers)
        ])
        self.classifier = torch.nn.Linear(d_model, num_classes) # num_classes = 1 for tuhz or 5 for dodh

        # Accumulate validation and test predictions and targets for computing metrics
        self.val_preds = []
        self.val_probs = []
        self.val_targets = []
        self.test_preds = []
        self.test_probs = []
        self.test_targets = []

    def forward(self, data: Data):
        """
        input data: torch_geometric.data.Data object

        Output: torch.Tensor of shape (B, 1) representing predictions
        """
        # print("data.x.size(): ", data.x.size())
        num_vertices = self.num_vertices
        batch, seqlen = None, None
        if self.dataset == 'tuhz':
            batch, seqlen, _ = data.x.size()
            batch = batch // num_vertices
        elif self.dataset == 'dodh':
            # currently torch.Size([256, 16, 129, 31])
            batch = data.x.size(0)
            seqlen = data.x.size(-1)
            dim = data.x.size(-2)
            data.x = data.x.permute(0, 1, 3, 2).contiguous().view(-1, 31, 129)
            # want to reshape to torch.Size([256 * 16, 31, 129])

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

        out = self.classifier(out)  # (B, 1)
        return out

    def training_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        
        if self.dataset == 'tuhz':
            targets = data.y.type(torch.float32).reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, targets)
            
            log_dict = {
                "train/loss": loss,
            }
        elif self.dataset == 'dodh':
            targets = data.y
            loss = F.cross_entropy(out, targets)

            log_dict = {
                "train/loss": loss,
            }
        else:
            raise NotImplementedError
        
        # Log all metrics at once
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

        return loss

    def validation_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"

        if self.dataset == 'tuhz':
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).type(torch.float32)
            targets = data.y.type(torch.float32).reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, targets)
            self.val_preds.append(preds)
            self.val_probs.append(probs)
            self.val_targets.append(targets)
            
            log_dict = {
                "val/loss": loss,
            }
        elif self.dataset == 'dodh':
            targets = data.y
            loss = F.cross_entropy(out, targets)
            probs = torch.softmax(out, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            # print("Unique predicted classes:", torch.unique(preds))
            # print("Unique true classes:", torch.unique(targets))
            # print("Predicted class counts:", torch.bincount(preds))
            # print("True class counts:", torch.bincount(targets))
            self.val_preds.append(preds)
            self.val_targets.append(targets)

            log_dict = {
                "val/loss": loss,
            }
        else:
            raise NotImplementedError
        
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

        return loss

    def test_step(self, data, batch_idx):
        out = self(data)
        assert out.size(0) == data.y.size(0), "Batch size mismatch"
        
        if self.dataset == 'tuhz':
            probs = torch.sigmoid(out)
            preds = (probs > 0.5).type(torch.float32)
            targets = data.y.type(torch.float32).reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, targets)

            self.test_preds.append(preds)
            self.test_probs.append(probs)
            self.test_targets.append(targets)
            log_dict = {
                "test/loss": loss,
            }
        elif self.dataset == 'dodh':
            targets = data.y
            loss = F.cross_entropy(out, targets)
            probs = torch.softmax(out, dim=-1)
            preds = torch.argmax(probs, dim=-1)
            self.test_preds.append(preds)
            self.test_targets.append(targets)

            log_dict = {
                "test/loss": loss,
            }
        else:
            raise NotImplementedError

        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=data.num_graphs, sync_dist=True)

        return loss
    
    def on_train_epoch_end(self):
        pass
        # if self.dataset == 'tuhz':
        #     self.accuracy.reset()
        #     self.f1.reset()
        #     self.auroc.reset()
        # elif self.dataset == 'dodh':
        #     self.macro_f1.reset()
        #     self.cohen_kappa.reset()
        # else:
        #     raise NotImplementedError
        
    def on_validation_epoch_end(self):
        if self.dataset == 'tuhz':
            preds, probs, targets = torch.cat(self.val_preds), torch.cat(self.val_probs), torch.cat(self.val_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            auroc = self.auroc(probs, targets)
            self.log_dict({
                "val/accuracy": accuracy,
                "val/f1": f1,
                "val/auroc": auroc,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_preds = []
            self.val_probs = []
            self.val_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.auroc.reset()
        elif self.dataset == 'dodh':
            preds, targets = torch.cat(self.val_preds), torch.cat(self.val_targets)
            macro_f1 = self.macro_f1(preds, targets)
            cohen_kappa = self.cohen_kappa(preds, targets)

            self.log_dict({
                "val/macro_f1": macro_f1,
                "val/cohen_kappa": cohen_kappa,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_preds = []
            self.val_targets = []
            self.macro_f1.reset()
            self.cohen_kappa.reset()
        else:
            raise NotImplementedError

    def on_test_epoch_end(self):
        if self.dataset == 'tuhz':
            preds, probs, targets = torch.cat(self.test_preds), torch.cat(self.test_probs), torch.cat(self.test_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            auroc = self.auroc(probs, targets)
            self.log_dict({
                "test/accuracy": accuracy,
                "test/f1": f1,
                "test/auroc": auroc,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.test_preds = []
            self.test_probs = []
            self.test_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.auroc.reset()
        elif self.dataset == 'dodh':
            preds, targets = torch.cat(self.test_preds), torch.cat(self.test_targets)
            macro_f1 = self.macro_f1(preds, targets)
            cohen_kappa = self.cohen_kappa(preds, targets)

            self.log_dict({
                "test/macro_f1": macro_f1,
                "test/cohen_kappa": cohen_kappa,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            
            self.test_preds = []
            self.test_targets = []
            self.macro_f1.reset()
            self.cohen_kappa.reset()
        else:
            raise NotImplementedError
    
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