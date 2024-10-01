import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.optim as optim
from mamba_ssm import TGMamba
from torchmetrics import Accuracy, F1Score, Recall, AUROC, CohenKappa
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, SequentialLR
from torch_geometric.nn import GraphConv
from einops import repeat
import time

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
                 num_tgmamba_layers=1,
                 gconv_after_all_layers=False,
                 optimizer_name='adamw', 
                 lr=5e-4,
                 weight_decay=1e-4,
                 dropout=0.1,
                 num_epochs=100,
                 rmsnorm=True,
                 edge_learner_layers=1,
                 edge_learner_attention=True,
                 edge_learner_time_varying=True,
                 attn_time_varying=False,
                 attn_threshold=0.1,
                 attn_softmax_temp=0.01,
                 init_skip_param=0.25,
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
            self.recall = Recall(task="binary")
            self.auroc = AUROC(task="binary")
        elif dataset == 'dodh':
            num_classes = 5
            self.num_vertices = 16
            # for DOD-H
            self.macro_f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)
            self.cohen_kappa = CohenKappa(task="multiclass", num_classes=num_classes)
        elif dataset == 'bcicha':
            num_classes = 1
            self.num_vertices = 56
            self.accuracy = Accuracy(task="binary")
            self.f1 = F1Score(task="binary")
            self.recall = Recall(task="binary")
            self.auroc = AUROC(task="binary")
        elif dataset == 'mamem':
            num_classes = 5
            self.num_vertices = 8
            self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.f1 = F1Score(task="multiclass", num_classes=num_classes)
            self.recall = Recall(task="multiclass", num_classes=num_classes)

        self.seq_pool_type = seq_pool_type
        self.vertex_pool_type = vertex_pool_type
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
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
                init_skip_param=init_skip_param,
            ) for _ in range(num_tgmamba_layers)
        ])
        self.dropout = torch.nn.Dropout(dropout)
        self.gconv_after_all_layers = gconv_after_all_layers
        if self.gconv_after_all_layers:
            self.gconv = GraphConv(d_model, d_model)
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
        if self.dataset == 'tuhz' or self.dataset == 'bcicha' or self.dataset == 'mamem':
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
        # out = data.x
        assert not torch.isnan(out).any(), "NaN in input data"
        edge_index, edge_weight = data.edge_index, data.edge_weight
        for i, block in enumerate(self.blocks):
            if self.pass_edges_to_next_layer:
                out, edge_index, edge_weight = block(out, edge_index, edge_weight)  # (B*V, L, d_model)
            else:
                out, _, _ = block(out, edge_index, edge_weight)
            assert not torch.isnan(out).any(), "NaN in block output at layer {}".format(i)
            if i < len(self.blocks) - 1:
                out = self.dropout(out)
        
        out = out.view(
            batch, num_vertices, seqlen, -1
        )  # (B, V, L, d_model)
        
        if self.gconv_after_all_layers:
            if edge_weight is None and edge_index is None:
                node_indices = torch.arange(self.num_vertices, device=out.device)
                edge_index = torch.cartesian_prod(node_indices, node_indices).t()
                edge_weight = torch.ones(edge_index.size(1), device=out.device)
                
                edge_index = edge_index.repeat(1, batch) + (torch.arange(batch, device=out.device) * self.num_vertices).repeat_interleave(edge_index.size(1))
                edge_weight = edge_weight.repeat(batch)

            out_reshaped = out.view(-1, out.size(-1))  # (B * V * L, d_model)
            out_conv = self.gconv(out_reshaped, edge_index, edge_weight)
            out = out_conv.view(batch, num_vertices, seqlen, -1)  # (B, V, L, d_model)
        
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
        
        if self.dataset == 'tuhz' or self.dataset == 'bcicha':
            targets = data.y.type(torch.float32).reshape(-1, 1)
            loss = F.binary_cross_entropy_with_logits(out, targets)
            # with torch.no_grad():
            #     probs = torch.sigmoid(out)
            #     preds = (probs > 0.5).float()
            #     # Count class occurrences
            #     pred_class_counts = torch.bincount(preds.long().flatten())
            #     true_class_counts = torch.bincount(targets.long().flatten())

            #     # Calculate percentages
            #     pred_total = pred_class_counts.sum().item()
            #     true_total = true_class_counts.sum().item()

            #     # print(f"Sample raw outputs: {out[:10]}")
            #     # print(f"Sample probabilities: {probs[:10]}")
            #     # print(f"Sample predictions: {preds[:10]}")
            #     # print(f"Sample true labels: {targets[:10]}")
                
            #     print("\nPredicted class balance:")
            #     for i, count in enumerate(pred_class_counts):
            #         percentage = (count.item() / pred_total) * 100
            #         print(f"Class {i}: {count.item()} ({percentage:.2f}%)")

            #     print("\nTrue class balance:")
            #     for i, count in enumerate(true_class_counts):
            #         percentage = (count.item() / true_total) * 100
            #         print(f"Class {i}: {count.item()} ({percentage:.2f}%)")
            
            log_dict = {
                "train/loss": loss,
            }
        elif self.dataset == 'dodh' or self.dataset == 'mamem':
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

        if self.dataset == 'tuhz' or self.dataset == 'bcicha':
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
        elif self.dataset == 'dodh' or self.dataset == 'mamem':
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
        
        if self.dataset == 'tuhz' or self.dataset == 'bcicha':
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
        elif self.dataset == 'dodh' or self.dataset == 'mamem':
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
        
    def on_validation_epoch_end(self):
        if self.dataset == 'tuhz' or self.dataset == 'bcicha':
            preds, probs, targets = torch.cat(self.val_preds), torch.cat(self.val_probs), torch.cat(self.val_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            recall = self.recall(preds, targets)
            auroc = self.auroc(probs, targets)
            self.log_dict({
                "val/auroc": auroc,
                "val/f1": f1,
                "val/recall": recall,
                "val/accuracy": accuracy,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_preds = []
            self.val_probs = []
            self.val_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.auroc.reset()
            self.recall.reset()
        elif self.dataset == 'mamem':
            preds, targets = torch.cat(self.val_preds), torch.cat(self.val_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            recall = self.recall(preds, targets)
            self.log_dict({
                "val/f1": f1,
                "val/recall": recall,
                "val/accuracy": accuracy,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.val_preds = []
            self.val_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.recall.reset()
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
        if self.dataset == 'tuhz' or self.dataset == 'bcicha':
            preds, probs, targets = torch.cat(self.test_preds), torch.cat(self.test_probs), torch.cat(self.test_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            auroc = self.auroc(probs, targets)
            recall = self.recall(preds, targets)

            self.log_dict({
                "test/auroc": auroc,
                "test/f1": f1,
                "test/recall": recall,
                "test/accuracy": accuracy,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.test_preds = []
            self.test_probs = []
            self.test_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.auroc.reset()
            self.recall.reset()
        elif self.dataset == 'mamem':
            preds, targets = torch.cat(self.test_preds), torch.cat(self.test_targets)
            accuracy = self.accuracy(preds, targets)
            f1 = self.f1(preds, targets)
            recall = self.recall(preds, targets)
            self.log_dict({
                "test/f1": f1,
                "test/recall": recall,
                "test/accuracy": accuracy,
            }, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.test_preds = []
            self.test_targets = []
            self.accuracy.reset()
            self.f1.reset()
            self.recall.reset()
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
            T_max=self.num_epochs,  # Total epochs - warmup epochs
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
        
        

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.

    Args:
        adj (Tensor): The dense adjacency matrix of shape
            :obj:`[num_nodes, num_nodes]` or
            :obj:`[batch_size, num_nodes, num_nodes]`.

    :rtype: (:class:`LongTensor`, :class:`Tensor`)

    Examples:

        >>> # Forr a single adjacency matrix
        >>> adj = torch.tensor([[3, 1],
        ...                     [2, 0]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1],
                [0, 1, 0]]),
        tensor([3, 1, 2]))

        >>> # For two adjacency matrixes
        >>> adj = torch.tensor([[[3, 1],
        ...                      [2, 0]],
        ...                     [[0, 1],
        ...                      [0, 2]]])
        >>> dense_to_sparse(adj)
        (tensor([[0, 0, 1, 2, 3],
                [0, 1, 0, 3, 3]]),
        tensor([3, 1, 2, 1, 2]))
    """
    if adj.dim() < 2 or adj.dim() > 3:
        raise ValueError(f"Dense adjacency matrix 'adj' must be 2- or "
                         f"3-dimensional (got {adj.dim()} dimensions)")

    edge_index = adj.nonzero().t()

    if edge_index.size(0) == 2:
        edge_attr = adj[edge_index[0], edge_index[1]]
        return edge_index, edge_attr
    else:
        edge_attr = adj[edge_index[0], edge_index[1], edge_index[2]]
        row = edge_index[1] + adj.size(-2) * edge_index[0]
        col = edge_index[2] + adj.size(-1) * edge_index[0]
        return torch.stack([row, col], dim=0), edge_attr
