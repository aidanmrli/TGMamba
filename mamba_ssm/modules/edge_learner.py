from typing import Optional, Tuple, Union
from torch_geometric.typing import OptTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
# from torch_sparse import SparseTensor
from einops import rearrange, repeat
# use torch geometric 2.3.0 versions
from torch_geometric.utils import remove_self_loops, add_self_loops, to_dense_adj

class EdgeLearner(nn.Module):
    def __init__(self, d_model, 
                 num_vertices, 
                 num_linear_layers=1, 
                 use_attention=True,
                 edge_learner_time_varying=True,
                 attn_time_varying=False,
                 attention_threshold=0.1, 
                 temperature=0.01,
                 init_skip_param=0.25):
        super().__init__()
        self.d_model = d_model
        self.num_vertices = num_vertices
        self.use_attention = use_attention
        self.edge_learner_time_varying = edge_learner_time_varying
        self.attn_time_varying = attn_time_varying
        self.num_linear_layers = num_linear_layers
        self.act = nn.SiLU()

        if use_attention:
            self.Q_proj = nn.Linear(d_model, d_model * 4)
            self.K_proj = nn.Linear(d_model, d_model * 4)
            # self.attn_skip_param = nn.Parameter(torch.tensor(0.5))
            self.attn_threshold = attention_threshold
            self.attn_scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float))
            self.softmax_temperature = temperature

        if edge_learner_time_varying and not attn_time_varying:
            self.edge_transform = nn.Sequential(
                nn.Linear(d_model * 2 + 1, d_model),  # 2 * d_model for pair of nodes, +1 for original edge weight
                self.act,
                nn.Linear(d_model, d_model // 2),
                self.act,
                nn.Linear(d_model // 2, 1)
            )
            self.skip_param = nn.Parameter(torch.tensor(init_skip_param))   # if 1.0, use original edge weights, if 0.0, use learned edge weights

    def forward(self, hidden_states, edge_index, edge_weight):
        """
        hidden_states: (batch_size * num_vertices, seq_len, d_model)
        edge_index: (2, batch_size * num_edges) or (2, batch_size * num_edges, seq_len)
        edge_weight: (batch_size * num_edges,) or (batch_size * num_edges, seq_len)

        Returns:
        edge_index, edge_weight: (2, batch_size * num_edges, seq_len), (batch_size * num_edges, seq_len)
        """
        batch_size = hidden_states.size(0) // self.num_vertices
        seq_len = hidden_states.size(1)
        d_model = hidden_states.size(-1)
        batch_adj_mat = None
        # means we only have edges for each item in the batch, not for each time step
        if edge_weight is not None and edge_weight.dim() < 2: # should always be true
            batch = torch.arange(batch_size, device=edge_index.device).repeat_interleave(self.num_vertices)
            batch_adj_mat = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight, max_num_nodes=self.num_vertices, batch_size=batch_size)
            assert batch_adj_mat.size() == (batch_size, self.num_vertices, self.num_vertices), "Batch adjacency matrix size mismatch"
            edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
        
        # edge_index should now have shape (2, batch_size * num_edges, seq_len)
        # edge_weight should now have shape (batch_size * num_edges, seq_len)
       
        if self.use_attention:
            hidden_states_new = hidden_states.view(batch_size, self.num_vertices, seq_len, -1)
            
            if batch_adj_mat is not None and batch_adj_mat.dim() == 2:
                batch_adj_mat = batch_adj_mat.unsqueeze(0).expand(batch_size, self.num_vertices, self.num_vertices).contiguous()
           
            if self.attn_time_varying:
                Q = self.Q_proj(hidden_states_new)  # (batch_size, num_vertices, seq_len, d_model)
                K = self.K_proj(hidden_states_new)  # (batch_size, num_vertices, seq_len, d_model)
                attention_scores = torch.einsum('bvld,bwld->bvwl', Q, K) / self.attn_scale
                attention_scores = F.softmax(attention_scores / self.softmax_temperature, dim=-2)  # (batch_size, num_vertices, num_vertices, seq_len)
                # attention_scores = F.softmax(attention_scores / self.softmax_temperature, dim=-3)  # (batch_size, num_vertices, num_vertices, seq_len)              

                # make the attention scores symmetric for the undirected graph
                attention_scores = (attention_scores + attention_scores.transpose(1, 2)) / 2
                
                # Prune attention scores below threshold
                attention_scores = torch.where(attention_scores >= self.attn_threshold, attention_scores, torch.zeros_like(attention_scores))
                edge_index_all = []
                edge_weight_all = []
                
                for i in range(seq_len):
                    curr_adj_mat = attention_scores[:, :, :, i]
                    if batch_adj_mat is not None:
                        assert curr_adj_mat.size() == (batch_size, self.num_vertices, self.num_vertices)
                        curr_adj_mat = curr_adj_mat + batch_adj_mat     # skip connection
                    
                    edge_index, edge_weight = dense_to_sparse(curr_adj_mat)
                
                    # add self-loop
                    edge_index, edge_weight = remove_self_loops(
                        edge_index=edge_index, edge_attr=edge_weight
                    )
                    edge_index, edge_weight = add_self_loops(
                        edge_index=edge_index,
                        edge_attr=edge_weight,
                        fill_value=1,
                    )
                    edge_index_all.append(edge_index)
                    edge_weight_all.append(edge_weight)
                # edge_index = torch.stack(edge_index_all, dim=-1)
                # edge_weight = torch.stack(edge_weight_all, dim=-1)
                
                max_edges = max(edge_index.size(1) for edge_index in edge_index_all)

                padded_edge_index_all = []
                padded_edge_weight_all = []

                # the number of these is equal to the sequence length
                for edge_index, edge_weight in zip(edge_index_all, edge_weight_all):
                    num_edges = edge_index.size(1)
                    padding_size = max_edges - num_edges
                    
                    padded_edge_index = F.pad(edge_index, (0, padding_size), value=-1)  # Use -1 as padding for edge_index
                    padded_edge_weight = F.pad(edge_weight, (0, padding_size), value=0)  # Use 0 as padding for edge_weight
                    
                    padded_edge_index_all.append(padded_edge_index)
                    padded_edge_weight_all.append(padded_edge_weight)

                edge_index = torch.stack(padded_edge_index_all, dim=-1)
                edge_weight = torch.stack(padded_edge_weight_all, dim=-1)
                    
            else:
                # attention_scores = attention_scores.mean(dim=-1)  # (batch_size, num_vertices, num_vertices)
                hidden_states_mean = hidden_states_new.mean(dim=2)   # (batch_size, num_vertices, d_model)
                hidden_states_mean = hidden_states_mean.mean(dim=0).unsqueeze(0)  # (1, num_vertices, d_model) 
                Q = self.Q_proj(hidden_states_mean)  # (batch_size, num_vertices, d_model)
                K = self.K_proj(hidden_states_mean)  # (batch_size, num_vertices, d_model)
                attention_scores = torch.einsum('bvd,bwd->bvw', Q, K) / self.attn_scale
                # print("attention scores shape: ", attention_scores.size())
                # print("attention_scores before softmax ", attention_scores[0, :, :])
                attention_scores = F.softmax(attention_scores / self.softmax_temperature, dim=-1)  # (batch_size, num_vertices, num_vertices)
                # print("attention_scores after softmax ", attention_scores[0, :, :])
                # print("attention scores shape: ", attention_scores.size())
                

                # make the attention scores symmetric for the undirected graph
                attention_scores = (attention_scores + attention_scores.transpose(-1, -2)) / 2

                # Prune attention scores below threshold
                attention_scores = torch.where(attention_scores >= self.attn_threshold, attention_scores, torch.zeros_like(attention_scores))
                
                if batch_adj_mat is not None:
                    # assert attention_scores.size() == (batch_size, self.num_vertices, self.num_vertices)
                    attention_scores = attention_scores + batch_adj_mat     # skip connection
                
                if attention_scores.shape[0] == 1:
                    attention_scores = attention_scores.repeat(batch_size, 1, 1)
                
                for i in range(self.num_vertices):
                    attention_scores[:, i, i] = 1.0
                edge_index, edge_weight = dense_to_sparse(attention_scores)
                
                # add self-loop
                # edge_index, edge_weight = remove_self_loops(
                #     edge_index=edge_index, edge_attr=edge_weight
                # )
                # edge_index, edge_weight = add_self_loops(
                #     edge_index=edge_index,
                #     edge_attr=edge_weight,
                #     fill_value=1,
                # )   # edge_index is now (2, batch_size * num_edges), edge_weight is now (batch_size * num_edges)
                edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
                edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
                
                # edge_index should now have shape (2, batch_size * num_edges, seq_len)
                # edge_weight should now have shape (batch_size * num_edges, seq_len)
        if edge_weight is None and edge_index is None:
            # identity matrix with self loops
            adj_mat = torch.rand(self.num_vertices, self.num_vertices, device=hidden_states.device)
            adj_mat = (adj_mat + adj_mat.t()) / 2
            adj_mat.fill_diagonal_(1)
            # Duplicate the adjacency matrix batch_size times
            adj_mat = adj_mat.unsqueeze(0).expand(batch_size, self.num_vertices, self.num_vertices)
            
            # Convert to edge_index and edge_weight
            edge_index, edge_weight = dense_to_sparse(adj_mat)
            edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
            # adj_mat = torch.eye(self.num_vertices, device=hidden_states.device).unsqueeze(0).expand(batch_size, self.num_vertices, self.num_vertices)
            # edge_index, edge_weight = dense_to_sparse(adj_mat)
            # edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            # edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
            
        if self.edge_learner_time_varying and not self.attn_time_varying:
            new_edge_weights = []
            for t in range(seq_len):
                # (batch_size * num_vertices, seq_len, d_model)
                hidden_states_t = hidden_states[:, t, :].view(batch_size, self.num_vertices, d_model)    # (batch_size, num_vertices, d_model)
                try:
                    edge_index_t = edge_index[:, :, t].reshape(2, batch_size, -1)  # (2, batch_size, num_edges)
                    edge_weight_t = edge_weight[:, t].view(batch_size, -1)   # (batch_size, num_edges)
                except Exception:
                    print("edge_index shape: ", edge_index.size())
                    print("edge_weight shape: ", edge_weight.size())
                num_edges = edge_index_t.size(-1)
                source_nodes = edge_index_t[0, :, :] % self.num_vertices # (batch_size, num_edges)
                target_nodes = edge_index_t[1, :, :] % self.num_vertices # (batch_size, num_edges)
                # print("hidden_states_t shape: ", hidden_states_t.size())
                batch_indices = torch.arange(batch_size, device=hidden_states_t.device)[:, None].expand(-1, num_edges)

                source_nodes_features = hidden_states_t[batch_indices, source_nodes]  # (batch_size, num_edges, d_model)
                target_nodes_features = hidden_states_t[batch_indices, target_nodes]  # (batch_size, num_edges, d_model)
                # print("source_nodes_features shape: ", source_nodes_features.size())
                # print("edge_weight_t shape: ", edge_weight_t.size())
                # Concatenate features and edge weights
                input_features = torch.cat([
                    source_nodes_features, 
                    target_nodes_features, 
                    edge_weight_t.unsqueeze(-1)
                ], dim=-1)  # (batch_size, num_edges, 2 * d_model + 1)
                
                edge_weights_t = self.edge_transform(input_features)  # (batch_size, num_edges, 1)
                edge_weights_t = torch.sigmoid(edge_weights_t) # (batch_size, num_edges, 1)
                
                new_edge_weight_t = self.skip_param * edge_weight_t + (1 - self.skip_param) * edge_weights_t.squeeze(-1)  # (batch_size, num_edges)
                new_edge_weights.append(new_edge_weight_t)
            
            # edge_index should now have shape (2, batch_size * num_edges, seq_len)
            # edge_weight should now have shape (batch_size * num_edges, seq_len)
            edge_weight = torch.stack(new_edge_weights, dim=-1).view(-1, seq_len)
            
        if edge_weight.dim() < 2:
            edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
        
        assert edge_index.size(0) == 2 and edge_index.size(-1) == seq_len and edge_index.dim() == 3, "Returned edge index size mismatch"
        assert edge_weight.size(-1) == seq_len and edge_weight.dim() == 2, "Returned edge weight size mismatch"

        return edge_index, edge_weight

def prune_adj_mat(adj_mat, k):
    values, _ = torch.topk(adj_mat, k, dim=-1)
    thresh = values[..., -1].unsqueeze(-1).expand_as(adj_mat)
    return torch.where(adj_mat >= thresh, adj_mat, torch.zeros_like(adj_mat))


def dense_to_sparse(adj: Tensor) -> Tuple[Tensor, Tensor]:
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
