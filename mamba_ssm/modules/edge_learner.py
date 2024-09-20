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
    def __init__(self, d_model, num_vertices, num_edges, num_layers=1, use_attention=True, attention_threshold=0.1, time_varying=True, temperature=0.01):
        super().__init__()
        self.d_model = d_model
        self.num_vertices = num_vertices
        self.num_edges = num_edges
        self.use_attention = use_attention
        self.time_varying = time_varying
        self.num_layers = num_layers
        self.act = nn.SiLU()

        if use_attention:
            self.Q_proj = nn.Linear(d_model, d_model)
            self.K_proj = nn.Linear(d_model, d_model)
            # self.attn_skip_param = nn.Parameter(torch.tensor(0.5))
            self.attn_threshold = attention_threshold
            self.attn_scale = torch.sqrt(torch.tensor(d_model, dtype=torch.float))
            self.softmax_temperature = temperature

        else:
            if num_layers == 1:
                self.edge_transform = nn.Linear(num_edges, num_edges)
            else:
                layers = []
                for i in range(num_layers):
                    if i == 0:
                        layers.append(nn.Linear(num_edges, num_edges * 2))
                    elif i == num_layers - 1:
                        layers.append(nn.Linear(num_edges * 2, num_edges))
                    else:
                        layers.append(nn.Linear(num_edges * 2, num_edges * 2))
                    if i < num_layers - 1:
                        layers.append(self.act)
                self.edge_transform = nn.Sequential(*layers)
            self.skip_param = nn.Parameter(torch.tensor(0.9))   # if 1.0, use original edge weights, if 0.0, use learned edge weights

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
        # means we only have edges for each item in the batch, not for each time step
        if edge_weight.dim() < 2: # should always be true
            batch = torch.arange(batch_size, device=edge_index.device).repeat_interleave(self.num_vertices)
            batch_adj_mat = to_dense_adj(edge_index, batch=batch, edge_attr=edge_weight, max_num_nodes=self.num_vertices, batch_size=batch_size)
            # print("batch_adj_mat.size(): ", batch_adj_mat.size())
            # print("batch_adj_mat: ", batch_adj_mat[1, :10, :10])
            assert batch_adj_mat.size() == (batch_size, self.num_vertices, self.num_vertices), "Batch adjacency matrix size mismatch"
            edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
        
        # edge_index should now have shape (2, batch_size * num_edges, seq_len)
        # edge_weight should now have shape (batch_size * num_edges, seq_len)
       
        if self.use_attention:
            hidden_states = hidden_states.view(batch_size, self.num_vertices, seq_len, -1)
            
            # if batch_adj_mat and batch_adj_mat.dim() == 2:
            #     batch_adj_mat = batch_adj_mat.unsqueeze(0).expand(batch_size, self.num_vertices, self.num_vertices).contiguous()
           
            if self.time_varying:
                Q = self.Q_proj(hidden_states)  # (batch_size, num_vertices, seq_len, d_model)
                K = self.K_proj(hidden_states)  # (batch_size, num_vertices, seq_len, d_model)
                attention_scores = torch.einsum('bvld,bwld->bvwl', Q, K) / self.attn_scale
                attention_scores = F.softmax(attention_scores / self.softmax_temperature, dim=-2)  # (batch_size, num_vertices, num_vertices, seq_len
                attention_scores = F.softmax(attention_scores / self.softmax_temperature, dim=-3)  # (batch_size, num_vertices, num_vertices, seq_len)              

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
                hidden_states = hidden_states.mean(dim=2)
                 
                Q = self.Q_proj(hidden_states)  # (batch_size, num_vertices, d_model)
                K = self.K_proj(hidden_states)  # (batch_size, num_vertices, d_model)
                attention_scores = torch.einsum('bvd,bwd->bvw', Q, K) / self.attn_scale
                # attention_scores = F.softmax(attention_scores, dim=2)  # (batch_size, num_vertices, num_vertices, seq_len)

                # make the attention scores symmetric for the undirected graph
                attention_scores = (attention_scores + attention_scores.transpose(1, 2)) / 2

                # Prune attention scores below threshold
                attention_scores = torch.where(attention_scores >= self.attn_threshold, attention_scores, torch.zeros_like(attention_scores))
                
                if batch_adj_mat is not None:
                    assert attention_scores.size() == (batch_size, self.num_vertices, self.num_vertices)
                    attention_scores = attention_scores + batch_adj_mat     # skip connection
                edge_index, edge_weight = dense_to_sparse(attention_scores)
                
                # add self-loop
                edge_index, edge_weight = remove_self_loops(
                    edge_index=edge_index, edge_attr=edge_weight
                )
                edge_index, edge_weight = add_self_loops(
                    edge_index=edge_index,
                    edge_attr=edge_weight,
                    fill_value=1,
                )
            # use the attention scores as the edge indices and weights

        else:
            # edge_index should now have shape (2, batch_size * num_edges, seq_len)
            # edge_weight should now have shape (batch_size * num_edges, seq_len)
            # Simple linear transformation of edge weights
            edge_weight = rearrange(edge_weight, '(b e) l -> (b l) e', b=batch_size, e=self.num_edges, l=seq_len)
            dynamic_edge_weight = self.edge_transform(edge_weight)
            dynamic_edge_weight = torch.sigmoid(dynamic_edge_weight) # (batch_size * seq_len, num_edges)
            edge_weight = self.skip_param * edge_weight + (1 - self.skip_param) * dynamic_edge_weight   # (batch_size * seq_len, num_edges)
            edge_weight = rearrange(edge_weight, '(b l) e -> (b e) l', b=batch_size, e=self.num_edges, l=seq_len)

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

# def edge_index_to_adj_matrix(edge_index, edge_weight, num_nodes, batch_size):
#     # Ensure edge_index and edge_weight are on the same device
#     device = edge_index.device
    
#     # Create a batch vector
#     batch = torch.arange(batch_size, device=device).repeat_interleave(num_nodes)
    
#     # Create a sparse tensor
#     adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight,
#                        sparse_sizes=(batch_size * num_nodes, batch_size * num_nodes))
    
#     # Convert to a dense tensor and reshape
#     adj_dense = adj.to_dense()
#     adj_matrix = adj_dense.view(batch_size, num_nodes, num_nodes)
    
#     return adj_matrix