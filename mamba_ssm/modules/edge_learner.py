import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch_geometric.utils import to_dense_adj, dense_to_sparse, cumsum, scatter



class EdgeLearner(nn.Module):
    def __init__(self, d_model, num_vertices, num_edges, num_layers=1, use_attention=True, time_varying=True):
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
            self.attn_skip_param = nn.Parameter(torch.tensor(0.5))
            self.attn_scale = nn.Parameter(torch.sqrt(torch.tensor(d_model, dtype=torch.float)), requires_grad=False)
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
            self.skip_param = nn.Parameter(torch.tensor(0.5))

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
        # print("edge_weight.shape: ", edge_weight.shape)
        # print("edge_weight.dim: ", edge_weight.dim())
        # means we only have edges for each item in the batch, not for each time step
        if edge_weight.dim() < 2:
            edge_index = repeat(edge_index, 'c b -> c b l', c=2, l=seq_len)
            edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
            
        # edge_index should now have shape (2, batch_size * num_edges, seq_len)
        # edge_weight should now have shape (batch_size * num_edges, seq_len)
        if self.use_attention:
            hidden_states = hidden_states.view(batch_size, self.num_vertices, seq_len, -1)
            if self.time_varying:
                # edge_weights = []
                # for timestep in range(seq_len):
                #     context = hidden_states[:, :, timestep, :]  # (batch_size, num_vertices, d_model)
                #     edge_index_curr_timestep = edge_index[:, :, timestep]  # (2, batch_size * num_edges)
                #     edge_weight_curr_timestep = edge_weight[:, timestep]  # (batch_size * num_edges)
                #     edge_weight_curr_timestep = self.attn_edge_weights(edge_index_curr_timestep, edge_weight_curr_timestep, batch_size, context)
                #     edge_weights.append(edge_weight_curr_timestep)

                # edge_weight = torch.stack(edge_weights, dim=-1)  # (batch_size * num_edges, seq_len)
                edge_weight = self.parallel_attn_edge_weights(edge_index, edge_weight, batch_size, hidden_states)
                assert edge_weight.shape == (batch_size * self.num_edges, seq_len), "Edge weight size mismatch"
            else:
                # NOTE: edge_index should be identical for all graphs in the batch
                edge_index_curr = edge_index[:, :, 0]  # (2, batch_size * num_edges)

                # mean pooling over time
                hidden_states = hidden_states.mean(dim=2) # (batch_size, num_vertices, d_model)
                edge_weight_curr = edge_weight.mean(dim=-1)  # (batch_size * num_edges)
                
                edge_weight = self.attn_edge_weights(edge_index_curr, edge_weight_curr, batch_size, hidden_states)
                edge_weight = repeat(edge_weight, 'b -> b l', l=seq_len)
                # print((batch_size * self.num_edges, seq_len))
        else:
            # edge_index should now have shape (2, batch_size * num_edges, seq_len)
            # edge_weight should now have shape (batch_size * num_edges, seq_len)
            # Simple linear transformation of edge weights
            edge_weight = rearrange(edge_weight, '(b e) l -> (b l) e', b=batch_size, e=self.num_edges, l=seq_len)
            dynamic_edge_weight = self.edge_transform(edge_weight)
            dynamic_edge_weight = torch.sigmoid(dynamic_edge_weight) # (batch_size * seq_len, num_edges)
            edge_weight = self.skip_param * edge_weight + (1 - self.skip_param) * dynamic_edge_weight   # (batch_size * seq_len, num_edges)
            edge_weight = rearrange(edge_weight, '(b l) e -> (b e) l', b=batch_size, e=self.num_edges, l=seq_len)

        assert edge_index.shape == (2, batch_size * self.num_edges, seq_len), "Returned edge index size mismatch"
        assert edge_weight.shape == (batch_size * self.num_edges, seq_len), "Returned edge weight size mismatch"
        return edge_index, edge_weight

    def attn_edge_weights(self, edge_index, edge_weight, batch_size, context):
        """
        Calculate edge weights using attention mechanism for a single timestep.
        Edge_index: (2, batch_size * num_edges)
        Edge_weight: (batch_size * num_edges,)
        Context: (batch_size, num_vertices, d_model)

        Returns edge_weight: (batch_size * num_edges,)
        """
        Q = self.Q_proj(context)
        K = self.K_proj(context)
        batch_index = torch.arange(batch_size, device=edge_index.device).repeat_interleave(self.num_edges)
        # print("batch_index[70:74]", batch_index[70:74], "\n\n\n")
        # TODO: create a batch tensor here for the to_dense_adj function.
        # The first 0-71 edges are for the first graph in the batch, 72-143 for the second, etc.
        assert edge_index.shape == (2, batch_size * self.num_edges), "Edge index size mismatch"
        assert edge_weight.shape == (batch_size * self.num_edges,), "Edge weight size mismatch"
        # corrected_edge_index = edge_index.clone()
        # corrected_edge_index[0] = edge_index[0] % self.num_vertices
        # corrected_edge_index[1] = edge_index[1] % self.num_vertices
        # print("edge_index min and max:", edge_index.min().item(), edge_index.max().item())
        # print("batch_index min and max:", batch_index.min().item(), batch_index.max().item())
        for i in range(5):  # Check first 5 edges
            print(f"Edge {i}: nodes {edge_index[0, i]}->{edge_index[1, i]}, batch {batch_index[i]}")
            print(f"Edge weight: {edge_weight[i]}")
        print("...\n\n\n")
        for i in range(5):  # Check first 5 edges in next batch
            print(f"Edge {72+i}: nodes {edge_index[0, 72+i]}->{edge_index[1, 72+i]}, batch {batch_index[72+i]}")
            print(f"Edge weight: {edge_weight[72+i]}")



        # max_num_nodes=self.num_vertices, batch_size=batch_size
        adj_mat_batch = to_dense_adj(edge_index, batch=batch_index, edge_attr=edge_weight, max_num_nodes=self.num_vertices, batch_size=batch_size)
        print("adj_mat_batch.shape: ", adj_mat_batch.shape, "\n\n\n")
        print("adj_mat_batch[1, 0:5, 0:5]: ", adj_mat_batch[1, 0:5, 0:5], "\n\n\n")
        # assert ((adj_mat_batch.size(0) == batch_size) and 
        #             (adj_mat_batch.size(1) == self.num_vertices) and 
        #             (adj_mat_batch.size(2) == self.num_vertices)), "Adjacency matrix size mismatch"
        non_zero_entries = torch.sum(adj_mat_batch != 0)
        # TODO: this is not matching, we have 814 vs expected 4608
        # TODO: add self-loop, i.e. 1 to the diagonal of the adjacency matrix
        print("Before Mask: non_zero_entries in batched adj-matrix ", non_zero_entries, "\n\n\n")
        print("Before Mask: expected non_zero_entries in adj-matrix ", batch_size * self.num_edges, "\n\n\n")
        print("Before mask: edge_weight non_zero_entries: ", torch.sum(edge_weight != 0), "\n\n\n")
        print("edge_weight.shape: ", edge_weight.shape, "\n\n\n")
        raise SystemExit("Stop here")
        mask = torch.where(adj_mat_batch > 0, torch.tensor(0.0), torch.tensor(-float('inf')))
        print("mask.shape: ", mask.shape, "\n\n\n")
        print("mask[0]: ", mask[0], "\n\n\n")
        print("mask[1]: ", mask[1], "\n\n\n")
        att_scores = torch.bmm(Q, K.transpose(-2, -1)) / self.attn_scale
        print("att_scores.shape: ", att_scores.shape, "\n\n\n")
        print("att_scores[0]: ", att_scores[0], "\n\n\n")
        print("att_scores[1]: ", att_scores[1], "\n\n\n")
        assert att_scores.shape == adj_mat_batch.shape, "Attention scores and adjacency matrix size mismatch"
        assert mask.shape == att_scores.shape, "Mask and attention scores size mismatch"
        # Mask the attention scores that don't correspond to edges in the original graph
        att_scores = att_scores + mask
        att_scores = F.softmax(att_scores, dim=-1) # (batch_size, num_vertices, num_vertices)
        print("After mask att_scores.shape: ", att_scores.shape, "\n\n\n")
        print("After mask att_scores[0]: ", att_scores[0], "\n\n\n")
        print("After mask att_scores[1]: ", att_scores[1], "\n\n\n")
        # add skip connection
        adj_mat_batch = self.attn_skip_param * adj_mat_batch + (1 - self.attn_skip_param) * att_scores
        print("adj_mat_batch.shape: ", adj_mat_batch.shape, "\n\n\n")
        # Convert batched adjacency matrix to batched edge index and batched edge weight
        print("adj_mat[0]: ", adj_mat_batch[0], "\n\n\n")
        print("adj_mat[1]: ", adj_mat_batch[0], "\n\n\n")
        print("adj_mat[2]: ", adj_mat_batch[0], "\n\n\n")
        # Find how many entries are non-zero in the adjacency matrix
        non_zero_entries = torch.sum(adj_mat_batch != 0)
        print("actual non_zero_entries: ", non_zero_entries, "\n\n\n")
        print("expected non_zero_entries: ", batch_size * self.num_edges, "\n\n\n")
        assert batch_size * self.num_edges == torch.sum(adj_mat_batch != 0), "Number of non-zero entries does not match expected value"
        _, edge_weight = dense_to_sparse(adj_mat_batch)
        print("edge_weight.shape: ", edge_weight.shape, "\n\n\n")
        positive_mask = edge_weight > 0
        edge_weight = edge_weight[positive_mask]
        print("edge_weight.shape after filter: ", edge_weight.shape, "\n\n\n")
        return edge_weight
    
    def parallel_attn_edge_weights(self, edge_index, edge_weight, batch_size, hidden_states):
        """
        Calculate edge weights using attention mechanism for all timesteps in parallel.

        hidden_states: (batch_size, num_vertices, seq_len, d_model)
        edge_index: (2, batch_size * num_edges, seq_len)
        edge_weight: (batch_size * num_edges, seq_len)

        Returns edge_weight: (batch_size * num_edges, seq_len)
        """
        seq_len = hidden_states.size(2)
        
        # Project hidden states to Q and K
        Q = self.Q_proj(hidden_states)  # (batch_size, num_vertices, seq_len, d_model)
        K = self.K_proj(hidden_states)  # (batch_size, num_vertices, seq_len, d_model)
        
        # Compute attention scores
        att_scores = torch.einsum('bvld,bwld->bvwl', Q, K) / self.attn_scale
        
        # Create adjacency matrices for all timesteps
        adj_mat_batch = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=self.num_vertices, batch_size=batch_size * seq_len)
        adj_mat_batch = adj_mat_batch.view(batch_size, seq_len, self.num_vertices, self.num_vertices).permute(0, 2, 3, 1)
        
        # Create mask
        mask = torch.where(adj_mat_batch > 0, torch.tensor(0.0), torch.tensor(-float('inf')))
        
        # Apply mask and softmax
        att_scores = att_scores + mask
        att_scores = F.softmax(att_scores, dim=2)
        
        # Add skip connection
        adj_mat_batch = self.attn_skip_param * adj_mat_batch + (1 - self.attn_skip_param) * att_scores
        
        # Reshape back to match the original edge_weight shape
        adj_mat_batch = adj_mat_batch.permute(0, 3, 1, 2).reshape(batch_size * seq_len, self.num_vertices, self.num_vertices)
        
        # Convert back to edge index and edge weight format
        _, edge_weight_new = custom_dense_to_sparse_parallel(adj_mat_batch)
        edge_weight_new = edge_weight_new.view(batch_size, self.num_edges, seq_len).transpose(1, 2).reshape(batch_size * self.num_edges, seq_len)
        
        return edge_weight_new


# def to_dense_adj(
#     edge_index: Tensor,
#     batch: OptTensor = None,
#     edge_attr: OptTensor = None,
#     max_num_nodes: Optional[int] = None,
#     batch_size: Optional[int] = None,
# ) -> Tensor:
#     r"""Converts batched sparse adjacency matrices given by edge indices and
#     edge attributes to a single dense batched adjacency matrix.

#     Args:
#         edge_index (LongTensor): The edge indices.
#         batch (LongTensor, optional): Batch vector
#             :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
#             node to a specific example. (default: :obj:`None`)
#         edge_attr (Tensor, optional): Edge weights or multi-dimensional edge
#             features.
#             If :obj:`edge_index` contains duplicated edges, the dense adjacency
#             matrix output holds the summed up entries of :obj:`edge_attr` for
#             duplicated edges. (default: :obj:`None`)
#         max_num_nodes (int, optional): The size of the output node dimension.
#             (default: :obj:`None`)
#         batch_size (int, optional): The batch size. (default: :obj:`None`)

#     :rtype: :class:`Tensor`

#     Examples:
#         >>> edge_index = torch.tensor([[0, 0, 1, 2, 3],
#         ...                            [0, 1, 0, 3, 0]])
#         >>> batch = torch.tensor([0, 0, 1, 1])
#         >>> to_dense_adj(edge_index, batch)
#         tensor([[[1., 1.],
#                 [1., 0.]],
#                 [[0., 1.],
#                 [1., 0.]]])

#         >>> to_dense_adj(edge_index, batch, max_num_nodes=4)
#         tensor([[[1., 1., 0., 0.],
#                 [1., 0., 0., 0.],
#                 [0., 0., 0., 0.],
#                 [0., 0., 0., 0.]],
#                 [[0., 1., 0., 0.],
#                 [1., 0., 0., 0.],
#                 [0., 0., 0., 0.],
#                 [0., 0., 0., 0.]]])

#         >>> edge_attr = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
#         >>> to_dense_adj(edge_index, batch, edge_attr)
#         tensor([[[1., 2.],
#                 [3., 0.]],
#                 [[0., 4.],
#                 [5., 0.]]])
#     """
#     if batch is None:
#         max_index = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
#         batch = edge_index.new_zeros(max_index)

#     if batch_size is None:
#         batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

#     one = batch.new_ones(batch.size(0))
#     num_nodes = scatter(one, batch, dim=0, dim_size=batch_size, reduce='sum')
#     cum_nodes = cumsum(num_nodes)

#     idx0 = batch[edge_index[0]]
#     idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
#     idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

#     if max_num_nodes is None:
#         max_num_nodes = int(num_nodes.max())

#     elif ((idx1.numel() > 0 and idx1.max() >= max_num_nodes)
#           or (idx2.numel() > 0 and idx2.max() >= max_num_nodes)):
#         mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
#         idx0 = idx0[mask]
#         idx1 = idx1[mask]
#         idx2 = idx2[mask]
#         edge_attr = None if edge_attr is None else edge_attr[mask]

#     if edge_attr is None:
#         edge_attr = torch.ones(idx0.numel(), device=edge_index.device)

#     size = [batch_size, max_num_nodes, max_num_nodes]
#     size += list(edge_attr.size())[1:]
#     flattened_size = batch_size * max_num_nodes * max_num_nodes

#     idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
#     adj = scatter(edge_attr, idx, dim=0, dim_size=flattened_size, reduce='sum')
#     adj = adj.view(size)

#     return adj

def custom_dense_to_sparse(adj, num_edges_per_graph=72):
    batch_size, num_nodes, _ = adj.size()

    # Flatten the last two dimensions
    adj_flat = adj.view(batch_size, -1)

    # Get the top 72 edge weights (in absolute value) for each graph in the batch
    top_values, top_indices = torch.topk(adj_flat.abs(), k=num_edges_per_graph, dim=1)

    # Get the actual values (not absolute) for these edges
    edge_weights = torch.gather(adj_flat, 1, top_indices)

    # Flatten the edge weights
    edge_weights = edge_weights.view(-1)

    return edge_weights

def custom_dense_to_sparse_parallel(adj):
    batch_size, seq_len, num_nodes, _ = adj.size()
    
    # Create a batch index tensor
    batch_index = torch.arange(batch_size * seq_len, device=adj.device).repeat_interleave(num_nodes * num_nodes)
    
    # Flatten the adjacency tensor
    adj_flat = adj.reshape(-1)
    
    # Get non-zero indices and values
    non_zero = adj_flat.nonzero().squeeze()
    edge_index = torch.stack([non_zero // (num_nodes * num_nodes), 
                              (non_zero % (num_nodes * num_nodes)) // num_nodes, 
                              (non_zero % (num_nodes * num_nodes)) % num_nodes], dim=0)
    edge_attr = adj_flat[non_zero]
    
    # Add batch and sequence dimensions to edge_index
    edge_index = torch.cat([batch_index[non_zero].unsqueeze(0), edge_index[1:]], dim=0)
    
    return edge_index, edge_attr


