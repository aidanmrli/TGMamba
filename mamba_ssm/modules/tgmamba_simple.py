# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv, ChebConv, GraphConv, GATv2Conv
from mamba_ssm.modules.edge_learner import EdgeLearner

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_ref

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update  # TO UPDATE
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated


class TGMamba(nn.Module):
    def __init__(
        self,
        d_model=16,     # dim of input and output embeddings
        d_state=16,    # dim of internal state in SSM
        d_conv=4,   # width of local 1D causal convolution
        expand=2,   # block expansion factor to get the number of channels D
        num_vertices=19,  # New parameter for number of EEG channels
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        conv_type="gcnconv",
        rmsnorm=True,
        edge_learner_layers=1,
        edge_learner_attention=True,
        edge_learner_time_varying=True,
        attn_time_varying=False,
        attn_threshold=0.1,
        attn_softmax_temp=0.01,
        init_skip_param=0.25,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model  # 16 dim of input and output embeddings
        self.d_state = d_state  # 16 SSM state expansion factor (N)
        self.d_conv = d_conv    # 4 Local convolution width
        self.expand = expand    # 2 Block expansion factor
        self.d_inner = int(self.expand * self.d_model)  # 32 the number of channels (D)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank   # 1
        self.use_fast_path = use_fast_path  # True or False, usually True
        self.layer_idx = layer_idx

        ################ NEW
        self.rmsnorm = rmsnorm
        self.num_vertices = num_vertices  # Store number of vertices (EEG channels)
        self.edge_learner = EdgeLearner(d_model, 
                                        num_vertices, 
                                        edge_learner_layers, 
                                        use_attention=edge_learner_attention,
                                        edge_learner_time_varying=edge_learner_time_varying,
                                        attn_time_varying=attn_time_varying,
                                        attention_threshold=attn_threshold,
                                        temperature=attn_softmax_temp,
                                        init_skip_param=init_skip_param)
        if edge_learner_time_varying and edge_learner_attention:
            self.time_varying_attention = True
        else: 
            self.time_varying_attention = False
        if conv_type == "gcnconv":
            self.gconv_A = GCNConv(self.d_state, self.d_state, bias=False)
            self.gconv_B = GCNConv(self.d_inner, self.d_inner, bias=False)
            self.gconv_C = GCNConv(self.d_state, self.d_state, bias=False)
        elif conv_type == "chebconv":
            self.gconv_A = ChebConv(self.d_state, self.d_state, K=2, bias=False)
            self.gconv_B = ChebConv(self.d_inner, self.d_inner, K=2, bias=False)
            self.gconv_C = ChebConv(self.d_state, self.d_state, K=2, bias=False)
        elif conv_type == "graphconv":
            self.gconv_A = GraphConv(self.d_state, self.d_state, bias=False)
            self.gconv_B = GraphConv(self.d_inner, self.d_inner, bias=False)
            self.gconv_C = GraphConv(self.d_state, self.d_state, bias=False)
        elif conv_type == "gatv2conv":
            self.gconv_A = GATv2Conv(self.d_state, self.d_state, heads=1, concat=False, bias=False)
            self.gconv_B = GATv2Conv(self.d_inner, self.d_inner, heads=1, concat=False, bias=False)
            self.gconv_C = GATv2Conv(self.d_state, self.d_state, heads=1, concat=False, bias=False)
        else:
            raise NotImplementedError("Only GCNConv, ChebConv, and GraphConv are supported")
        ################
        
        # weights have shape (output_dim, input_dim). So (d_inner*2, d_model) = (64, 16)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)  # weights (64, 16) 

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,   # 32
            out_channels=self.d_inner,  # 32
            bias=conv_bias, # True (default)
            kernel_size=d_conv, # 4
            groups=self.d_inner,    # 32
            padding=d_conv - 1,    # 3
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()      # TODO: Check what to do about this activation function

        # linear projection layer that transforms from d_inner dim to dt_rank + d_state * 2 dim (from 32 to 33)
        # used to computer parameters for the SSM in Mamba
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        # used to compute delta time values for SSM
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # learnable parameters. 
        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()  # (D, N) or (d_inner, d_state) representing D different structured N x N matrices for each of the D channels
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # Extra normalization layer right before output projection
        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_inner, eps=1e-5, norm_before_gate=False, **factory_kwargs)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, edge_index, edge_weight, inference_params=None):
        """        
        hidden_states: (B*V, L, D) is torch_geometric.data.Data.x
            B*V: batch size * num_vertices
            L: sequence length
            D: dimension of the input (d_model)
        edge_index: (2, batch_size * num_edges) or (2, batch_size * num_edges, seq_len)
            E: total number of edges in the batch
        edge_index.shape:  torch.Size([2, batch_size * (72) num_edges_per_graph])
        edge_weight: (E,) total number of edges in the batch
        edge_weight.shape:  torch.Size([batch_size * (72) num_edges_per_graph])
        Returns: same shape as hidden_states
        """
        batch = hidden_states.shape[0] // self.num_vertices
        num_vertices = self.num_vertices
        seqlen = hidden_states.shape[1]
        # batch_idx = data.batch
        
        batch, seqlen, dim = hidden_states.shape    # batch = batch_size*vertices

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch, num_vertices=self.num_vertices)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # Linear projection for both the x and z parts of the input
        # We do matmul and transpose BLH -> HBL at the same time
        # in_proj weights have dim (d_inner*2, d_model) = (64, 16) so we take d and map it to 2 * d_inner.
        # We then split the 2 * d_inner into two parts, x and z later on.
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state) or (D, N)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        # if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
        #     out = mamba_inner_fn(
        #         xz,     # (B * V, 2 * d_inner, L) 
        #         self.conv1d.weight,     # (d_inner, 1, d_conv)
        #         self.conv1d.bias,    # (d_inner)
        #         self.x_proj.weight,  # (dt_rank + d_state * 2, d_inner)
        #         self.dt_proj.weight,  # (d_inner, dt_rank)
        #         self.out_proj.weight,  # (d_inner, d_model)
        #         self.out_proj.bias,  # (d_model)
        #         A,  # (D, N) or (d_inner, d_state) representing D different structured N x N matrices for each of the D channels
        #         None,  # input-dependent B
        #         None,  # input-dependent C
        #         self.D.float(),  # (d_inner) ie (D)
        #         delta_bias=self.dt_proj.bias.float(),   # (d_inner)
        #         delta_softplus=True,
        #         # NEW. Remember that A, B, C are now equivalent to W_A, W_B, W_C projections now.
        #         gconv_A = self.gconv_A,
        #         gconv_B = self.gconv_B,
        #         gconv_C = self.gconv_C,
        #         edge_index=edge_index, 
        #         edge_weight=edge_weight,
        #         num_vertices=self.num_vertices
        #     )
        # else:
        x, z = xz.chunk(2, dim=1)  # we go from 2 * d_inner to d_inner. x and z are (B*V, d_inner, L)

        # # Compute short convolution
        if conv_state is not None:
            # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B*V, D, W)

        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            assert self.activation in ["silu", "swish"]

            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
            
            # x has shape (B*V, d_inner, L)
        
            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

            # reshape x shape (BV, d_inner, L) to ((BV)*L, d_inner).
            # lin proj self.x_proj transforms dimension d from d_inner to dt_rank + d_state * 2.
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # output ((bv)l d)  ie ((B*V)*L, dt_rank + d_state * 2)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # these Deltas, A, B, and C are used as input-dependent projection weights 
        # varying with time in the new formula.
        dt = self.dt_proj.weight @ dt.t()   # (d_inner, dt_rank) @ ((B*V)*L, dt_rank)^T = (d_inner, (B*V)*L)
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)  # (BV, d_inner, L)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # (BV, d_state, L)
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()  # (BV, d_state, L)
        # print("Before selective_scan_ref")
        # print("A.shape: ", A.shape, "B.shape: ", B.shape, "C.shape: ", C.shape)
        assert self.activation in ["silu", "swish"]

        # print("edge_weight before edge_learner: ", edge_weight)
        # print("edge_weight.shape: ", edge_weight.shape)
        # print("edge_weight.dim: ", edge_weight.dim())
        # print("\n\n\n")
        edge_index, edge_weight = self.edge_learner(hidden_states, edge_index, edge_weight)
        # print("edge_weight after edge_learner: ", edge_weight)
        # print("edge_weight.shape: ", edge_weight.shape)
        # raise SystemExit("Stop here")
        assert edge_weight.dim() == 2, "edge_weight should have 2 dimensions"
        assert edge_weight.shape[-1] == seqlen, "edge_weight should have the same length as the sequence length"
        assert edge_index.dim() == 3, "edge_index should have 3 dimensions" 
        assert edge_index.shape[-1] == seqlen, "edge_index should have the same length as the sequence length"
        assert edge_index.shape[0] == 2, "edge_index should have 2 rows"
        assert not torch.isnan(x).any(), "NaN in x before ssm"

        y = selective_scan_ref(
            x,   # (B*V, d_inner, L)
            dt,  # (B*V, d_inner, L)
            A,   # (d_inner, d_state) aka (D, N)
            B,   # (B*V, d_state, L)
            C,   # (B*V, d_state, L)
            self.D.float(),  # (d_inner)
            z=z if not self.rmsnorm else None,  # (B*V, d_inner, L)
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=ssm_state is not None,
            # NEW
            gconv_A = self.gconv_A,
            gconv_B = self.gconv_B,
            gconv_C = self.gconv_C,
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_vertices=self.num_vertices,
            act=self.act,
            time_varying_attention=self.time_varying_attention
        )
        if ssm_state is not None:
            y, last_state = y
            ssm_state.copy_(last_state)
        y = rearrange(y, "b d l -> b l d")  # (B*V, L, d_inner)
        assert not torch.isnan(y).any(), "NaN in y after ssm"

        # Multiply "gate" branch and apply extra normalization layer
        if self.rmsnorm:
            z = rearrange(z, "b d l -> b l d")  # (B*V, L, d_inner)
            y = self.norm(y, z)

        out = self.out_proj(y)  # (B*V, L, d_model)
        assert not torch.isnan(y).any(), "NaN in y after out_proj"
        return out, edge_index, edge_weight

    def step(self, hidden_states, conv_state, ssm_state, edge_index, edge_weight):
        """
        hidden_states: (B*V, L, D)
            B: batch size * num_vertices
            V: number of vertices
            L: sequence length
            D: number of channels/features/heads (d_model)

        conv_state: (B*V, D, W)
            B: batch size * num_vertices
            D: number of channels/features/heads (d_model * expand)
            W: convolution width (d_conv)
        
        ssm_state: (B*V, D, N)
            B: batch size * num_vertices
            D: number of channels/features/heads (d_model * expand)
            N: SSM state expansion factor (d_state)
        
        d_state = N is the SSM state expansion factor
        d_model is the number of features of the input
        """
        batch, seqlen, dim = hidden_states.shape    # batch = batch_size*vertices
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        # project from dimension d_model to dimension N = 2 * d_inner
        xz = self.in_proj(hidden_states.squeeze(1))  # (B*V, 2 * d_inner)
        x, z = xz.chunk(2, dim=-1)  # (B*V, d_inner)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B*V, D, W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B*V, D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype) # (B*V, D)
        else:
            x = causal_conv1d_update(
                x,  # (B*V, D)
                conv_state,  # (B*V, D, W)
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,   # (D)
                self.activation,    # "silu"
            )

        # project from d_inner to dt_rank + d_state * 2
        x_db = self.x_proj(x)  # (B*V, dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B*V, d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        # if selective_state_update is None:
        # Discretize A and B
        dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))  # (B*V, d_inner)
        dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))   # (B*V, d_inner, d_state) (B*V, )
        dB = torch.einsum("bd,bn->bdn", dt, B)  # (B*V, d_inner, d_state)

        # perform A and B convolutions on prev hidden state ssm_state and current input x
        conv_A_out = self.gconv_A(ssm_state, edge_index, edge_weight)  # (batch_size*V, d_state)
        conv_B_out = self.gconv_B(x, edge_index, edge_weight)  # (batch_size*V, d_inner)

        # x is (B*V, d_inner). dB is (B*V, d_inner, d_state)
        ssm_state.copy_(conv_A_out * dA + rearrange(conv_B_out, "b d -> b d 1") * dB)

        # perform C convolution on current ssm hidden state here
        conv_C_out = self.gconv_C(ssm_state, edge_index)   # (batch_size*V, d_state)
        y = torch.einsum("bdn,bn->bd", conv_C_out.to(dtype), C)  # (B*V, d_inner)
        y = y + self.D.to(dtype) * x
        # y = y * self.act(z)  # (B*V, d_inner)
        # else:
        #     y = selective_state_update(
        #         ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
        #     )

        y = self.norm(y, z)
        out = self.out_proj(y)  # (B*V, d_model)
        return out.unsqueeze(1), conv_state, ssm_state  # (B*V, 1, d_model), (B*V, D, W), (B*V, D, N)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, num_vertices=1, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size * num_vertices, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size * num_vertices, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, num_vertices=1, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size * num_vertices,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size * num_vertices,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
