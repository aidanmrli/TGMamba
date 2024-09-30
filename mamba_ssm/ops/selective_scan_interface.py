# Copyright (c) 2023, Tri Dao, Albert Gu.

import torch
import torch.nn.functional as F
import torch_geometric
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn
    import causal_conv1d_cuda
except ImportError:
    causal_conv1d_fn = None
    causal_conv1d_cuda = None

# import selective_scan_cuda

def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                       gconv_A=None, gconv_B=None, gconv_C=None,
                       edge_index=None, edge_weight=None, num_vertices=19, act=None, time_varying_attention=False, return_last_state=False):
    """
    Implements the selective scan operation for the Mamba SSM.
    
    Args:
    u: Input tensor of shape r(B*V, D, L)
    delta: Delta values of shape r(B*V, D, L)
    A: State transition matrix of shape r(D, N) or c(D,N) (real or complex)
    B: Input projection matrix, shape c(D N) or r(B*V, N, L) or r(B*V, N, 2L) or r(B*V, G N L) or (B*V, G N L)
    C: Output projection matrix, shape c(D N) or r(B*V, N, L) or r(B*V, N, 2L) or r(B*V, G N L) or (B*V, G N L)
    D: Skip connection parameter of shape r(D)
    z: Gating tensor of shape r(B*V, D, L)
    delta_bias: Bias for delta values of shape r(D), fp32
    delta_softplus: Boolean, whether to apply softplus to delta
    return_last_state: Boolean, whether to return the last state

    adj_mat is (batch*seq_len, num_nodes, num_nodes) -> edge_index, edge_weight
    edge_index: Edge index tensor of shape (2, batch_size * num_edges, seq_len) 
    edge_weight: Edge weight tensor of shape (batch_size * num_edges, seq_len)

    Returns:
    out: Output tensor of shape r(B, D, L)
    last_state (optional): Last state tensor of shape r(B D dstate) or c(B D dstate)
    """
    # if time_varying_attention:
    #     edge_mask = (edge_index[0] != -1) & (edge_index[1] != -1)  # Assume -1 was used for padding
    #     assert edge_mask.shape == edge_weight.shape, "Edge mask and edge weight shape mismatch"
    #     edge_index = edge_index * edge_mask.unsqueeze(0)
    #     edge_weight = edge_weight * edge_mask
    #     assert edge_index.shape[0] == 2, "Edge index should have shape (2, num_edges)"
    dtype_in = u.dtype
    u = u.float()
    batch, input_dim, seqlen = u.shape
    batch = batch // num_vertices

    delta = delta.float()
    assert not torch.isnan(delta).any(), "NaN in computed delta"
    
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    delta = delta.view(-1, num_vertices, input_dim, seqlen)  # (B, V, D, L)
    if z is not None:
        z = z.view(-1, num_vertices, input_dim, seqlen)
    dim, dstate = A.shape[0], A.shape[1]    # (D, N)
    assert not torch.isnan(delta).any(), "NaN in computed delta after softplus"
    # Check if B and C are input-dependent (variable) or fixed
    is_variable_B = B.dim() >= 3
    is_variable_C = C.dim() >= 3

    # Handle complex numbers for A, B, and C
    if A.is_complex():
        if is_variable_B:
            B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
        if is_variable_C:
            C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
    else:
        B = B.float()
        C = C.float()

    # Initialize state tensor x and output tensor y
    x = A.new_zeros((batch, num_vertices, dim, dstate))     # (B, V, D, N)
    ys = []

    # Precompute some values for efficiency. set max to 10 to avoid overflow
    deltaA = torch.exp(torch.clamp(torch.einsum('bvdl,dn->bvdln', delta, A), max=10.0))    # (B, V, D, L, N)
    
    assert not torch.isnan(u).any(), "NaN in input u"
    if gconv_B is not None:
        # perform the graph convolution on the input u
        u = u.reshape(-1, input_dim)  # (B*V*L, D)
        # collapse the seqlen dimension into the batch dimension
        # TODO: check if the reshape is being done correctly
        # is this B*V*L or B*L*V? do the edge indices and edge weight correspond to the correct batch item in u?
        # after the first gradient update
        u = gconv_B(u, edge_index.reshape(2, -1), edge_weight.reshape(-1))  # (B*V*L, D)
        u = u.view(batch, num_vertices, input_dim, seqlen)  # (B, V, D, L)
        assert not torch.isnan(u).any(), "NaN in input u after B graph convolution"
        

    # Compute deltaB * u, handling different shapes of B
    # B has shape (B*V, N, L) if variable
    assert not torch.isnan(B).any(), "NaN in computed B"
    if not is_variable_B:
        deltaB_u = torch.einsum('bvdl,dn,bvdl->bvdln', delta, B, u) # (B, V, D, L, N)
    else:
        if B.dim() == 3:
            B = B.view(batch, num_vertices, B.shape[1], B.shape[2])    # (B, V, N, L)
            deltaB_u = torch.einsum('bvdl,bvnl,bvdl->bvdln', delta, B, u)    # (B, V, D, L, N)
        else:
            B = B.view(-1, num_vertices, *B.shape[1:])
            B = repeat(B, "B V G N L -> B V (G H) N L", H=dim // B.shape[2])
            deltaB_u = torch.einsum('bvdl,bvdnl,bvdl->bvdln', delta, B, u)  # (B, V, D, L, N)
    if is_variable_C and C.dim() == 4:
        C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])  # (B*V, D, L, N)
    assert not torch.isnan(deltaB_u).any(), "NaN in precomputed deltaB_u"
    
    last_state = None
    C = C.view(-1, num_vertices, *C.shape[1:])  # (B, V, D, L, N)

    # Main selective scan loop
    for i in range(seqlen):
        # perform convolution on x before applying w_A
        if gconv_A is not None:
            # repeat edge_index .repeat(1, dstate) and edge_weight .repeat(dstate)
            x = gconv_A(rearrange(x, "b v d n -> (b v d) n", n=dstate), edge_index[:, :, i], edge_weight[:, i])  # (B*V*N, D)
            x = x.view(batch, num_vertices, input_dim, dstate)  # (B, V, D, N)
            assert not torch.isnan(x).any(), "NaN in hidden state x after A graph convolution at time step {}".format(i)

        # A_t h_t-1 + B_t u_t
        # want (B, V, D, N) + (B, V, D, N)
        # x = act(x * deltaA[:, :, :, i] + deltaB_u[:, :, :, i])
        # print("x.shape: ", x.shape)
        x = torch.addcmul(deltaB_u[:, :, :, i], x, deltaA[:, :, :, i])
        # x = x * deltaA[:, :, :, i] + deltaB_u[:, :, :, i]
        assert not torch.isnan(x).any(), "NaN in hidden state x at time step {} after ssm equation".format(i)
            
        if not is_variable_C:
            y = torch.einsum('bvdn,dn->bvd', x, C)
        else:
            if gconv_C is not None:               
                conv_C = gconv_C(rearrange(x, "b v d n -> (b v d) n", n=dstate), edge_index[:, :, i], edge_weight[:, i])  # (B*V*N, D)
                conv_C = conv_C.view(batch, num_vertices, input_dim, dstate)  # (B, V, D, N)
                assert not torch.isnan(conv_C).any(), "NaN in conv_C at time step {}".format(i)

            # C has shape (B, V, N, L) if variable
            # want y to have shape (B, V, D)
            if C.dim() == 4:
                # y = torch.einsum('bdn,bn->bd', conv_C, C[:, :, :, i])   # want y= (B, V, D)
                y = torch.einsum('bvdn,bvn->bvd', conv_C, C[:, :, :, i])   # want y= (B, V, D)
            else:   # C dim is 5
                y = torch.einsum('bdn,bdn->bd', conv_C, C[:, :, :, :, i])

            del conv_C
        if y.is_complex():
            y = y.real * 2
        assert not torch.isnan(y).any(), "NaN in output token y at time step {}".format(i)
        ys.append(y)
    
    # stack y's into a single sequence tensor
    y = torch.stack(ys, dim=3) # (batch dim L)

    # Apply skip connection if D is provided
    out = y if D is None else y + u * rearrange(D, "d -> 1 1 d 1")

    # Apply gating if z is provided
    if z is not None:
        out = out * act(z)  # act should be F.silu()
    
    # Reshape output back to (B*V, D, L)
    out = out.reshape(batch * num_vertices, dim, seqlen)
    out = out.to(dtype=dtype_in)
    if return_last_state:
        x = x.reshape(-1, dim, dstate) # (B*V, D, dstate)
        return out, x
    else:    
        return out
    

# y = SSM(A_, B_, C)(x)
def mamba_inner_ref(
    xz, conv1d_weight, conv1d_bias, x_proj_weight, delta_proj_weight,
    out_proj_weight, out_proj_bias,
    A, B=None, C=None, D=None, delta_bias=None, B_proj_bias=None,
    C_proj_bias=None, delta_softplus=True,
    gconv_A=None, gconv_B=None, gconv_C=None,
    edge_index=None, edge_weight=None, num_vertices=19
):
    """
        xz,     # (B * V, 2 * d_inner, L) 
        self.conv1d.weight,     # (d_inner, 1, d_conv)
        self.conv1d.bias,    # (d_inner)
        self.x_proj.weight,  # (dt_rank + d_state * 2, d_inner)
        self.dt_proj.weight,  # (d_inner, dt_rank)
        self.out_proj.weight,  # (d_inner, d_model)
        self.out_proj.bias,  # (d_model)
        A,  # (D, N) or (d_inner, d_state) representing D different structured N x N matrices for each of the D input_dim
        None,  # input-dependent B
        None,  # input-dependent C
        self.D.float(),  # (d_inner) ie (D)
        delta_bias=self.dt_proj.bias.float(),   # (d_inner)
        delta_softplus=True,

        pass to the selective_scan_ref function:
        u: Input tensor of shape r(B*V, D, L)
        delta: Delta values of shape r(B*V, D, L)
        A: State transition matrix of shape r(D, N) or c(D,N) (real or complex)
        B: Input projection matrix, shape c(D N) or r(B*V, N, L) or r(B*V, N, 2L) or r(B*V, G N L) or (B*V, G N L)
        C: Output projection matrix, shape c(D N) or r(B*V, N, L) or r(B*V, N, 2L) or r(B*V, G N L) or (B*V, G N L)
        D: Skip connection parameter of shape r(D)
        z: Gating tensor of shape r(B*V, D, L)
        delta_bias: Bias for delta values of shape r(D), fp32
        delta_softplus: Boolean, whether to apply softplus to delta
        return_last_state: Boolean, whether to return the last state

        Returns:
        out: Output tensor of shape r(B, D, L)
        last_state (optional): Last state tensor of shape r(B D dstate) or c(B D dstate) 
    """
    assert causal_conv1d_fn is not None, "causal_conv1d_fn is not available. Please install causal-conv1d."
    batch, dim, seqlen = xz.shape  
    batch = batch // num_vertices
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1] * (1 if not A.is_complex() else 2)

    # split input xz into main input x and gated input z
    x, z = xz.chunk(2, dim=1)   # (B * V, d_inner, L) for each of x and z

    # Apply causal_conv1d to each vertex separately
    # TODO: or replace this with a convolution over the current graph
    x = causal_conv1d_fn(x, rearrange(conv1d_weight, "d 1 w -> d w"), conv1d_bias, activation="silu")   # (B * V, d_inner, L)

    # We're being very careful here about the layout, to avoid extra transposes.
    # We want delta to have d as the slowest moving dimension
    # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.

    x_dbl = F.linear(rearrange(x, 'b d l -> (b l) d'), x_proj_weight)  # (bvl d)
    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", l=seqlen)  # (B * V, D, L)

    if B is None:  # variable B
        B = x_dbl[:, delta_rank:delta_rank + d_state]  # (bvl d)
        if B_proj_bias is not None:
            B = B + B_proj_bias.to(dtype=B.dtype)
        if not A.is_complex():
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        else:
            B = rearrange(B, "(b l) (dstate two) -> b dstate (l two)", l=seqlen, two=2).contiguous()
    if C is None:  # variable C
        C = x_dbl[:, -d_state:]  # (bvl d)
        if C_proj_bias is not None:
            C = C + C_proj_bias.to(dtype=C.dtype)
        if not A.is_complex():
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        else:
            C = rearrange(C, "(b l) (dstate two) -> b dstate (l two)", l=seqlen, two=2).contiguous()
    
    y = selective_scan_ref(x, delta, A, B, C, D, z=z, delta_bias=delta_bias, delta_softplus=True,
                          gconv_A=gconv_A, gconv_B=gconv_B, gconv_C=gconv_C,
                          edge_index=edge_index, edge_weight=edge_weight, num_vertices=num_vertices)
    return F.linear(rearrange(y, "b d l -> b l d"), out_proj_weight, out_proj_bias)
