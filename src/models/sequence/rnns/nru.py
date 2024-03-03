
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
import numpy as np

import torch
import numpy as np
from jax.tree_util import tree_flatten, tree_unflatten
from typing import overload, Callable, Iterable, List, TypeVar, Any, Literal, Union, Sequence, Tuple, Optional
from functools import partial
from einops import rearrange, repeat
import math

from omegaconf import OmegaConf, DictConfig

from src.models.nn import LinearActivation, Activation, DropoutNd
from src.models.sequence.base import SequenceModule
import src.utils as utils
import src.utils.registry as registry

import src.utils.train
log = src.utils.train.get_logger(__name__)

"""
Jax-Pytorch ported functions, mostly interfaces are kept the same but unsupported features are removed:
* Jax-Keyed RNGs are sampled from global RNG
* Canonical/Named shapes/dtypes/etc are now regular shapes,dtypes
"""

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...

@overload
def safe_map(f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> List[T]: ...

@overload
def safe_map(f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> List[T]: ...

@overload
def safe_map(f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], *args) -> List[T]: ...

def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

def slice_along_axis(start, end, stride=None, axis=0):
    return (slice(None),) * axis + (slice(start, end, stride),)

# Pytorch impl. of jax.lax.associative_scan
def associative_scan(operator, elems, axis=0, reverse=False):
    if not callable(operator):
        raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    def combine(a_flat, b_flat):
        # Lower `fn` to operate on flattened sequences of elems.
        a = tree_unflatten(tree, a_flat)
        b = tree_unflatten(tree, b_flat)
        c = operator(a, b)
        c_flat, _ = tree_flatten(c)
        return c_flat

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[axis]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.
        reduced_elems = combine(
          [elem[slice_along_axis(0, -1, stride=2, axis=axis)] for elem in elems],
          [elem[slice_along_axis(1, None, stride=2, axis=axis)] for elem in elems])

        # Recursively compute scan for partially reduced tensors.
        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = combine(
                [e[slice_along_axis(0, -1, axis=axis)] for e in odd_elems],
                [e[slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])
        else:
            even_elems = combine(
                odd_elems,
                [e[slice_along_axis(2, None, stride=2, axis=axis)] for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.
        even_elems = [
          torch.cat([elem[slice_along_axis(0, 1, axis=axis)], result], dim=axis)
          if result.shape.numel() > 0 and elem.shape[axis] > 0 else
          result if result.shape.numel() > 0 else
          elem[slice_along_axis(0, 1, axis=axis)]  # Jax allows/ignores concat with 0-dim, Pytorch does not
          for (elem, result) in zip(elems, even_elems)]

        return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

    scans = _scan(elems_flat)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(tree, scans)

def _interleave(a, b, axis):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    if b_trunc := (a.shape[axis] == b.shape[axis] + 1):
        pad = [0, 0] * b.ndim
        pad[(b.ndim-axis-1)*2+1] = 1 # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis+1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis+1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = interleaved[slice_along_axis(0, b.shape[axis]+a.shape[axis]-1, axis=axis)]
    return interleaved

def binary_operator_diag_torch(element_i, element_j):
    # Binary operator for parallel scan of linear recurrence.
    a_i, bu_i = element_i
    a_j, bu_j = element_j
    return a_j * a_i, a_j * bu_i + bu_j

nonlin = lambda x: getattr(F, x) if x != 'identity' else lambda x: x
    
def bin_op_cumsum(e_i, e_j):
    return e_i + e_j

class NRU_v1(nn.Module):
    def __init__(
        self,
        d_model,
        d_memory,
        # Initial S4 Studd
        bottleneck=None,
        activation='gelu',
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        #
        alpha_v_bias=True,
        # TODO: For a more granular control over biases, uncomment these
        # alpha_plus_bias=True, v_plus_bias=True,
        # alpha_minus_bias=True, v_minus_bias=True,
        alpha_plus_nonlin='identity', 
        v_plus_nonlin='identity',
        alpha_minus_nonlin='identity', 
        v_minus_nonlin='identity',
        num_heads=1, 
        rank=1, 
        norm_p=5,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        gated=False,
        gate_loc='input',
        gate_activation='gelu',
        gate_expansion=1,
        layer='s4',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.d_out = d_model
        self.d_memory = d_memory
        
        # self.gated = gated
        # if self.gated:
        #     # Either put the gating at the input or after the memory augmentation
        #     assert gate_loc in ["input", "memory"]
        #     self.gate_loc = gate_loc
        
        self.num_heads = num_heads
        self.norm_p = norm_p
        
        self.alpha_plus_nonlin = nonlin(alpha_plus_nonlin)
        self.v_plus_nonlin = nonlin(v_plus_nonlin)
        self.alpha_minus_nonlin = nonlin(alpha_minus_nonlin)
        self.v_minus_nonlin = nonlin(v_minus_nonlin)
        self.rank = rank
        
        sqrt_dim = int(math.sqrt(num_heads * self.d_memory))
        self.sqrt_dim = sqrt_dim
        self.alpha_v = nn.Linear(self.d_model, 4 * self.d_memory, bias=alpha_v_bias)
        
        # Gating
        # if self.gated:
        #     inp_int = self.d_model + self.d_memory if gate_loc == "memory" else self.d_model
        #     out_int = inp_int * gate_expansion
        #     self.gate_int1 = nn.Linear(inp_int, out_int) if self.gated else nn.Identity()
            
        #     self.gate_activation = Activation(gate_activation) if self.gated else nn.Identity()
        #     self.gate_int2 = nn.Linear(inp_int, out_int) if self.gated else nn.Identity()
            
        #     self.gate_out = nn.Linear(out_int, inp_int) if self.gated else nn.Identity()
        
        # if self.gated and gate_loc == "input":
        #     ssm_dim = (self.d_model * gate_expansion + self.d_memory) * gate_expansion 
        # elif self.gated:
        #     ssm_dim = (self.d_model + self.d_memory) * gate_expansion
        # else:
            
        ssm_dim = self.d_model + self.d_memory
        
        layer_cfg = layer_args.copy()['layer_args']
        layer_cfg['_name_'] = layer
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        
        self.ssm = utils.instantiate(registry.layer, layer_cfg, ssm_dim)
        
        self.gated_out = nn.Linear(ssm_dim, self.d_model)
        
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # self.alpha_plus = nn.Linear(d_model, d_memory, bias=alpha_plus_bias)
        # self.v_plus = nn.Linear(d_model, d_memory, bias=v_plus_bias)
        # self.alpha_minus = nn.Linear(d_model, d_memory, bias=alpha_minus_bias)
        # self.v_minus = nn.Linear(d_model, d_memory, bias=v_minus_bias)
        
        # S4 stuff

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
                
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.d_out,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )

    def forward_parallel(self, input_sequence, state=None, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        """
        d = self.d_memory
        x = input_sequence

        if self.gate is not None:
            v = self.input_gate(x)
            
        if self.bottleneck is not None:
            x = self.input_linear(x)
        
        # if self.gated and self.gate_loc == "input":
        #     x = self.gate_int2(x)
        
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        alpha_v = self.alpha_v(x[:, :-1])
        m_init = torch.zeros((b, 1, d), dtype=x.dtype, device=x.device)
        # ms = x[:, :-1, :d]
        # ms_inp = (F.normalize(self.alpha_plus_nonlin(alpha_v[..., 0]) * self.v_plus_nonlin(alpha_v[..., 2:(d+2)]), p=self.norm_p, dim=2, eps=1e-12)
        #         - F.normalize(self.alpha_minus_nonlin(alpha_v[..., 1]) * self.v_minus_nonlin(alpha_v[..., (d+2):]), p=self.norm_p, dim=2, eps=1e-12))
        ms_inp = (self.alpha_plus_nonlin(alpha_v[..., :d]) * self.v_plus_nonlin(alpha_v[..., d:2*d])
                - self.alpha_minus_nonlin(alpha_v[..., 2*d:3*d]) * self.v_minus_nonlin(alpha_v[..., 3*d:4*d]))
        
        # # TODO: if you need a more granular control over biases,
        # # uncomment these
        # # ms_inp = self.alpha_plus(x) * self.v_plus(x) - self.alpha_minus(x) * self.v_minus(x)

        # # TODO: I noticed some error accumulation of approx 2e-7 per step
        # # (i.e. the abs error between par scan cumsum and linear cumsum will be 0.1 after 512k steps)
        # # though it does not contribute a lot to the final ssm output it can be completely eliminated
        # # by casting the input to float64 and then back to dtype after par scan
        # # ms_inp = ms_inp.double()
        # ms = ms_inp
        ms = associative_scan(bin_op_cumsum, ms_inp, axis=1)
        ms = torch.cat([m_init, ms], dim=1)
        # ms = torch.cat([m_init, ms], dim=1)
        # ms = ms.float()
        # ssm_inp = torch.cat([x, ms], axis=2) # size is [batch, length, in_size + m_size]
        ssm_inp = torch.cat([x, ms], axis=2)
        # ssm_inp = x
        # if self.gated and self.gate_loc == "memory":
        #     ssm_inp = self.gate_int2(ssm_inp)
        
        ssm_out, state = self.ssm.forward(ssm_inp)
        # if self.gated:
        #     if self.gate_loc == "input":
        #         ssm_out = self.gated_out(ssm_out)
        #         ssm_out = self.gate_out(self.activation(self.gate_int1(input_sequence)) * ssm_out)
        #     elif self.gate == "memory":
        #         ssm_out = self.gate_out(self.activation(self.gate_int1(ssm_inp)) * ssm_out)
        #         ssm_out = self.gated_out(ssm_out)
        # else:
        ssm_out = self.gated_out(ssm_out)

        y = ssm_out

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        
        return y, state
    
    def forward(self, x, state=None, **kwargs):
        return self.forward_parallel(x, state, **kwargs)

    def forward_sequential(self, input_sequence, state, m_state, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        state.shape == [batch, d_state]
        m_state.shape == [batch, d_memory]
        """
        d = self.d_memory
        x = input_sequence

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        
        if self.gated and self.gate_loc == "input":
            x = self.gate_int2(x)
            
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        
        # input for M
        alpha_v = self.alpha_v(x[:, :-1])
        ms_inp = (self.alpha_plus_nonlin(alpha_v[..., :d]) * self.v_plus_nonlin(alpha_v[..., d:2*d])
                - self.alpha_minus_nonlin(alpha_v[..., 2*d:3*d]) * self.v_minus_nonlin(alpha_v[..., 3*d:4*d]))
        ms_inp = torch.cat([m_state.unsqueeze(1), ms_inp], dim=1)
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # ms_inp = self.alpha_plus(x) * self.v_plus(x) - self.alpha_minus(x) * self.v_minus(x)
        ms = torch.zeros(ms_inp.shape, dtype=ms_inp.dtype, device=ms_inp.device)
        for j in range(1, ms_inp.shape[1]):
            ms[:, j] = ms[:, j - 1] + ms_inp[:, j]
            
        ssm_inp = torch.cat([x, ms], axis=2) # size is [batch, length, in_size + m_size]
        
        ssm_out, state = self.ssm.step(ssm_inp, state)
        
        # Reshape to NRU shape
        if self.gated and self.gate_loc == "memory":
            ssm_inp = self.gate_int2(ssm_inp)
            
        ssm_out, state = self.ssm.step(ssm_inp, state)
        
        if self.gated:
            if self.gate_loc == "input":
                ssm_out = self.output_gate(ssm_out)
                ssm_out = self.gated_out(self.activation(self.gate_int1(input_sequence)) * ssm_out)
            elif self.gated == "memory":
                ssm_out = self.gate_out(self.activation(self.gate_int1(ssm_inp)) * ssm_out)
                ssm_out = self.gated_out(ssm_out)
        else:
            ssm_out = self.gated_out(ssm_out)

        y = ssm_out

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        
        return y, state, ms[:, -1]
    
    def step(self, x, state=None, m_state=None, **kwargs):
        return self.forward_sequential(x.unsqueeze(1), state, m_state, **kwargs)
    
    @property
    def d_state(self):
        """Size after converting state to a single tensor."""
        return self.ssm.d_state

    @property
    def d_output(self):
        """Size of output."""
        return self.d_out

    @property
    def state_to_tensor(self):
        """Convert state into a single tensor output."""
        return lambda state: state


class NRU(nn.Module):
    def __init__(
        self,
        d_model,
        d_memory,
        # Initial S4 Studd
        bottleneck=None,
        activation='gelu',
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        # NRU params
        alpha_v_bias=True,
        # TODO: For a more granular control over biases, uncomment these
        # alpha_plus_bias=True, v_plus_bias=True,
        # alpha_minus_bias=True, v_minus_bias=True,
        alpha_plus_nonlin='identity', 
        v_plus_nonlin='identity',
        alpha_minus_nonlin='identity', 
        v_minus_nonlin='identity',
        num_heads=1, 
        rank=1, 
        norm_p=5,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        gated=False,
        gate_loc='input',
        gate_activation='gelu',
        gate_expansion=1,
        layer='s4',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()
        
        self.d_model = d_model
        
        self.d_out = d_model
        self.d_memory = d_memory
        
        # self.gated = gated
        # if self.gated:
        #     # Either put the gating at the input or after the memory augmentation
        #     assert gate_loc in ["input", "memory"]
        #     self.gate_loc = gate_loc
        
        self.num_heads = num_heads
        self.norm_p = norm_p
        
        self.alpha_plus_nonlin = nonlin(alpha_plus_nonlin)
        self.v_plus_nonlin = nonlin(v_plus_nonlin)
        self.alpha_minus_nonlin = nonlin(alpha_minus_nonlin)
        self.v_minus_nonlin = nonlin(v_minus_nonlin)
        
        self.rank = rank
        self.sqrt_dim = int(math.sqrt(num_heads * self.d_memory))
        
        self.alpha_v = nn.Linear(self.d_model, 2 * self.num_heads + 4 * self.sqrt_dim * self.rank, bias=alpha_v_bias)
            
        self.ssm_dim = self.d_model + self.d_memory
        
        layer_cfg = layer_args.copy()['layer_args']
        layer_cfg['_name_'] = layer
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        
        self.ssm = utils.instantiate(registry.layer, layer_cfg, self.ssm_dim)
        
        self.gated_out = nn.Linear(self.ssm_dim, self.d_model)
        
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # self.alpha_plus = nn.Linear(d_model, d_memory, bias=alpha_plus_bias)
        # self.v_plus = nn.Linear(d_model, d_memory, bias=v_plus_bias)
        # self.alpha_minus = nn.Linear(d_model, d_memory, bias=alpha_minus_bias)
        # self.v_minus = nn.Linear(d_model, d_memory, bias=v_minus_bias)
        
        # S4 stuff

        self.gate = gate
        self.bottleneck = bottleneck

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
                
            self.output_gate = LinearActivation(
                self.d_model * gate,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.d_out,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )

    def forward_parallel(self, input_sequence, state=None, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        """
        d = self.d_memory
        x = input_sequence

        if self.gate is not None:
            v = self.input_gate(x)
            
        if self.bottleneck is not None:
            x = self.input_linear(x)
        
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        
        # Initial Memory
        m_init = torch.zeros((b, 1, d), dtype=x.dtype, device=x.device)
        
        # Construct linear memory
        
        # Amounts for writing/erasing
        alpha_v = self.alpha_v(x)
        alpha, beta = alpha_v[..., :self.num_heads], alpha_v[..., self.num_heads:self.num_heads * 2]
        
        # Get writing/erasing vectors
        v = alpha_v[..., 2 * self.num_heads:].view(b, l, 4, self.rank, self.sqrt_dim)
        v_alpha1, v_alpha2, v_beta1, v_beta2 = \
        v[:, :, 0].contiguous(), \
        v[:, :, 1].contiguous(), \
        v[:, :, 2].contiguous(), \
        v[:, :, 3].contiguous()
        
        # Build directions and normalize
        v_alpha = torch.einsum('btkd,btkp->btdp', v_alpha1, v_alpha2).view(b, l, self.num_heads, self.d_memory)
        v_alpha = F.normalize(v_alpha, p=self.norm_p, dim=3, eps=1e-12)
        
        v_beta = torch.einsum('btkd,btkp->btdp', v_beta1, v_beta2).view(b, l, self.num_heads, self.d_memory)
        v_beta = F.normalize(v_beta, p=self.norm_p, dim=3, eps=1e-12)
        
        # Sum over directions
        alpha = torch.einsum('btk,btkd->btd', alpha, v_alpha)
        beta = torch.einsum('btk,btkd->btd', beta, v_beta)
        
        # Final memory direction
        ms_inp = alpha - beta
    
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # ms_inp = self.alpha_plus(x) * self.v_plus(x) - self.alpha_minus(x) * self.v_minus(x)

        # TODO: I noticed some error accumulation of approx 2e-7 per step
        # (i.e. the abs error between par scan cumsum and linear cumsum will be 0.1 after 512k steps)
        # though it does not contribute a lot to the final ssm output it can be completely eliminated
        # by casting the input to float64 and then back to dtype after par scan
        # ms_inp = ms_inp.double()
        
        ms = ms_inp # Test memory as just a transformed input
        # ms = associative_scan(bin_op_cumsum, ms_inp, axis=1)
        # ms = torch.cumsum(ms_inp, dim=1)
        # assert torch.allclose(ms_scan, ms) # Seems that associate scan using the cumulative sum operator is not the same as a torch cumulative sum.
        ms = torch.cat([m_init, ms[:, :-1]], dim=1)
        
        ssm_inp = torch.cat([x, ms], axis=2)
        ssm_out, state = self.ssm.forward(ssm_inp)
        ssm_out = self.gated_out(ssm_out)

        y = ssm_out

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        
        return y, state
    
    def forward(self, x, state=None, **kwargs):
        return self.forward_parallel(x, state, **kwargs)

    def forward_sequential(self, input_sequence, state, m_state, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        state.shape == [batch, d_state]
        m_state.shape == [batch, d_memory]
        """
        d = self.d_memory
        x = input_sequence

        if self.gate is not None:
            v = self.input_gate(x)
            
        if self.bottleneck is not None:
            x = self.input_linear(x)
            
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        
        if m_state is None:
            m_state_old = torch.zeros((b, l, d), dtype=x.dtype, device=x.device)
        else:
            m_state_old = m_state
        
        for j in range(1, x.shape[1]):
            alpha_v = self.alpha_v(x[:, j-1]) # [b, in_dim]
            alpha, beta = alpha_v[..., :self.num_heads], alpha_v[..., self.num_heads:self.num_heads * 2]
            
            v = alpha_v[..., 2 * self.num_heads:].view(b, 4, self.rank, self.sqrt_dim)
            
            v_alpha1, v_alpha2, v_beta1, v_beta2 = \
                v[:, 0].contiguous(), \
                v[:, 1].contiguous(), \
                v[:, 2].contiguous(), \
                v[:, 3].contiguous()
                
            v_alpha = torch.einsum('bkd,bkp->bdp', v_alpha1, v_alpha2).view(b, self.num_heads, self.d_memory)
            v_alpha = F.normalize(v_alpha, p=self.norm_p, dim=2, eps=1e-12)
            
            v_beta = torch.einsum('bkd,bkp->bdp', v_beta1, v_beta2).view(b, self.num_heads, self.d_memory)
            v_beta = F.normalize(v_beta, p=self.norm_p, dim=2, eps=1e-12)
            
            alpha = torch.einsum('bk,bkd->bd', alpha, v_alpha)
            beta = torch.einsum('bk,bkd->bd', beta, v_beta)
            
            m_state = m_state_old + alpha - beta
            
        ssm_inp = torch.cat([x.sequeeze(), m_state], axis=2) # size is [batch, length, in_size + m_size]
        
        ssm_out, state = self.ssm.step(ssm_inp, state)

        y = ssm_out

        y = self.activation(y)

        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        
        return y, state, m_state
    
    def step(self, x, state=None, m_state=None, **kwargs):
        return self.forward_sequential(x.unsqueeze(1), state, m_state, **kwargs)
    
    @property
    def d_state(self):
        """Size after converting state to a single tensor."""
        return self.d_hidden

    @property
    def d_output(self):
        """Size of output."""
        return self.d_out

    @property
    def state_to_tensor(self):
        """Convert state into a single tensor output."""
        return lambda state: state
