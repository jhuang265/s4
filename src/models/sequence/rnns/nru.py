
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

class LRU(SequenceModule):
    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int,
        activation='gelu',
        mult_act=None,
        # LRU Parameters
        rnn_parameterization: str ='lru',
        rmin: int = 0, 
        rmax: int = 1,
        max_phase_factor: float = 2.0,
        # Default
        dropout: float = 0.0,
        tie_dropout=False,
        transposed: bool = False,
        **kwargs
    ):
        assert rnn_parameterization in ['real', 'lru']
        assert transposed == False
        
        super().__init__()
        self.d_input = d_input
        self.d_out = d_output
        self.d_model = d_model
        
        self.transposed = transposed
        
        self.D = nn.Parameter(torch.randn([self.d_input])/math.sqrt(self.d_input))
        self.diag_D = True
            
        self.rnn_parameterization = rnn_parameterization
        
        if rnn_parameterization == 'lru':
            u1 = torch.rand(self.d_state)
            u2 = torch.rand(self.d_state)
            
            self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin**2)))
            self.theta_log = nn.Parameter(torch.log(np.pi * max_phase_factor * u2))
            
            Lambda_mod = torch.exp(-torch.exp(self.nu_log))
            self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))
            
            B_re = torch.randn([self.d_state, self.d_input]) / math.sqrt(2 * self.d_input)
            B_im = torch.randn([self.d_state, self.d_input]) / math.sqrt(2 * self.d_input)
            self.B = nn.Parameter(torch.complex(B_re, B_im))
            
            C_re = torch.randn([self.d_output, self.d_state]) / math.sqrt(self.d_state)
            C_im = torch.randn([self.d_output, self.d_state]) / math.sqrt(self.d_state)
            self.C = nn.Parameter(torch.complex(C_re, C_im))
            # self.state = torch.complex(torch.zeros(d_state),torch.zeros(d_state))
            
        elif rnn_parameterization == 'real':
            #self.Lambda = nn.Parameter(torch.exp(torch.randn([self.d_state])))
            self.Lambda = nn.Parameter(torch.randn([self.d_state]))
            self.B = nn.Parameter(torch.randn([self.d_state, d_input]) / math.sqrt(self.d_input))
            self.C = nn.Parameter(torch.randn([self.d_output, self.d_state]) / math.sqrt(self.d_state))
            

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)
        
        self.mult_act = mult_act
        if self.mult_act in ["full_glu"]:
            self.out1 = nn.Linear(self.d_output, self.d_output)
            self.out2 = nn.Linear(self.d_output, self.d_output)
        elif self.mult_act in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Linear(self.d_output, self.d_output)
        
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

    def forward_parallel(self, input_sequence, state=None, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        """
        
        # Change from [batch, length, input_size] to [batch, input_size, length]. Technically we can't do this now so it will always be false
        if self.transposed:
            input_sequence = rearrange(input_sequence, 'b d ... -> b ... d')
        
        # Set up exp(A) and B for the recurrence (as well as our input)
        if self.rnn_parameterization == 'lru':
            Lambda = torch.exp(-self.nu_log.exp() + 1j * self.theta_log.exp())
            B_norm = self.B * self.gamma_log.exp().unsqueeze(1)
            B_input =  input_sequence + 0j
        
        elif self.rnn_parameterization == 'real':
            Lambda = torch.exp(-torch.exp(self.Lambda))
            B_norm = self.B
            B_input =  input_sequence

        # Recurrence through scanning operation
        Lambda_elements = Lambda.repeat(input_sequence.shape[1], 1)
        # Bu_elements = torch.einsum('btd,nd->btn', input_sequence + 0j, B_norm) # [batch, len, dim] @ [dim, outdim] = [batch, len, outdim]
        Bu_elements = torch.einsum('btd,nd->btn', B_input, B_norm)
        elements = (Lambda_elements.unsqueeze(0).to(Bu_elements.dtype), Bu_elements)
        _, inner_states = associative_scan(binary_operator_diag_torch, elements, axis=1)
        
        # Get the proper output component
        if self.rnn_parameterization == 'lru':
            y = torch.einsum('btd,nd->btn', inner_states, self.C).real
        
        elif self.rnn_parameterization == 'real':
            y = torch.einsum('btd,nd->btn', inner_states, self.C)
        
        # Add the direct connection from input to output
        if self.diag_D:
            y += self.D * input_sequence
        else:
            y += torch.einsum('btd,nd->btn', input_sequence, self.D)
        
        # Intermediate activation (if any)    
        y = self.activation(y)
        
        # Additional activation if specified
        if self.mult_act in ["full_glu"]:
            y = self.drop(y)
            y = self.out1(y) * F.sigmoid(self.out2(y))
        elif self.mult_act in ["half_glu1"]:
            y = self.drop(y)
            y = y * F.sigmoid(self.out2(y))
        elif self.mult_act in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.drop(y)
            y = y * F.sigmoid(self.out2(x1))
        
        # Rearrange to proper shape
        if self.transposed: 
            y = rearrange(y, 'b d ... -> b ... d')
        
        return y, None

    def forward_sequential(self, input_sequence, state, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        state.shape == [batch, d_state]
        """
        if self.rnn_parameterization == 'lru':
            Lambda = torch.exp(-self.nu_log.exp() + 1j * self.theta_log.exp())
            B_norm = self.B * self.gamma_log.exp().unsqueeze(1)
            B_input =  input_sequence + 0j
            
        elif self.rnn_parameterization == 'real':
            Lambda = torch.exp(-torch.exp(self.Lambda))
            B_norm = self.B
            B_input =  input_sequence

        output = torch.empty(input_sequence.shape[:-1] + (self.out_features,), dtype=input_sequence.dtype, device=self.B.device)
        C = self.C.t()
        D = self.D.t()
        B = B_norm.t()
        x = state
        Us = B_input.transpose(0, 1).contiguous()
        # Us.shape == [length, batch, in_features]
        
        for j in range(input_sequence.shape[1]):
            u = Us[j]
            x = Lambda * x + (u @ B)
            
            if self.diag_D:
                output[:, j] = (x @ C).real + D * u
                
            else:
                # print('x', x.shape, 'C', C.shape, 'D', D.shape, 'u', u.shape)
                # print('Cx', (x @ C).real.shape, 'Du', (u.real @ D).shape)
                # print('y', output[:, j].shape)
                output[:, j] = (x @ C).real + u.real @ D
      
        return output, x
    
    def forward(self, inputs, state=None, **kwargs):
        return self.forward_parallel(inputs, state, **kwargs)
    
    def step(self, x, state=None, **kwargs):
        return self.forward_sequential(x.unsqueeze(1), state, **kwargs)

    def default_state(self, *batch_shape, device=None):
        return torch.zeros(
            *batch_shape, self.d_state,
            device=device,
            requires_grad=False,
        )

    @property
    def d_state(self):
        """Size after converting state to a single tensor."""
        return self.d_model

    @property
    def d_output(self):
        """Size of output."""
        return self.d_out

    @property
    def state_to_tensor(self):
        """Convert state into a single tensor output."""
        return lambda state: state

nonlin = lambda x: getattr(F, x) if x != 'identity' else lambda x: x
    
def bin_op_cumsum(e_i, e_j):
    return e_i + e_j

class NRU(nn.Module):
    def __init__(
        self,
        d_input,
        d_hidden,
        d_memory,
        alpha_v_bias=True,
        # TODO: For a more granular control over biases, uncomment these
        # alpha_plus_bias=True, v_plus_bias=True,
        # alpha_minus_bias=True, v_minus_bias=True,
        alpha_plus_nonlin='sigmoid', 
        v_plus_nonlin='sigmoid',
        alpha_minus_nonlin='tanh', 
        v_minus_nonlin='tanh',
        num_heads=4, rank=1, norm_p=5,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        layer='lru',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()
        self.d_input = d_input
        self.d_out = d_input
        self.d_state = d_hidden
        self.d_memory = d_memory
        
        self.num_heads = num_heads
        self.norm_p = norm_p
        
        self.alpha_plus_nonlin = nonlin(alpha_plus_nonlin)
        self.v_plus_nonlin = nonlin(v_plus_nonlin)
        self.alpha_minus_nonlin = nonlin(alpha_minus_nonlin)
        self.v_minus_nonlin = nonlin(v_minus_nonlin)self.rank = rank
        
        sqrt_dim = int(math.sqrt(num_heads * self.d_memory))
        self.sqrt_dim = sqrt_dim
        self.alpha_v = nn.Linear(self.d_input, 2 * num_heads + 4 * sqrt_dim * rank, bias=alpha_v_bias)
        
        # self.alpha_v = nn.Linear(self.d_input, 4 * self.d_memory, bias=alpha_v_bias)
        self.B = nn.Linear(self.d_input, self.d_state, bias=False)
        
        layer_cfg = layer_args.copy()
        layer_cfg['_name_'] = layer
        layer_cfg['d_output'] = d_input
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        self.ssm = utils.instantiate(registry.layer, layer_cfg, self.d_input + self.d_memory)
        
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # self.alpha_plus = nn.Linear(d_model, m_features, bias=alpha_plus_bias)
        # self.v_plus = nn.Linear(d_model, m_features, bias=v_plus_bias)
        # self.alpha_minus = nn.Linear(d_model, m_features, bias=alpha_minus_bias)
        # self.v_minus = nn.Linear(d_model, m_features, bias=v_minus_bias)

    def forward_parallel(self, input_sequence, state=None, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        """
        d = self.d_memory
        x = input_sequence
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        # input for M
        alpha_v = self.alpha_v(x)
        # print(alpha_v.shape)
        alpha, beta = alpha_v[..., :self.num_heads], alpha_v[..., self.num_heads:self.num_heads * 2]
        # print(alpha_v[..., 2 * self.num_heads:].shape)
        v = alpha_v[..., 2 * self.num_heads:].view(b, l, 4, self.rank, self.sqrt_dim)
        # print(v.shape)
        v_alpha1, v_alpha2, v_beta1, v_beta2 = \
        v[:, :, 0].contiguous(), \
        v[:, :, 1].contiguous(), \
        v[:, :, 2].contiguous(), \
        v[:, :, 3].contiguous()
        v_alpha = torch.einsum('btkd,btkp->btdp', v_alpha1, v_alpha2).view(b,l, self.num_heads, self.m_features)
        # print(v_alpha.shape)
        v_alpha = F.normalize(v_alpha, p=self.norm_p, dim=3, eps=1e-12)
        # v_alpha = v_alpha / v_alpha.max(3, keepdim=True)[0]
        v_beta = torch.einsum('btkd,btkp->btdp', v_beta1, v_beta2).view(b,l, self.num_heads, self.m_features)
        # print(v_beta.shape)
        v_beta = F.normalize(v_beta, p=self.norm_p, dim=3, eps=1e-12)
        # v_beta = v_beta / v_beta.max(3, keepdim=True)[0]
        alpha = torch.einsum('btk,btkd->btd', alpha, v_alpha)
        beta = torch.einsum('btk,btkd->btd', beta, v_beta)
        # print(alpha.shape, beta.shape)
        ms_inp = alpha - beta
        m_init = torch.zeros((b, 1, d), dtype=x.dtype, device=x.device)
        
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # ms_inp = self.alpha_plus(x) * self.v_plus(x) - self.alpha_minus(x) * self.v_minus(x)

        # TODO: I noticed some error accumulation of approx 2e-7 per step
        # (i.e. the abs error between par scan cumsum and linear cumsum will be 0.1 after 512k steps)
        # though it does not contribute a lot to the final ssm output it can be completely eliminated
        # by casting the input to float64 and then back to dtype after par scan
        # ms_inp = ms_inp.double()
        ms = associative_scan(bin_op_cumsum, ms_inp, axis=1)
        ms = torch.cat([m_init, ms], dim=1)
        # ms = ms.float()
        ssm_inp = torch.cat([x.to(ms.dtype), ms], axis=2) # size is [batch, length, in_size + m_size]
        return self.ssm.forward_parallel(ssm_inp)
    
    def forward(self, x, state=None, **kwargs):
        return self.forward_parallel(x, state, **kwargs)

    def forward_sequential(self, input_sequence, state, m_state, **kwargs):
        """
        input_sequence.shape == [batch, length, d_model]
        state.shape == [batch, d_state]
        m_state.shape == [batch, m_features]
        """
        d = self.d_memory
        x = input_sequence
        b, l, inp_dim = x.shape[0], x.shape[1], x.shape[2]
        # input for M

        # palpha_v = self.alpha_v(x)
        # # print(alpha_v.shape)
        # palpha, pbeta = palpha_v[..., :self.num_heads], palpha_v[..., self.num_heads:self.num_heads * 2]
        # # print(alpha_v[..., 2 * self.num_heads:].shape)
        # pv = palpha_v[..., 2 * self.num_heads:].view(b, l, 4, self.rank, self.sqrt_dim)
        # # print(v.shape)
        # pv_alpha1, pv_alpha2, pv_beta1, pv_beta2 = \
        #   pv[:, :, 0].contiguous(), \
        #   pv[:, :, 1].contiguous(), \
        #   pv[:, :, 2].contiguous(), \
        #   pv[:, :, 3].contiguous()
        # pv_alpha = torch.einsum('btkd,btkp->btdp', pv_alpha1, pv_alpha2).view(b,l, self.num_heads, self.m_features)
        # # print(v_alpha.shape)
        # pv_alpha = F.normalize(pv_alpha, p=self.norm_p, dim=3, eps=1e-12)
        # pv_beta = torch.einsum('btkd,btkp->btdp', pv_beta1, pv_beta2).view(b,l, self.num_heads, self.m_features)
        # # print(v_beta.shape)
        # v_beta = F.normalize(v_beta, p=self.norm_p, dim=3, eps=1e-12)
        # alpha = torch.einsum('btk,btkd->btd', alpha, v_alpha)
        # beta = torch.einsum('btk,btkd->btd', beta, v_beta)
        # # print(alpha.shape, beta.shape)
        # ms_inp = (alpha - beta)

        # ms_inp = torch.cat([m_state.unsqueeze(1), ms_inp], dim=1)
        # TODO: if you need a more granular control over biases,
        # uncomment these
        # ms_inp = self.alpha_plus(x) * self.v_plus(x) - self.alpha_minus(x) * self.v_minus(x)
        ms = torch.zeros((b, l, d), dtype=x.dtype, device=x.device)
        ms[:, 0] = m_state
        for j in range(1, x.shape[1]):
            alpha_v = self.alpha_v(x[:, j-1]) # [b, in_dim]
            alpha, beta = alpha_v[..., :self.num_heads], alpha_v[..., self.num_heads:self.num_heads * 2]
            # print(alpha_v[..., 2 * self.num_heads:].shape)
            v = alpha_v[..., 2 * self.num_heads:].view(b, 4, self.rank, self.sqrt_dim)
            # print(v.shape)
            v_alpha1, v_alpha2, v_beta1, v_beta2 = \
                v[:, 0].contiguous(), \
                v[:, 1].contiguous(), \
                v[:, 2].contiguous(), \
                v[:, 3].contiguous()
            v_alpha = torch.einsum('bkd,bkp->bdp', v_alpha1, v_alpha2).view(b, self.num_heads, self.m_features)
            # print(v_alpha.shape)
            v_alpha = F.normalize(v_alpha, p=self.norm_p, dim=2, eps=1e-12)
            # v_alpha = v_alpha / v_alpha.max(2, keepdim=True)[0]
            v_beta = torch.einsum('bkd,bkp->bdp', v_beta1, v_beta2).view(b, self.num_heads, self.m_features)
            # print(v_beta.shape)
            v_beta = F.normalize(v_beta, p=self.norm_p, dim=2, eps=1e-12)
            # v_beta = v_beta / v_beta.max(2, keepdim=True)[0]
            alpha = torch.einsum('bk,bkd->bd', alpha, v_alpha)
            beta = torch.einsum('bk,bkd->bd', beta, v_beta)
            # print(alpha.shape, beta.shape)
            ms[:, j] = ms[:, j - 1] + alpha - beta
        ssm_inp = torch.cat([x, ms], axis=2) # size is [batch, length, in_size + m_size]
        return *self.ssm.forward_sequential(ssm_inp, state), ms[:, -1]
    
    def step(self, x, state=None, m_state=None, **kwargs):
        return self.forward_sequential(x.unsqueeze(1), state, m_state, **kwargs)
    
    @property
    def d_state(self):
        """Size after converting state to a single tensor."""
        return self.d_model

    @property
    def d_output(self):
        """Size of output."""
        return self.d_out

    @property
    def state_to_tensor(self):
        """Convert state into a single tensor output."""
        return lambda state: state