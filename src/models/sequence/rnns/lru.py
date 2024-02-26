
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
        
        super().__init__()
        self.d_input = d_input
        self.d_out = d_input
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
        if self.transposed: 
            input_sequence = rearrange(input_sequence, 'b d ... -> b ... d')
        
        if self.rnn_parameterization == 'lru':
            Lambda = torch.exp(-self.nu_log.exp() + 1j * self.theta_log.exp())
            B_norm = self.B * self.gamma_log.exp().unsqueeze(1)
            B_input =  input_sequence + 0j
        
        elif self.rnn_parameterization == 'real':
            Lambda = torch.exp(-torch.exp(self.Lambda))
            B_norm = self.B
            B_input =  input_sequence

        Lambda_elements = Lambda.repeat(input_sequence.shape[1], 1)
        # Bu_elements = torch.einsum('btd,nd->btn', input_sequence + 0j, B_norm) # [batch, len, dim] @ [dim, outdim] = [batch, len, outdim]
        Bu_elements = torch.einsum('btd,nd->btn', B_input, B_norm)
        elements = (Lambda_elements.unsqueeze(0).to(Bu_elements.dtype), Bu_elements)
        _, inner_states = associative_scan(binary_operator_diag_torch, elements, axis=1)
        
        if self.rnn_parameterization == 'lru':
            y = torch.einsum('btd,nd->btn', inner_states, self.C).real
        
        elif self.rnn_parameterization == 'real':
            y = torch.einsum('btd,nd->btn', inner_states, self.C)
        
        if self.diag_D:
            y += self.D * input_sequence
        
        else:
            y += torch.einsum('btd,nd->btn', input_sequence, self.D)
            
        y = self.activation(y)
        
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