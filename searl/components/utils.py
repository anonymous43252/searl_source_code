from collections import namedtuple

import torch

fields = ('state', 'action', 'next_state', 'reward', 'done', 'weight', 'index')
Transition = namedtuple('Transition', fields)
Transition.__new__.__defaults__ = (None,) * len(Transition._fields)


def to_tensor(ndarray, requires_grad=False):
    return torch.from_numpy(ndarray).float().requires_grad_(requires_grad)
