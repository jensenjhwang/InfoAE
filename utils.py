import torch
import torch.nn as nn
import math
import time

def donsker_varadhan_loss(l, m):
    '''

    Note that vectors should be sent as 1x1.

    Args:
        l: Local feature map.
        m: Multiple globals feature map.

    Returns:
        torch.Tensor: Loss.

    '''
    # print(l.shape, m.shape)
    N, units, n_locals = l.size()
    n_multis = m.size(2)

    # First we make the input tensors the right shape.
    l = l.view(N, units, n_locals)
    l = l.permute(0, 2, 1)
    l = l.reshape(-1, units)

    m = m.view(N, units, n_multis)
    m = m.permute(0, 2, 1)
    m = m.reshape(-1, units)

    # Outer product, we want a N x N x n_local x n_multi tensor.
    u = torch.mm(m, l.t())
    u = u.reshape(N, n_multis, N, n_locals).permute(0, 2, 3, 1)

    # Since we have a big tensor with both positive and negative samples, we need to mask.
    mask = torch.eye(N).to(l.device)
    n_mask = 1 - mask

    # Positive term is just the average of the diagonal.
    # print("mask:", mask.shape, "u.mean(2):", u.mean(2).shape)
    # E_pos = (u.mean(2) * mask).sum() / mask.sum()
    E_pos = (u.mean(2) * mask.unsqueeze(-1)).sum() / mask.sum()


    # Negative term is the log sum exp of the off-diagonal terms. Mask out the positive.
    # u -= 10 * (1 - n_mask)
    # u_max = torch.max(u)
    # E_neg = torch.log((n_mask * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    # loss = E_neg - E_pos
    u -= 10 * (1 - n_mask.unsqueeze(-1).unsqueeze(-1))
    u_max = torch.max(u)
    E_neg = torch.log((n_mask.unsqueeze(-1).unsqueeze(-1) * torch.exp(u - u_max)).sum() + 1e-6) + u_max - math.log(n_mask.sum())
    loss = E_neg - E_pos

    return loss

def weights_init(m):
    """
    Initialise weights of the model.
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def normalize_to_zero_one(x):
    """
    Linearly transform so that min = 0, max = 1
    """
    return x - torch.min(x) / (torch.max(x) - torch.min(x))

class Timer():
    def __init__(self, name):
        self.name = name
        self.timer = time.time()
    
    def stop(self):
        # print(f"{self.name} took {time.time() - self.timer}s")
        pass

def time_this_fn(fn):
    def inner(*args, **kwargs):
        start_time = time.time()
        fn(*args, **kwargs)
        print(f"{fn.__name__} took {time.time() - self.timer}s")
    return inner