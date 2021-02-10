from collections import defaultdict

import torch
from torch.optim.optimizer import Optimizer


class FTRL(Optimizer):
    """ Implements FTRL online learning algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        alpha (float, optional): alpha parameter (default: 1.0)
        beta (float, optional): beta parameter (default: 1.0)
        l1 (float, optional): L1 regularization parameter (default: 1.0)
        l2 (float, optional): L2 regularization parameter (default: 1.0)
    .. _Ad Click Prediction: a View from the Trenches: 
        https://www.eecs.tufts.edu/%7Edsculley/papers/ad-click-prediction.pdf
    """

    def __init__(self, params, alpha=1.0, beta=1.0, l1=1.0, l2=1.0):
        if not 0.0 < alpha:
            raise ValueError("Invalid alpha parameter: {}".format(alpha))
        if not 0.0 < beta:
            raise ValueError("Invalid beta parameter: {}".format(beta))
        if not 0.0 <= l1:
            raise ValueError("Invalid l1 parameter: {}".format(l1))
        if not 0.0 <= l2:
            raise ValueError("Invalid l2 parameter: {}".format(l2))

        defaults = dict(alpha=alpha, beta=beta, l1=l1, l2=l2)
        super(FTRL, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    state["z"] = torch.zeros_like(p.data)
                    state["n"] = torch.zeros_like(p.data)

                z, n = state["z"], state["n"]

                theta = (n + grad ** 2).sqrt() / group["alpha"] - n.sqrt()
                z.add_(grad - theta * p.data)
                n.add_(grad ** 2)

                p.data = (
                    -1
                    / (group["l2"] + (group["beta"] + n.sqrt()) / group["alpha"])
                    * (z - group["l1"] * z.sign())
                )
                p.data[z.abs() < group["l1"]] = 0

        return loss

class DataPrefetcher():
    def __init__(self, loader, device):
        self.loader = iter(loader)
        self.device = device
        self.stream = torch.cuda.Stream()
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            #     self.next_input = self.next_input.float()

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def md_solver(n, alpha, d0=None, B=None, round_dim=True, k=None):
    '''
    An external facing function call for mixed-dimension assignment
    with the alpha power temperature heuristic
    Inputs:
    n -- (torch.LongTensor) ; Vector of num of rows for each embedding matrix
    alpha -- (torch.FloatTensor); Scalar, non-negative, controls dim. skew
    d0 -- (torch.FloatTensor); Scalar, baseline embedding dimension
    B -- (torch.FloatTensor); Scalar, parameter budget for embedding layer
    round_dim -- (bool); flag for rounding dims to nearest pow of 2
    k -- (torch.LongTensor) ; Vector of average number of queries per inference
    '''
    n, indices = torch.sort(n)
    k = k[indices] if k is not None else torch.ones(len(n))
    d = alpha_power_rule(n.type(torch.float) / k, alpha, d0=d0, B=B)
    if round_dim:
        d = pow_2_round(d)
    undo_sort = [0] * len(indices)
    for i, v in enumerate(indices):
        undo_sort[v] = i
    return d[undo_sort]


def alpha_power_rule(n, alpha, d0=None, B=None):
    if d0 is not None:
        lamb = d0 * (n[0].type(torch.float) ** alpha)
    elif B is not None:
        lamb = B / torch.sum(n.type(torch.float) ** (1 - alpha))
    else:
        raise ValueError("Must specify either d0 or B")
    d = torch.ones(len(n)) * lamb * (n.type(torch.float) ** (-alpha))
    for i in range(len(d)):
        if i == 0 and d0 is not None:
            d[i] = d0
        else:
            d[i] = 1 if d[i] < 1 else d[i]
    return (torch.round(d).type(torch.long))

def pow_2_round(dims):
    return 2 ** torch.round(torch.log2(dims.type(torch.float)))