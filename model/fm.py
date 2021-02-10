import torch

from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiMLPEmb

import time

class FactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Factorization Machine.

    Reference:
        S Rendle, Factorization Machines, 2010.
    """

    def __init__(self, params):
        super().__init__()
        field_dims, embed_dim, device = params["field_dims"], params["dim"], params["device"]
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.mlps = MultiMLPEmb(len(field_dims), embed_dim, device)
        self.device = params["device"]
        self.time_sum = 0
        self.time_cnt = 0

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        self.time_cnt += 1
        self.time_sum -= time.time()
        x = self.linear(x) + self.fm(self.embedding(x))
        # x = self.linear(x) + self.fm(self.mlps(self.embedding(x)))
        self.time_sum += time.time()
        return x.squeeze(1)

    def copy(self, pre_state_dict):
        self.linear.bias.data = pre_state_dict['linear.bias'].to(self.device)
        self.linear.fc.weight.data = pre_state_dict['linear.fc.weight'].to(self.device)
        self.embedding.embedding.weight.data = pre_state_dict['embedding.embedding.weight'].to(self.device)
