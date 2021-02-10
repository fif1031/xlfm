import torch
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of DeepFM.
    Reference:
        H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.
    """

    def __init__(self, params, mlp_dims, dropout):
        super().__init__()
        self.dim = params["dim"]
        self.field_dims = params["field_dims"]

        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.embed_output_dim = len(self.field_dims) * self.dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

        for name,_ in self.named_parameters():
            print(name)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.fm(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)

    def copy(self, pre_state_dict):
        for name, param in self.named_parameters():
            param.data = pre_state_dict[name]
        # sdict = self.state_dict()
        # for key in sdict.keys():
        #     print(key, pre_state_dict[key])
        #     sdict[key] = pre_state_dict[key]

        self.weigt_on_cpu = np.float32(pre_state_dict['embedding.embedding.weight'].cpu())
