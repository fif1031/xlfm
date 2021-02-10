  
import torch
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear
import torch.nn.functional as F 
from scipy.cluster.vq import vq

class NeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, params, mlp_dims, dropouts):
        super().__init__()
        self.dim = params["dim"]
        self.field_dims = params["field_dims"]

        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(reduce_sum=False),
            torch.nn.BatchNorm1d(self.dim),
            torch.nn.Dropout(dropouts[0])
        )
        self.mlp = MultiLayerPerceptron(self.dim, mlp_dims, dropouts[1])

        # for name,_ in self.named_parameters():
        #     print(name)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        cross_term = self.fm(self.embedding(x))
        x = self.linear(x) + self.mlp(cross_term)
        return x.squeeze(1)
    
    def copy(self, pre_state_dict):
        for name, param in self.named_parameters():
            param.data = pre_state_dict[name]
        # sdict = self.state_dict()
        # for key in sdict.keys():
        #     print(key, pre_state_dict[key])
        #     sdict[key] = pre_state_dict[key]

        self.weigt_on_cpu = np.float32(pre_state_dict['embedding.embedding.weight'].cpu())
