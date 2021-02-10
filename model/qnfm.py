  
import torch
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, MultiLayerPerceptron, FeaturesLinear
from layer import QuatizationEmbedding
import torch.nn.functional as F 
from scipy.cluster.vq import vq

class QuatNeuralFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of Neural Factorization Machine.
    Reference:
        X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.
    """

    def __init__(self, params, mlp_dims, dropouts):
        super().__init__()
        self.dim = params["dim"]
        self.field_dims = params["field_dims"]
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:]), dtype=np.long)
        self.field_len = len(self.field_dims)
        self.M = params["M"]
        self.K = params["K"]
        self.device = params["device"]

        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.quatization = QuatizationEmbedding(self.field_dims, self.dim, self.K, self.M, self.device)
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
        cross_term = self.fm(self.quatization(x))
        x = self.linear(x) + self.mlp(cross_term)
        return x.squeeze(1)
    
    def copy(self, pre_state_dict):
        for name, param in self.named_parameters():
            if name in pre_state_dict.keys():
                param.data = pre_state_dict[name]
        # sdict = self.state_dict()
        # for key in sdict.keys():
        #     print(key, pre_state_dict[key])
        #     sdict[key] = pre_state_dict[key]

        self.weigt_on_cpu = np.float32(pre_state_dict['embedding.embedding.weight'].cpu())

    def quat_copy(self):
        self.quatization.load()

    def Embedding_pQ(self, method="pq"):
        self.quatization.initial_params(raw_weight=self.weigt_on_cpu)

    def update_b(self):
        self.quatization.update_cb_index(raw_weight=self.weigt_on_cpu)

    def calcu_distance_field(self):
        plen = int(self.dim/self.M)
        distance = torch.ones(self.field_len)
        for i in range(self.field_len):
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            ind = self.quatization.cb_index.weight.data[self.offsets[i]: self.offsets[i+1], ] + i*self.K
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                cluster_result[:, j*plen:j*plen+plen] = self.quatization.codebooks(ind[:, j])[:, j*plen:j*plen+plen]
            distance[i] = F.pairwise_distance(material, cluster_result, p=2).mean()
        # print(distance)
        return distance