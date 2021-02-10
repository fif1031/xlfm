import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, QuatizationEmbedding, MultiMLPEmb
from scipy.cluster.vq import vq
import nanopq
import time

class QuantizationFactorizationMachine(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.dim = self.params["dim"]
        self.field_dims = self.params["field_dims"]
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:]), dtype=np.long)
        self.field_len = len(self.field_dims)
        self.M = params["M"]
        self.K = params["K"]
        self.q_size = int(self.dim/self.M)
        self.device = params["device"]
        print(self.field_dims)
        print(self.offsets)
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.quatization = QuatizationEmbedding(self.field_dims, self.dim, self.K, self.M, self.device)
        # self.mlps = MultiMLPEmb(self.field_len, self.dim, self.device)
        import pickle
        with open(params["popular_path"], "rb") as f:
            popular = pickle.load(f)
        self.popular = torch.from_numpy(popular).to(self.device)
        print(self.popular.shape)
        self.cost = [0 for _ in range(6)]
        self.time_sum = 0
        self.time_cnt = 0

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        self.time_cnt += 1
        self.time_sum -= time.time()
        # x_emb = self.mlps(self.quatization(x))
        x_emb = self.quatization(x)
        x = self.linear(x) + self.fm(x_emb)
        self.time_sum += time.time()
        return x.squeeze(1)

    def copy(self, pre_state_dict):
        self.linear.bias.data = pre_state_dict['linear.bias'].to(self.device)
        self.linear.fc.weight.data = pre_state_dict['linear.fc.weight'].to(self.device)
        self.embedding.embedding.weight.data = pre_state_dict['embedding.embedding.weight'].to(self.device)

        self.linear.bias.requires_grad = False
        self.linear.fc.weight.requires_grad = False
        self.embedding.embedding.weight.requires_grad = False

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
        w_distance = torch.ones(self.field_len)
        for i in range(self.field_len):
            if self.field_dims[i] < self.K:
                distance[i] = 0 
                continue
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            ind = self.quatization.cb_index.weight.data[self.offsets[i]: self.offsets[i+1], ] + i*self.K
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                cluster_result[:, j*plen:j*plen+plen] = self.quatization.codebooks(ind[:, j])[:, j*plen:j*plen+plen]
            dis = F.pairwise_distance(material, cluster_result, p=2)
            distance[i] = dis.mean()
            w_distance[i] = (dis*self.popular[self.offsets[i]:self.offsets[i+1]]).sum()
        # print(distance)
        return distance, w_distance
    
    def calcu_weighted_distance_field(self):
        plen = int(self.dim/self.M)
        distance = torch.ones(self.field_len)
        for i in range(self.field_len):
            if self.field_dims[i] < self.K:
                distance[i] = 0 
                continue
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            ind = self.quatization.cb_index.weight.data[self.offsets[i]: self.offsets[i+1], ] + i*self.K
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                cluster_result[:, j*plen:j*plen+plen] = self.quatization.codebooks(ind[:, j])[:, j*plen:j*plen+plen]
            # print(F.pairwise_distance(material, cluster_result, p=2).shape, self.popular[self.offsets[i]:self.offsets[i+1]].shape)
            distance[i] = (F.pairwise_distance(material, cluster_result, p=2)*self.popular[self.offsets[i]:self.offsets[i+1]]).sum()

        # print(distance)
        return distance


