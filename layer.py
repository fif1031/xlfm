import time
import torch
import nanopq
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import Counter
from scipy.cluster.vq import vq
from torch.nn.parameter import Parameter

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MLP(nn.Module):

    def __init__(self, neural_num, dim, layers=100):
        super(MLP, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.output = nn.Linear(neural_num, dim)
        self.neural_num = neural_num

    def forward(self, x):
        # x = x.to(dtype=torch.float)
        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break
        return self.output(x)

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)


class MLP_D(nn.Module):

    def __init__(self, in_dim, neural_num, out_dim, layers=100):
        super(MLP_D, self).__init__()
        self.input = nn.Linear(in_dim, neural_num)
        self.in_bn = nn.BatchNorm1d(neural_num)
        self.linears = nn.ModuleList([nn.Linear(neural_num, neural_num, bias=False) for i in range(layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(neural_num) for i in range(layers)])
        self.output = nn.Linear(neural_num, out_dim)
        self.neural_num = neural_num

    def forward(self, x):
        # x = x.to(dtype=torch.float)
        x = self.input(x)
        x = self.in_bn(x)
        x = torch.relu(x)
        for (i, linear), bn in zip(enumerate(self.linears), self.bns):
            x = linear(x)
            x = bn(x)
            x = torch.relu(x)

            if torch.isnan(x.std()):
                print("output is nan in {} layers".format(i))
                break
        return torch.sign(self.output(x))


class MultiMLPEmb(torch.nn.Module):

    def __init__(self, field_num, dim, device):
        super().__init__()
        self.field_num = field_num
        self.dim = dim
        self.proj = nn.ModuleList([nn.Linear(dim, dim, bias=False) for i in range(field_num)])
        self.device = device

    def forward(self, x):
        emb = torch.zeros_like(x,device=self.device)
        for i in range(self.field_num):
            emb[:, i*self.dim:i*self.dim+self.dim, :] = self.proj[i](x[:, i*self.dim:i*self.dim+self.dim, :])
        return emb

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)



class QuatizationEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, max_K, M, device):
        super().__init__()
        self.K = max_K
        self.M = M
        self.dim = embed_dim
        self.field_dims = field_dims
        self.field_len = len(field_dims)
        self.device = device
        self.offsets_a = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:]), dtype=np.long)
        self.index_offsets = np.array([i*max_K for i in range(len(field_dims))], dtype=np.long)

        self.codebooks = torch.nn.Embedding(len(field_dims)*max_K, embed_dim)
        self.cb_index = torch.nn.Embedding(sum(field_dims), M)
        self.cb_index.weight.requires_grad = False
        self.cb_index.weight.data = torch.randint(max_K, (sum(field_dims), M)).to(dtype=torch.long)
        

        self.cost =  [0 for _ in range(6)]
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets_a).unsqueeze(0)
        index = self.cb_index(x).to(dtype=torch.long)
        emb = torch.zeros((x.shape[0], x.shape[1], self.dim),device=self.device)
        plen = int(self.dim/self.M)
        # for i in range(self.field_len):
        #     for j in range(self.M):
        #         index_i = index[:,i,j] + index[:,i,j].new_tensor(self.index_offsets[i]).unsqueeze(0)
        #         emb[:,i,j*plen:j*plen+plen] = self.codebooks(index_i)[:,j*plen:j*plen+plen]
        for i in range(self.M):
            index_i = index[:,:,i] + index[:,:,i].new_tensor(self.index_offsets).unsqueeze(0)
            emb[:,:,i*plen:i*plen+plen] = self.codebooks(index_i)[:,:,i*plen:i*plen+plen]
        return emb

    def initial_params(self, raw_weight=None, method="pq"):
        plen = int(self.dim/self.M)
        for i in range(self.field_len):
            emb_weight = raw_weight[self.offsets[i]:self.offsets[i+1], ]
            begin = i*self.K
            end = (i+1)*self.K
            if self.field_dims[i] < self.K:
                emb_weight = torch.from_numpy(emb_weight).to(self.device)
                self.codebooks.weight.data[begin:begin+self.field_dims[i],] = emb_weight
                index = torch.from_numpy(np.arange(self.field_dims[i])).to(self.device, dtype=torch.long)
                for j in range(self.M):
                    self.cb_index.weight.data[self.offsets[i]:self.offsets[i+1],j] = index
                continue
            if method == "pq": 
                pq = nanopq.PQ(M=self.M, Ks=self.K, verbose=False)
            elif method == "opq":
                pq = nanopq.OPQ(M=self.M, Ks=self.K, verbose=False)
            pq.fit(emb_weight)
            weight_encode = pq.encode(emb_weight).astype(np.long)
            weight_encode = torch.from_numpy(weight_encode).to(self.device)
            codewords = torch.from_numpy(pq.codewords).to(self.device)
            # print(weight_encode)
            for j in range(self.M):
                self.codebooks.weight.data[begin:end, j*plen:j*plen+plen] = codewords[j]
                self.cb_index.weight.data[self.offsets[i]:self.offsets[i+1],j] = weight_encode[:,j]

    def update_cb_index(self, raw_weight):
        self.codebook_on_cpu = np.float32(self.codebooks.weight.data.cpu())
        
        plen = int(self.dim/self.M)
        for i in range(self.field_len):
            begin = i*self.K
            if self.field_dims[i] < self.K:
                end = begin + self.field_dims[i]
            else: 
                end = begin + self.K
            for j in range(self.M):
                codewords = self.codebook_on_cpu[begin:end, j*plen:j*plen+plen]
                ind, _ = vq(raw_weight[self.offsets[i]:self.offsets[i+1], j*plen:j*plen+plen], codewords)
                self.cb_index.weight.data[self.offsets[i]:self.offsets[i+1], j] = torch.from_numpy(ind).to(self.device, dtype=torch.long)


    def save(self):
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/q_dim32_1024_f.pt"
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/deep_q_dim32_2048_f.pt"
        path = "/data2/home/gangwei/project/pytorch-fm/chkpt/criteo/quat/q_dim32_1024_f.pt"
        torch.save(self.state_dict(), path)

    def load(self):
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/q_dim32_1024_f.pt"
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/deep_q_dim32_2048_f.pt"
        path = "/data2/home/gangwei/project/pytorch-fm/chkpt/criteo/quat/q_dim32_1024_f.pt"
        self.load_state_dict(torch.load(path))
        self.eval()



class WeightedSumQuatEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim, max_K, M, ACTION, device):
        super().__init__()
        self.M = M
        self.K = max_K
        self.dim = embed_dim
        self.device = device
        self.ACTION = ACTION
        self.field_dims = field_dims
        self.field_len = len(field_dims)
        self.offsets_a = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:]), dtype=np.long)
        self.index_offsets = max_K*np.arange(len(field_dims), dtype=np.long)

        self.codebooks = torch.nn.Embedding(len(field_dims)*max_K, embed_dim)
        self.cb_index = torch.nn.ModuleList([torch.nn.Embedding(sum(field_dims), M) for _ in range(len(ACTION))])
        for i in range(len(ACTION)):
            self.cb_index[i].weight.requires_grad = False
            self.cb_index[i].weight.data = torch.randint(max_K, (sum(field_dims), M)).to(dtype=torch.long)

        self.cost =  [0 for _ in range(6)]
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, arch_prob):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        plen = int(self.dim/self.M)
        b = time.time()
        x = x + x.new_tensor(self.offsets_a).unsqueeze(0)
        emb = torch.zeros((x.shape[0], x.shape[1], self.dim),device=self.device)
        for j in range(len(self.ACTION)):
            index = self.cb_index[j](x).to(dtype=torch.long)
            arch_prob_j = arch_prob[:, j].reshape(len(arch_prob[:, j]),1).repeat(1,plen)
            for i in range(self.M):
                index_i = index[:,:,i] + index[:,:,i].new_tensor(self.index_offsets).unsqueeze(0)
                emb[:,:,i*plen:i*plen+plen] += arch_prob_j * self.codebooks(index_i)[:,:,i*plen:i*plen+plen]
        self.cost[2] += time.time()-b
        return emb

    def initial_params(self, raw_weight=None, method="pq"):
        plen = int(self.dim/self.M)
        for i in range(self.field_len):
            emb_weight = raw_weight[self.offsets[i]:self.offsets[i+1], ]
            begin = i*self.K
            max_K = self.K
            for k in range(len(self.ACTION)):
                if self.field_dims[i] < self.ACTION[k]:
                    max_K = self.ACTION[k-1]
                    break
            end = begin+max_K
            if max_K == 1:
                emb_weight = torch.from_numpy(emb_weight).to(self.device)
                self.codebooks.weight.data[begin:begin+self.field_dims[i],] = emb_weight
                index = torch.from_numpy(np.arange(self.field_dims[i])).to(self.device, dtype=torch.long)
                for j in range(self.M):
                    self.cb_index[0].weight.data[self.offsets[i]:self.offsets[i+1],j] = index
                continue
            # print(max_K)
            if method == "pq": 
                pq = nanopq.PQ(M=self.M, Ks=max_K, verbose=False)
            elif method == "opq":
                pq = nanopq.OPQ(M=self.M, Ks=max_K, verbose=False)

            pq.fit(emb_weight)
            weight_encode = pq.encode(emb_weight)
            sorted_codewords = np.zeros(pq.codewords.shape)
            for j in range(self.M):
                d = Counter(weight_encode[:, j])
                d_s = sorted(d.items(),key=lambda x:x[1],reverse=True)
                for k in range(len(d_s)):
                    sorted_codewords[j, k, :] = pq.codewords[j, d_s[k][0], :]

            for j in range(self.M):
                self.codebooks.weight.data[begin:end, j*plen:j*plen+plen] = torch.from_numpy(sorted_codewords[j]).to(self.device)
                for k in range(1, len(self.ACTION)):
                    if self.ACTION[k] > self.field_dims[i]:
                        break
                    ind, _ = vq(emb_weight[:, j*plen:j*plen+plen], sorted_codewords[j,:self.ACTION[k],:])
                    self.cb_index[k].weight.data[self.offsets[i]:self.offsets[i+1], j] = torch.from_numpy(ind).to(self.device, dtype=torch.long)


    def update_cb_index(self, raw_weight, flag):
        b2 = time.time()
        self.codebook_on_cpu = np.float32(self.codebooks.weight.data.cpu())
        
        plen = int(self.dim/self.M)
        for i in range(self.field_len):
            begin = i*self.K
            for k in range(1, len(self.ACTION)):
                if flag[i, k] != 1:
                    continue
                end = begin + self.ACTION[k]
                for j in range(self.M):
                    codewords = self.codebook_on_cpu[begin:end, j*plen:j*plen+plen]
                    ind, _ = vq(raw_weight[self.offsets[i]:self.offsets[i+1], j*plen:j*plen+plen], codewords)
                    self.cb_index[k].weight.data[self.offsets[i]:self.offsets[i+1], j] = torch.from_numpy(ind).to(self.device, dtype=torch.long)

    def save(self):
        path = "/data2/home/gangwei/project/pytorch-fm/exp/avazu/quat/wq_500_512.pt"
        torch.save(self.state_dict(), path)

    def load(self):
        path = "/data2/home/gangwei/project/pytorch-fm/exp/avazu/quat/wq_500_512.pt"
        self.load_state_dict(torch.load(path))
        self.eval()



class MonopolySumQuatEmbedding(nn.Module):

    def __init__(self, field_dims, embed_dim, max_K, M, ACTION, device):
        super().__init__()        
        self.M = M
        self.K = max_K
        self.dim = embed_dim
        self.device = device
        self.ACTION = ACTION
        self.field_dims = field_dims
        self.field_len = len(field_dims)
        self.offsets_a = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:]), dtype=np.long)
        self.index_offsets = np.array([ACTION[i]*np.arange(len(field_dims)) for i in range(len(ACTION))], dtype=np.long)
        self.threshold = 500
        self.index_offsets[0,:] = self.threshold * np.arange(len(field_dims))

        self.codebooks = nn.ModuleList([nn.Embedding(len(field_dims)*ACTION[i], embed_dim) for i in range(len(ACTION))])
        self.codebooks[0] = nn.Embedding(len(field_dims)*self.threshold, embed_dim)

        self.cb_index = nn.ModuleList([nn.Embedding(sum(field_dims), M) for _ in range(len(ACTION))])
        for i in range(len(ACTION)):
            self.cb_index[i].weight.requires_grad = False
            self.cb_index[i].weight.data = torch.randint(ACTION[i], (sum(field_dims), M)).to(dtype=torch.long)

        self.cost =  [0 for _ in range(6)]

        # for name,_ in self.named_parameters():
        #     print(name)
        # torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x, arch_prob_raw, arch_p=None, prior=None, temperature=None, flag=0):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        if flag==0:
            arch_prob = F.gumbel_softmax(arch_p*prior,tau=temperature, hard=True, eps=1e-10, dim=-1)
        else:
            arch_prob = arch_prob_raw
        plen = int(self.dim/self.M)
        b = time.time()
        x = x + x.new_tensor(self.offsets_a).unsqueeze(0)
        emb = torch.zeros((x.shape[0], x.shape[1], self.dim),device=self.device)
        for j in range(len(self.ACTION)):
            index = self.cb_index[j](x).to(dtype=torch.long)
            arch_prob_j = arch_prob[:, j].reshape(len(arch_prob[:, j]),1).repeat(1,plen)
            for i in range(self.M):
                index_i = index[:,:,i] + index[:,:,i].new_tensor(self.index_offsets[j, :]).unsqueeze(0)
                emb[:,:,i*plen:i*plen+plen] += arch_prob_j * self.codebooks[j](index_i)[:,:,i*plen:i*plen+plen]
        self.cost[2] += time.time()-b
        return emb

    def initial_params(self, raw_weight=None, method="pq"):
        plen = int(self.dim/self.M)
        for i in range(self.field_len):
            emb_weight = raw_weight[self.offsets[i]:self.offsets[i+1], ]
            for k in range(len(self.ACTION)):
                if self.field_dims[i] < self.ACTION[k]:
                    break
                max_K = self.ACTION[k]
                if max_K == 1:
                    if self.field_dims[i] > self.threshold:
                        continue
                    begin = i*self.threshold
                    emb_weight_t = torch.from_numpy(emb_weight).to(self.device)
                    self.codebooks[0].weight.data[begin:begin+self.field_dims[i],] = emb_weight_t
                    index = torch.from_numpy(np.arange(self.field_dims[i])).to(self.device, dtype=torch.long)
                    for j in range(self.M):
                        self.cb_index[0].weight.data[self.offsets[i]:self.offsets[i+1],j] = index
                    continue
                begin = i*self.ACTION[k]
                end = begin+max_K
                if method == "pq": 
                    pq = nanopq.PQ(M=self.M, Ks=max_K, verbose=False)
                elif method == "opq":
                    pq = nanopq.OPQ(M=self.M, Ks=max_K, verbose=False)

                pq.fit(emb_weight)
                weight_encode = (pq.encode(emb_weight)).astype(np.long)
                weight_encode = torch.from_numpy(weight_encode).to(self.device)
                codewords = torch.from_numpy(pq.codewords).to(self.device)
            
                for j in range(self.M):
                    self.codebooks[k].weight.data[begin:end, j*plen:j*plen+plen] = codewords[j]
                    self.cb_index[k].weight.data[self.offsets[i]:self.offsets[i+1],j] = weight_encode[:,j]

    def update_cb_index(self, raw_weight, flag):
        plen = int(self.dim/self.M)
        for k in range(1, len(self.ACTION)):
            self.codebook_on_cpu = np.float32(self.codebooks[k].weight.data.cpu())
            for i in range(self.field_len):        
                if flag[i, k] != 1:
                    continue
                begin = i*self.ACTION[k]
                end = begin + self.ACTION[k]
                for j in range(self.M):
                    codewords = self.codebook_on_cpu[begin:end, j*plen:j*plen+plen]
                    ind, _ = vq(raw_weight[self.offsets[i]:self.offsets[i+1], j*plen:j*plen+plen], codewords)
                    self.cb_index[k].weight.data[self.offsets[i]:self.offsets[i+1], j] = torch.from_numpy(ind).to(self.device, dtype=torch.long)

    def save(self):
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/mq_dim32_2048_f.pt"
        path = "/data2/home/gangwei/project/pytorch-fm/chkpt/criteo/quat/mq_dim32_2048_f.pt"
        torch.save(self.state_dict(), path)

    def load(self):
        # path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/quat/mq_dim32_2048_f.pt"
        path = "/data2/home/gangwei/project/pytorch-fm/chkpt/criteo/quat/mq_dim32_2048_f.pt"
        self.load_state_dict(torch.load(path))
        self.eval()

