import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
from layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MonopolySumQuatEmbedding, WeightedSumQuatEmbedding
from scipy.cluster.vq import vq
import nanopq
import time
from collections import Counter
from torch.autograd import Variable

ACTION = [1, 64, 128, 256, 512, 1024, 2048]
# ACTION = [1, 64, 128, 256, 512]


def _concat(xs):
	return torch.cat([x.view(-1) for x in xs])

class GumbelSoftmaxQuantizationFM(torch.nn.Module):

    def __init__(self, params):
        super().__init__()
        self.params = params

        self.M = params["M"]
        self.K = params["K"]
        self.dim = params["dim"]
        self.device = params["device"]
        self.share = params["share"]
        self.field_dims = params["field_dims"]
        # print(self.share)
        self.cnt = 0
        self.threshold = 500
        self.q_size = int(self.dim/self.M)
        self.field_len = len(self.field_dims)
        self.offsets = np.array((0, *np.cumsum(self.field_dims)[:]), dtype=np.long)
        self.base_men = self.offsets[-1]*32*self.dim
        # self.criterion = criterion
        print(self.field_dims)
        print(self.offsets)
        self.embedding = FeaturesEmbedding(self.field_dims, self.dim)
        self.linear = FeaturesLinear(self.field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)
        # self.quatization = QuatizationEmbedding(self.field_dims, self.dim, self.K, self.M, self.device)
        if self.share == 1:
            self.quatization = WeightedSumQuatEmbedding(self.field_dims, self.dim, self.K, self.M, ACTION, self.device)
        else:
            self.quatization = MonopolySumQuatEmbedding(self.field_dims, self.dim, self.K, self.M, ACTION, self.device)
        
        self._arch_parameters = Variable(
            torch.ones((self.field_len, len(ACTION)), dtype=torch.float, device=self.device) / 2, requires_grad=True)
        self._arch_parameters.data.add_(
            torch.randn_like(self._arch_parameters)*1e-3)
        # print(self._arch_parameters.grad)
        self.prior_flag = torch.ones(self.field_len, len(ACTION), device=self.device)* -1e5
        # for i in range(self.field_len):
        #     if self.field_dims[i] < self.threshold:
        #         self.prior_flag[i, 0] = 1
        #     for k in range(1, len(ACTION)):
        #         if ACTION[k] > self.field_dims[i]:
        #             break
        #         self.prior_flag[i, k] = 1
        #         self._arch_parameters.data[i, k] += 0.001*k
        #     if self.field_dims[i] > 10000:
        #         self._arch_parameters.data[i, len(ACTION)-1] = 0.85
            # self._arch_parameters.data[i, 6] = 1
        import pickle
        with open(params["popular_path"], "rb") as f:
            popular = pickle.load(f)
        self.popular = torch.from_numpy(popular).to(self.device)
        self.arch_max = []
        for i in range(self.field_len):
            self.arch_max.append(0)
            if self.field_dims[i] < 150:
                self.prior_flag[i, 0] = 1
            for k in range(1, len(ACTION)):
                if ACTION[k]*2.5 > self.field_dims[i]:
                    break
                self.prior_flag[i, k] = 1
                self.arch_max[i] = k
                self._arch_parameters.data[i, k] -= 0.002*k
            # if self.field_dims[i] > 190000:
            #     self._arch_parameters.data[i, len(ACTION)-1] = 0.85
        
        self.arch_prob = torch.zeros(self.field_len, len(ACTION), device=self.device)
        self.cost = [0 for _ in range(6)]
        self.temperature = 0
        self.time_sum = 0
        self.time_cnt = 0

    def new(self):
        model_new = GumbelSoftmaxQuantizationFM(self.params).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data = y.data.clone()
        return model_new

    def used_parameters(self):
        for name, param in self.named_parameters(recurse=True):
            if param.requires_grad:
                # print(name)
                yield param

    def arch_parameters(self):
        return [self._arch_parameters]

    def g_softmax(self, temperature):
        self.temperature = temperature
        return
        # self.arch_prob = F.gumbel_softmax(self._arch_parameters*self.prior_flag,
                                            # tau=temperature, hard=False, eps=1e-10, dim=-1)

    def set_arch(self, arch):
        ind = []
        for i in range(self.arch_prob.shape[0]):
            for j in range(len(ACTION)):
                if ACTION[j] == arch[i]:
                    m = j
                    break
            for j in range(self.arch_prob.shape[1]):
                if j == m:
                    self.arch_prob[i,j] = 1
                else:
                    self.arch_prob[i,j] = 0
        return ind 

    def arch_argmax(self):
        ind = []
        for i in range(self.arch_prob.shape[0]):
            m = (self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()
            for j in range(self.arch_prob.shape[1]):
                if j == m:
                    self.arch_prob[i,j] = 1
                else:
                    self.arch_prob[i,j] = 0
            ind.append(m)
        return ind 

    def calcu_memory_cost(self):
        cost = 0.0
        for i in range(self.field_len):
            select =  ACTION[(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()]
            if select==1:
                cost += self.field_dims[i]*32*self.dim
                continue
            cost += select*32*self.dim
            cost += self.field_dims[i]*np.log2(select)*self.M
        # print(cost/float(self.base_men))
        return cost/float(self.base_men)

    def genotype(self):
        genotype = []
        gen = []
        for i in range(self.field_len):
            # if self.prior_flag[i, 0] == 1:
            #     continue
            genotype.append((self.field_dims[i], ACTION[(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()]))
            gen.append(genotype[i][1])
            # print(self.field_dims[i], self._arch_parameters[i, :].data)
        # print(genotype)
        # genotype = [(self.field_dims[i],ACTION[(self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()]) for i in range(self.field_len)]
        return genotype, gen

    def forward(self, x, flag=0):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        self.time_cnt+=1
        self.time_sum-=time.time()
        x_emb = self.quatization(x, self.arch_prob, self._arch_parameters,self.prior_flag, self.temperature, flag)
        x = self.linear(x) + self.fm(x_emb)
        
        self.time_sum+=time.time()
        return x.squeeze(1)

    def clip(self):
        m = nn.Hardtanh(0.01, 1)
        self._arch_parameters.data = m(self._arch_parameters)

    def step(self, x, labels, criterion, temperature, arch_optimizer, 
                x_valid=None, labels_valid=None, unrolled=False):
        self.criterion = criterion
        self.zero_grad()
        arch_optimizer.zero_grad()
        self.g_softmax(temperature)
        if unrolled:
            loss = self._backward_step_unrolled(x, labels, x_valid, labels_valid, 1e-4)
        else:
            # loss = self._backward_step(x, labels)
            loss = self._backward_step(x_valid, labels_valid)
        arch_optimizer.step()

        # self.g_softmax(temperature)
        # for parms in self.arch_parameters():	
        #     print(parms,'-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
        return loss

    def step_with_limit(self, x, labels, criterion, temperature, arch_optimizer, 
                x_valid=None, labels_valid=None, unrolled=False):
        self.criterion = criterion
        self.zero_grad()
        arch_optimizer.zero_grad()
        self.g_softmax(temperature)
        if unrolled:
            loss = self._backward_step_unrolled(x, labels, x_valid, labels_valid, 1e-4)
        else:
            # loss = self._backward_step(x, labels)
            loss = self._backward_step(x_valid, labels_valid)
        last = self._arch_parameters.clone()
        last_grad =self._arch_parameters.grad.clone()
        arch_optimizer.step()

        # if self.calcu_memory_cost() > 1.0/22.0:
        # print(self.calcu_memory_cost())
        if self.calcu_memory_cost() > 1.0/24.0:
            # self._arch_parameters = last
            # print(last_grad)
            for i in range(self.field_len):
                # s = (self._arch_parameters[i, :]*self.prior_flag[i, :]).argmax().item()
                # s = self.arch_max[i]
                # if last_grad[i,s] < 0:
                #     self._arch_parameters[i, s].data.fill_(last[i, s]) 
                s = (last[i, :]*self.prior_flag[i, :]).argmax().item()
                # print(s)
                for j in range(s+1, len(ACTION)):
                    if last_grad[i,j] < 0:
                        # print(1)
                        self._arch_parameters[i, j].data.fill_(last[i, j]) 
                # self._arch_parameters[i, s] = last[i, s].clone()
        # self.g_softmax(temperature)
        # for parms in self.arch_parameters():	
        #     print(parms,'-->grad_requirs:',parms.requires_grad,' -->grad_value:',parms.grad)
        return loss

    def _backward_step(self, x, labels):
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        loss.backward()
        # print(self._arch_parameters.grad[5:11,])
        # print(self._arch_parameters.data[5:11,])
        return loss
    
    def _backward_step_unrolled(self, x_train, labels_train,
		                            x_valid, labels_valid, lr):
        unrolled_model = self._compute_unrolled_model(x_train, labels_train, lr)
        unrolled_model.g_softmax(self.temperature)
        unrolled_inference = unrolled_model(x_valid)
        unrolled_loss = self.criterion(unrolled_inference, labels_valid.float())
        
        unrolled_loss.backward()
        
        dalpha = [v.grad for v in unrolled_model.arch_parameters()]
        vector = [v.grad for v in unrolled_model.used_parameters()]
        # print(vector)
        implicit_grads = self._hessian_vector_product(vector, x_train, labels_train)
        
        for g,ig in zip(dalpha,implicit_grads):
            g.sub_(lr, ig)
        
        for v,g in zip(self.arch_parameters(), dalpha):
            v.grad = g.clone()
        return unrolled_loss
    
    def _compute_unrolled_model(self, x, labels, lr):
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        # print(type(self.used_parameters()))
        theta = _concat(self.used_parameters())
        dtheta = _concat(torch.autograd.grad(loss, self.used_parameters()))
        unrolled_model = self._construct_model_from_theta(theta.sub(dtheta, alpha=lr))
        return unrolled_model
    
    def _construct_model_from_theta(self, theta):
        model_new = self.new()
        model_dict = self.state_dict()
        params, offset = {}, 0
        for k,v in self.named_parameters():
            if v.requires_grad:
                v_length = np.prod(v.size())
                params[k] = theta[offset: offset+v_length].view(v.size())
                offset += v_length
            else:
                params[k] = v.clone()
        # print(params)
        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        model_new.linear.bias.requires_grad = False
        model_new.linear.fc.weight.requires_grad = False
        model_new.embedding.embedding.weight.requires_grad = False
        return model_new.to(self.device)
    
    def _hessian_vector_product(self, vector, x, labels, r=1e-2):
        R = r / _concat(vector).norm()
        for p,v in zip(self.used_parameters(), vector):
            p.data.add_(R, v)
            
        self.g_softmax(self.temperature)
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        grads_p = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.used_parameters(), vector):
            p.data.sub_(2*R, v)
        
        self.g_softmax(self.temperature)
        inferences = self(x)
        loss = self.criterion(inferences, labels.float())
        grads_n = torch.autograd.grad(loss, self.arch_parameters())

        for p,v in zip(self.used_parameters(), vector):
            p.data.add_(R, v)

        return [(i-y).div_(2*R) for i,y in zip(grads_p,grads_n)]



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
        self.quatization.update_cb_index(raw_weight=self.weigt_on_cpu, flag=self.prior_flag)

    # def update_b(self, choice=None):
    #     b2 = time.time()
    #     self.codebook_on_cpu = np.float32(self.quatization.codebooks.weight.data.cpu())
        
    #     plen = int(self.dim/self.M)
    #     for i in range(self.field_len):
    #         begin = i*self.K
    #         for k in range(1, len(ACTION)):
    #             if self.prior_flag[i, k] != 1:
    #                 continue
    #             end = begin + ACTION[k]
    #             for j in range(self.M):
    #                 codewords = self.codebook_on_cpu[begin:end, j*plen:j*plen+plen]
    #                 ind, _ = vq(self.weigt_on_cpu[self.offsets[i]:self.offsets[i+1], j*self.q_size:j*self.q_size+self.q_size], codewords)
    #                 self.quatization.cb_index[k].weight.data[self.offsets[i]:self.offsets[i+1],j] = torch.from_numpy(ind).to(self.device, dtype=torch.long)


    def calcu_distance_field(self):
        plen = int(self.dim/self.M)
        used_ind = self.arch_argmax()
        # print(used_ind)
        distance = torch.ones(self.field_len)
        w_distance = torch.ones(self.field_len)
        # distance_max = torch.ones(self.field_len)
        for i in range(self.field_len):
            material = self.embedding.embedding.weight.data[self.offsets[i]: self.offsets[i+1], ]
            # ind = self.quatization.cb_index[used_ind[i]].weight.data[self.offsets[i]: self.offsets[i+1], ] + i*self.K
            if used_ind[i] == 0:
                distance[i] = 0
                continue
            else:
                ind = self.quatization.cb_index[used_ind[i]].weight.data[self.offsets[i]: self.offsets[i+1], ] + i*ACTION[used_ind[i]]
            cluster_result = torch.ones_like(material)
            for j in range(self.M):
                # cluster_result[:, j*plen:j*plen+plen] = self.quatization.codebooks(ind[:, j])[:, j*plen:j*plen+plen]
                cluster_result[:, j*plen:j*plen+plen] = self.quatization.codebooks[used_ind[i]](ind[:, j])[:, j*plen:j*plen+plen]
            dis = F.pairwise_distance(material, cluster_result, p=2)
            distance[i] = dis.mean()
            w_distance[i] = (dis*self.popular[self.offsets[i]:self.offsets[i+1]]).sum()# distance_max[i] = F.pairwise_distance(material, cluster_result, p=2).max()
        # print(distance)
        # print(distance_max)
        return distance, w_distance
    



