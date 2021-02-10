import torch
from sklearn.metrics import roc_auc_score, log_loss
from torch.utils.data import DataLoader

import os
import sys
sys.path.append("/data2/home/gangwei/project/xlightfm")

from utils import md_solver

from dataset.avazu import AvazuDataset
from dataset.criteo import CriteoDataset
from dataset.movielens import MovieLens1MDataset, MovieLens20MDataset

from model.fm import FactorizationMachineModel
from model.qfm import QuantizationFactorizationMachine
from model.qfm_gs import GumbelSoftmaxQuantizationFM
from model.nfm import NeuralFactorizationMachineModel
from model.deepfm import DeepFactorizationMachineModel
from model.qdeepfm import QuatDeepFactorizationMachineModel
from model.qnfm import QuatNeuralFactorizationMachineModel


import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import datetime
import time
import tqdm


class Train(object):
    def __init__(self, params=None):
        self.params = params
        
        self.dim = self.params["dim"]
        self.epoch = self.params["epoch"]
        self.q_loss = self.params["q_loss"]
        self.method = self.params["method"]
        self.model_name = self.params["model"]
        self.data_size = self.params["data_size"]
        self.batch_size = self.params["batch_size"]
        self.nas_method = self.params["nas_method"]
        self.feature_size = self.params["feature_size"]
        self.weight_decay = self.params["weight_decay"]
        self.learning_rate = self.params["learning_rate"]
        self.test_data_size = self.params["test_data_size"]
        self.device = torch.device(self.params["device"])

        self.step = 0
        self.temperature = 2.5 * np.exp(-0.036 * self.step)
        self.real_loss = 0
        self.loss_record = []
        self.loss_curve = []
        self.dis_curve = []
        self.i_dis_curve = []
        self.w_dis_curve = []
        self.wi_dis_curve = []
        self.type_record = []
        self.cost = [0 for _ in range(10)]

    def set_dataloader(self, name=None):
        dataset = None
        path = self.params["dataset_path"]
        cache_path = self.params["cache_path"]
        if name == 'movielens1M':
            dataset = MovieLens1MDataset(path, cache_path)
        elif name == 'movielens20M':
            dataset = MovieLens20MDataset(path, cache_path)
        elif name == 'criteo':
            dataset = CriteoDataset(path, cache_path)
        elif name == 'avazu':
            dataset = AvazuDataset(path, cache_path)
        else:
            raise ValueError('unknown dataset name: ' + name)

        self.params["field_dims"] = dataset.field_dims
        train_length = int(len(dataset) * 0.8)
        valid_length = int(len(dataset) * 0.1)
        test_length = len(dataset) - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length), generator=torch.Generator().manual_seed(19))
        self.train_data_loader = DataLoader(train_dataset, shuffle=True, batch_size=self.batch_size, num_workers=32)
        # self.train_data_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=32)
        self.valid_data_loader = DataLoader(valid_dataset, batch_size=self.batch_size, num_workers=32)
        self.test_data_loader = DataLoader(test_dataset, batch_size=20480, num_workers=32)

        # if self.q_loss == 2:
        # self.dim_importance = md_solver(torch.Tensor(dataset.field_dims.astype(np.float32)), 0.3, d0=self.dim, round_dim=False)
        # self.dim_importance = 10/self.dim_importance
        # self.dim_importance = (self.dim_importance - self.dim_importance.min())/(self.dim_importance.max() - self.dim_importance.min())
        self.dim_importance = md_solver(torch.Tensor(dataset.field_dims.astype(np.float32)), 0.15, d0=self.dim, round_dim=False)
        self.dim_importance = (self.dim_importance - self.dim_importance.min())/(self.dim_importance.max() - self.dim_importance.min())
        print(self.dim_importance)
        # for step, (f, t) in enumerate(self.train_data_loader):
        #     # users_valid, items_vali = next(iter(self.test_data_loader))
        #     print(step)
        # exit()

    def initial_criterion(self, loss_name):
        if loss_name == "mse":
            return torch.nn.MSELoss()
        else:
            return torch.nn.BCEWithLogitsLoss()

    def initial_optimizer(self, nas=False):
        if nas:
            return torch.optim.Adam(params=self.model.parameters(),
                                    lr=self.learning_rate, weight_decay=self.weight_decay), torch.optim.Adam(params=self.model.arch_parameters(),
                                    lr=self.learning_rate,weight_decay=self.weight_decay)
        else:
            return torch.optim.Adam(params=self.model.parameters(),
                                    lr=self.learning_rate, weight_decay=self.weight_decay), None

    def initial_model(self, model_name, optimizer='adam', loss='bce'):
        self.model_name = model_name
        if model_name == "fm":
            self.model = FactorizationMachineModel(self.params).to(self.device)
        elif model_name == "qrfm":
            self.model = QRFactorizationMachineModel(self.params).to(self.device)
        elif model_name == "mdfm":
            self.model = MDFactorizationMachineModel(self.params).to(self.device)
        elif model_name == "dfm":
            self.model = DFactorizationMachineModel(self.params).to(self.device)
        elif model_name == "dhefm":
            self.model = DHEFactorizationMachineModel(self.params).to(self.device)
        elif model_name == "nfm":
            self.model = NeuralFactorizationMachineModel(self.params, (64,), (0.2,0.2)).to(self.device)
        elif model_name =="deepfm":
            self.model = DeepFactorizationMachineModel(self.params, (16,16), 0.2).to(self.device)
        else:
           raise ValueError('unknown model name: ' + model_name)
        
        self.criterion = self.initial_criterion(loss)


        self.optimizer, _ = self.initial_optimizer()
 
    def initial_pq_model(self, pre_state_dict, optimizer='adam', loss='bce', pre_quat=1):
        if self.model_name == "fm":
            if self.nas_method == "bc":
                self.model = BinRecQuantizationFM(self.params).to(self.device)
            elif self.nas_method == "gs":
                self.model = GumbelSoftmaxQuantizationFM(self.params).to(self.device)
                
                print(self.model.genotype())
            else:
                self.model = QuantizationFactorizationMachine(self.params).to(self.device)
            # self.model = QFM(self.params).to(self.device)
        elif self.model_name == "nfm":
            self.model = QuatNeuralFactorizationMachineModel(self.params, (64,), (0.2,0.2)).to(self.device)
        elif self.model_name =="deepfm":
            self.model = QuatDeepFactorizationMachineModel(self.params, (16,16), 0.2).to(self.device)
        else:
           raise ValueError('unknown model name with pq func: ' + self.model_name)

        begin = time.time()
        self.model.copy(pre_state_dict)
        # print(self.model.state_dict())
        if pre_quat == 1:
            self.model.quat_copy()
            if self.q_loss != 0:
                dis, w_dis = self.model.calcu_distance_field()
                dis_loss  = dis.mean().cpu().item()
                i_dis_loss = (dis*self.dim_importance).mean().cpu().item()
                print("dis loss:", dis_loss , "importance dis loss:", i_dis_loss)
                w_dis_loss  = w_dis.mean().cpu().item()
                wi_dis_loss = (w_dis*self.dim_importance).mean().cpu().item()
                print("weight dis loss:", w_dis_loss , "weighted importance dis loss:", wi_dis_loss)
            # torch.nn.init.xavier_uniform_(self.model.quatization.codebooks.weight)
        else:
            self.model.Embedding_pQ()
            self.model.quatization.save()
        print("-----finish initial product quatization model------")

        self.criterion = self.initial_criterion(loss)
        self.optimizer, self.arch_optimizer = self.initial_optimizer(self.nas_method)

        # self.model.calcu_distance_field()

    def train(self, data_loader):
        self.model.train()
        total_loss = 0
        real_loss = 0
        update_size= 2000000
        data_cnt = 0

        begin1 = time.time()
        tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
        for i, (fields, target) in enumerate(tk0):
            begin = time.time()
            data_cnt += len(target)
            if self.data_size < i*self.batch_size and self.data_size != -1:
                break
            if self.model_name == "pq" and data_cnt>update_size:
                # self.model.update_b()
                data_cnt = 0
            fields = fields.to(device=self.device, dtype=torch.long)
            target = target.to(self.device)
                
            if self.nas_method == "bc":
                loss_valid = self.model.step(fields, target, self.criterion, self.arch_optimizer)
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                self.model.binarize()
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                # self.model.zero_grad()
                self.model.recover()
            elif self.nas_method == "gs":
                loss_valid = self.model.step(fields, target, self.criterion, self.temperature, self.arch_optimizer)
                self.optimizer.zero_grad()
                self.arch_optimizer.zero_grad()
                self.model.g_softmax(self.temperature)
                y = self.model(fields)
                loss = self.criterion(y, target.float())
            else:
                y = self.model(fields)
                loss = self.criterion(y, target.float())
                real_loss += loss.item()
                # dis, w_dis = self.model.calcu_distance_field()
                # dis_loss  = dis.mean().cpu().item()
                # i_dis_loss = (dis*self.dim_importance).mean().cpu().item()
                # # print("dis loss:", dis_loss , "importance dis loss:", i_dis_loss)
                # w_dis_loss  = w_dis.mean().cpu().item()
                # wi_dis_loss = (w_dis*self.dim_importance).mean().cpu().item()
                # # print("weight dis loss:", w_dis_loss , "weighted importance dis loss:", wi_dis_loss)
                # self.loss_curve.append(loss.cpu().item())
                # self.dis_curve.append(dis_loss)
                # self.i_dis_curve.append(i_dis_loss)
                # self.w_dis_curve.append(w_dis_loss)
                # self.wi_dis_curve.append(wi_dis_loss)
                # print(dis_loss)
                if self.q_loss == 1:
                    # loss 1
                    if loss < 0.47:
                        dis,_ = self.model.calcu_distance_field()
                        dis_loss = dis.mean()
                        # print(dis_loss)
                    
                        loss += dis_loss/3.0
                elif self.q_loss == 2:
                    # loss 1
                    if loss < 0.47:
                        dis,_ = self.model.calcu_distance_field()
                        dis_loss = (dis*self.dim_importance).mean()
                        # print(dis_loss)
                        loss += dis_loss*0.5
                        # loss = loss + np.exp((0.377-loss.cpu().item())*3)*dis_loss
                        # loss = loss + torch.exp(500*(0.377-loss.detach()))*dis_loss
                elif self.q_loss == 3:
                    # loss 1
                    if loss < 0.47:
                        dis = self.model.calcu_weighted_distance_field()
                        dis_loss = dis.mean()
                        # print(dis_loss)
                    
                        loss += dis_loss/5.0
                elif self.q_loss == 4:
                    # loss 1
                    if loss < 0.47:
                        dis = self.model.calcu_weighted_distance_field()
                        dis_loss = (dis*self.dim_importance).mean()
                        # print(dis_loss)
                        loss += dis_loss*1.2
            self.model.zero_grad()
            loss.backward()

            if self.nas_method is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optimizer.step()
                self.model.clip()
            else:
                self.optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 100 == 0:
                self.step += 1
                self.temperature = 1.5 * np.exp(-0.024 * self.step)
                tk0.set_postfix(loss=total_loss / 100)
                # self.loss_record.append(total_loss / 100)
                # self.loss_curve.append(real_loss / 100)
                # dis, w_dis = self.model.calcu_distance_field()
                # dis_loss  = dis.mean().cpu().item()
                # i_dis_loss = (dis*self.dim_importance).mean().cpu().item()
                # print("loss:", real_loss / 100, "dis loss:", dis_loss , "importance dis loss:", i_dis_loss)
                # w_dis_loss  = w_dis.mean().cpu().item()
                # wi_dis_loss = (w_dis*self.dim_importance).mean().cpu().item()
                # print("weight dis loss:", w_dis_loss , "weighted importance dis loss:", wi_dis_loss)

                if self.nas_method is not None:
                    # print(self.model._arch_parameters.grad)
                    print(self.model._arch_parameters.data*self.model.prior_flag)
                    print(self.model.genotype())
                total_loss = 0
                real_loss = 0
            self.cost[5] += time.time()-begin
        # if self.model_name == "pq":
        #     self.model.update_b()
        if self.q_loss != 0:
            dis, w_dis = self.model.calcu_distance_field()
            dis_loss  = dis.mean().cpu().item()
            i_dis_loss = (dis*self.dim_importance).mean().cpu().item()
            print("dis loss:", dis_loss , "importance dis loss:", i_dis_loss)
            w_dis_loss  = w_dis.mean().cpu().item()
            wi_dis_loss = (w_dis*self.dim_importance).mean().cpu().item()
            print("weight dis loss:", w_dis_loss , "weighted importance dis loss:", wi_dis_loss)


        # self.model.calcu_distance_field()
        # self.cost[0] += time.time()-begin1
            # print(self.model.cost)
            # print(self.model.quatization.cost)
        # for k in self.cost:
        #     print(round(k,5), end=" ")
        # print(" ")
        self.cost = [0 for _ in range(10)]
        # print('---- loss:', np.mean(loss_record))    
        # return np.mean(loss_record)

    def test(self, data_loader):
        self.model.eval()
        targets, predicts = list(), list()
        loss_record = 0
        cnt = 0
        if self.nas_method == "bc":
            self.model.binarize()
        elif self.nas_method == "gs":
            self.model.arch_argmax()
        with torch.no_grad():
            for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)):
                if self.test_data_size < i*self.batch_size and self.test_data_size != -1:
                    break
                # print(fields.shape)
                fields = fields.to(device=self.device, dtype=torch.long)
                target = target.to(self.device)
                y = self.model(fields)
                loss_record += self.criterion(y, target.float()).item()
                cnt += 1
                targets.extend(target.tolist())
                predicts.extend(y.tolist())
                # for i in fields.tolist():
                #     id_list.append(",".join([str(s) for s in i[:68]]))
        if self.nas_method == "bc":
            self.model.recover()
        # print("average loss: ", loss_record/cnt)
        auc = roc_auc_score(targets, predicts)
        return auc, loss_record/cnt

    def save(self, path):
        # print(self.model.state_dict())
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        pre_state_dict = torch.load(path)
        # print(pre_state_dict)
        self.model.copy(pre_state_dict)
        self.model.eval()

    def early_stopper(self, num_trials, save_path, record):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path
        self.record = record

    def is_continuable(self, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            if self.record == 1:
                self.save(self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default="avazu")
    parser.add_argument('--dim', type=int, default=40)
    parser.add_argument('--decay', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--learning', type=float, default=0.001)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--dataset", type=int, default=-1)
    parser.add_argument("--K", type=int, default=256)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--test_dataset", type=int, default=-1)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--optimizer", type= str, default="adam")
    parser.add_argument("--loss", type= str, default="bce")
    parser.add_argument("--model", type=str, default="fm")
    parser.add_argument("--method", type=str, default="None")
    parser.add_argument("--machine", type=str, default="1")    
    parser.add_argument("--nas_method", type=str, default= None)
    parser.add_argument("--record", type=int, default=0)
    parser.add_argument("--pre_train", type=int, default=1)
    parser.add_argument("--pre_quat", type=int, default=1)
    parser.add_argument("--share", type=int, default=1)    
    parser.add_argument("--q_loss", type=int, default=0)
    args = parser.parse_args()

    if args.data_name == "avazu":
        dataset_path = "/data2/home/gangwei/project/dataset/Avazu_raw/train"
        cache_path = "/data2/home/gangwei/project/pytorch-fm/project/.avazu"
        popular_path = "/data2/home/gangwei/project/xlightfm/dataset/avazu_popular.pkl"
    elif args.data_name == "criteo":
        dataset_path = "/data2/home/gangwei/project/dataset/Criteo/train.txt"
        cache_path = "/data2/home/gangwei/project/pytorch-fm/project/.criteo"
        popular_path = "/data2/home/gangwei/project/xlightfm/dataset/criteo_popular.pkl"

    params = {"M":              args.M,
              "K":              args.K,
              "device":         args.device,
              "dim":            args.dim,
              "weight_decay":   args.decay,
              "epoch":          args.epoch,
              'method':         args.method,
              "cache_path":     cache_path,
              "dataset_path":   dataset_path,
              "data_size":      args.dataset,
              "share":          args.share,
              "nas_method":     args.nas_method,
              "popular_path":   popular_path,
              "test_data_size": args.test_dataset,
              "batch_size":     args.batch_size,
              "optimizer":      args.optimizer,
              "loss":           args.loss,
              "q_loss":         args.q_loss,
              "model":          args.model,
              "learning_rate":  args.learning,
              'feature_size':   dataset_path+args.data_name+"/feature/feature_size.pkl"}

    os.environ["CUDA_VISIBLE_DEVICES"] = args.machine

    ISOTIMEFORMAT = '%m%d-%H%M%S'
    timestamp = str(datetime.datetime.now().strftime(ISOTIMEFORMAT))
    import json
    params_str = json.dumps(params)

    result = []
    score = []

    for i in range(args.times):
        score.append([])

        task = Train(params)
        task.set_dataloader(name=args.data_name)

        if args.data_name == "avazu":
            model_path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/fm_32.pt"  
        elif args.data_name == "criteo":
            model_path = "/data2/home/gangwei/project/pytorch-fm/chkpt/criteo/fm_32_1.pt"
        # model_path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/deepfm_32_2.pt" 
        # model_path = "/data2/home/gangwei/project/pytorch-fm/chkpt/avazu/nfm_32_"+str(i)+"_random.pt" 
        if args.pre_train == 0:
            task.initial_model(args.model, loss=args.loss)
            # task.load(model_path)
        else:
            pre_state_dict = torch.load(model_path)
            task.initial_pq_model(pre_state_dict, loss=args.loss, pre_quat=args.pre_quat)
            auc, logloss = task.test(task.test_data_loader)
            print(auc, logloss)
            # score[i].ap[pend((auc, logloss))
                
        
        task.early_stopper(num_trials=10, save_path=model_path, record=args.record)
        for j in range(args.epoch):
            task.train(task.train_data_loader)
            auc, logloss = task.test(task.test_data_loader)
            print('epoch:', j, 'validation: auc:', auc, 'logloss: ', logloss)
            score[i].append((auc, logloss))
            if not task.is_continuable(auc):
                print(f'validation: best auc: {task.best_accuracy}')
                break

            if j % 10 == 9:
                filename = os.path.join("/data2/home/gangwei/project/pytorch-fm/result3",args.model+"-"+timestamp+'.txt')
                with open(filename, 'w') as f:
                    f.write(params_str)
                    f.write('\n')
                    for i in range(len(score)):
                        auc = [str(reward[0]) for reward in score[i]]
                        logloss = [str(reward[1]) for reward in score[i]]
                        f.write("time"+str(i)+": "+" ".join(auc)+"\n")
                        f.write("time"+str(i)+": "+" ".join(logloss)+"\n")

        result.append(task.best_accuracy)
        filename = os.path.join("/data2/home/gangwei/project/pytorch-fm/result3",args.model+"-"+timestamp+'.txt')
        with open(filename, 'w') as f:
            f.write(params_str)
            f.write('\n')
            for i in range(len(score)):
                auc = [str(reward[0]) for reward in score[i]]
                logloss = [str(reward[1]) for reward in score[i]]
                f.write("time"+str(i)+": "+" ".join(auc)+"\n")
                f.write("time"+str(i)+": "+" ".join(logloss)+"\n")
            s = [str(a) for a in task.loss_curve]
            f.write("loss: "+" ".join(s)+"\n")
            s = [str(a) for a in task.dis_curve]
            f.write("dis: "+" ".join(s)+"\n")
            s = [str(a) for a in task.i_dis_curve]
            f.write("idis: "+" ".join(s)+"\n")
            s = [str(a) for a in task.w_dis_curve]
            f.write("wdis: "+" ".join(s)+"\n")
            s = [str(a) for a in task.wi_dis_curve]
            f.write("widis: "+" ".join(s)+"\n")
        print("result: ", result)
    