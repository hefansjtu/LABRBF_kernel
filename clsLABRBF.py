import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LABRBF(nn.Module):
    def __init__(self, x_sv, y_sv, weight_ini, lamda=0.001):
        super(LABRBF, self).__init__()

        self.num_samples = x_sv.shape[1]
        self.feature_dim = x_sv.shape[0]
        self.lamda = torch.tensor(lamda).float().cuda()
        self.alpha = torch.ones(1, self.num_samples).cuda()
        self.beta = y_sv.mean() #torch.tensor(0).cuda().float()  #
        self.x_sv = x_sv
        self.y_sv = y_sv.reshape(-1)
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.feature_dim, self.num_samples), requires_grad=True)
        self.weight.data = weight_ini
        # self.weight.data = torch.sqrt(torch.tensor(0.1).float() / self.x_train.shape[0]) * torch.ones(self.feature_dim, self.num_samples)
        self.sv_ker_norm = 0.0

        # calculate the initial inner-product
        hat_x_sv = self.x_sv.repeat(self.num_samples, 1, 1)
        self.sv_gram = hat_x_sv - torch.transpose(hat_x_sv, 0, 2)

    def forward(self, x_train):
        assert x_train.shape[0] == self.x_sv.shape[0], 'but found {} and {}'.format(x_train.shape[0],
                                                                                     self.x_sv.shape[0])
        bias = 0.0001
        device = x_train.device
        N_val = x_train.shape[1]
        N = self.num_samples
        M = self.x_sv.shape[0]

        # hat_w_train = F.relu(self.weight) + bias
        hat_w_train = self.weight + bias
        hat_w_train = hat_w_train.repeat(N, 1, 1)
        hat_w_train = torch.transpose(hat_w_train, 0, 2)

         # N, N
        # x_train_kernel = torch.exp(-1.0 * torch.pow(torch.linalg.norm(hat_w_train * self.train_gram, dim=1), 2))
        x_sv_kernel = torch.exp(-1.0 * torch.linalg.norm(hat_w_train * self.sv_gram, dim=1))
        # calculate the initial inner-product of the val data
        hat_x_sv = self.x_sv.repeat(N_val, 1, 1)
        hat_x_sv = torch.transpose(hat_x_sv, 0, 2)  # N, M, N_val
        hat_x_train = x_train.repeat(N, 1, 1)  # N, M,  N_val

        hat_w = F.relu(self.weight) + bias
        hat_w = hat_w.repeat(N_val, 1, 1)
        hat_w = torch.transpose(hat_w, 0, 2)

        train_sv_gram = hat_x_train - hat_x_sv
        # x_val_kernel = torch.exp(-1.0 * torch.pow(torch.linalg.norm(hat_w * val_train_gram, dim=1), 2))  # N, N_val
        x_train_kernel = torch.exp(-1.0 * torch.linalg.norm(hat_w * train_sv_gram, dim=1))  # N, N_val

        ele1 = self.y_sv.reshape(1, N) - self.beta.repeat(1, N)  # y_mean
        ele2 = 1 * x_sv_kernel + self.lamda * torch.eye(N, N, device=device, dtype=torch.double)
        # self.alpha = torch.linalg.solve(ele2, ele1.T).T   # 1, N
        self.alpha = torch.matmul(ele1, torch.linalg.inv(ele2)) * 1
        self.sv_ker_norm = torch.matmul(torch.matmul(self.alpha, x_sv_kernel), torch.transpose(self.alpha, 1, 0))
        self.alpha_norm = torch.norm(self.alpha)
        y_pred_train = torch.matmul(self.alpha, x_train_kernel) + self.beta  # (1, N)

        return y_pred_train

    def weight_loss(self, ll):
        loss = ll * self.weight.norm()
        return loss

    @staticmethod
    def mad_loss(pred, target):
        # mean absolute deviation
        loss = (torch.abs(pred.reshape(-1) - target.reshape(-1))).sum() / target.shape[0]
        return loss

    @staticmethod
    def rsse_loss(pred, target):
        # RSSE
        tmp = ((target.reshape(-1) - target.mean()) ** 2).sum()
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / tmp
        return loss

    @staticmethod
    def mse_loss(pred, target):
        # MSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        return loss

    @staticmethod
    def rmse_loss(pred, target):
        # RMSE
        loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / target.shape[0]
        loss = torch.sqrt(loss)
        return loss

    def norm_print(self):
        print("train kernel norm: ", format(self.sv_ker_norm))
        print("alpha norm:  ", format(self.alpha_norm))
        print("weight norm:", format(self.weight.norm()))
