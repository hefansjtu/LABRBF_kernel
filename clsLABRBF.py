import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class LABRBF(nn.Module):
    def __init__(self, x_sv, y_sv, weight_ini, lamda=0.001):
        super(LABRBF, self).__init__()

        self.num_sv = x_sv.shape[1]
        self.feature_dim = x_sv.shape[0]
        self.lamda = torch.tensor(lamda).float()
        self.alpha = torch.ones(1, self.num_sv)
        self.beta = y_sv.mean()
        self.x_sv = x_sv
        self.y_sv = y_sv.reshape(-1)
        self.weight = torch.nn.Parameter(torch.FloatTensor(self.feature_dim, self.num_sv), requires_grad=True)
        self.weight.data = weight_ini
        self.device = x_sv.device
        # calculate the initial inner-product
        hat_x_sv = self.x_sv.repeat(self.num_sv, 1, 1)
        self.sv_gram = hat_x_sv - torch.transpose(hat_x_sv, 0, 2)

        print('Number of trainable parameters:', sum(p.numel() for p in self.parameters() if p.requires_grad) )

    def forward(self, x_train):
        assert x_train.shape[0] == self.x_sv.shape[0], 'but found {} and {}'.format(x_train.shape[0],
                                                                                     self.x_sv.shape[0])

        self.alpha = self.update_alpha()

        # x_train_kernel = Gau_kernel_theta(x_train, self.x_sv, self.weight)
        x_train_kernel = Lap_kernel_theta(x_train, self.x_sv, self.weight)

        y_pred_train = self.alpha @ x_train_kernel + self.beta  # (1, N)

        return y_pred_train

    def update_alpha(self):
        if self.training:
            # x_sv_kernel = Gau_kernel_theta(self.x_sv, self.x_sv, self.weight)
            x_sv_kernel = Lap_kernel_theta(self.x_sv, self.x_sv, self.weight)
            ele1 = self.y_sv.reshape(1, self.num_sv) - self.beta.repeat(1, self.num_sv)  # y_mean
            ele2 = 1 * x_sv_kernel + self.lamda * torch.eye(self.num_sv, self.num_sv,
                                                            device=self.device, dtype=torch.float)
            alpha = torch.matmul(ele1, torch.linalg.inv(ele2)) * 1
        else:
            alpha = self.alpha
        return alpha


    @staticmethod
    def mae_loss(pred, target):
        # MAE
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

def euclidean_dis_theta(x, x_sv, w):
    x_sv = (x_sv * w).T  # N_sv, F
    x_sv_norm = torch.sum(x_sv ** 2, dim=1, keepdim=True)  # N_sv, 1
    inner_prod = (x_sv * w.T) @ x  # N_sv, N
    x = x.unsqueeze(0)  # 1,F,N
    w = w.T.unsqueeze(2)  # N_sv,F,1
    x_norm = torch.sum((x * w) ** 2, dim=1, keepdim=False)  # N_sv, N
    dis = x_sv_norm - 2 * inner_prod + x_norm
    return dis

def Gau_kernel_theta(x, x_sv, w):
    f_dim = x_sv.shape[0]
    kernel_mat = torch.exp(-1 * euclidean_dis_theta(x, x_sv, w))
    return kernel_mat

def Lap_kernel_theta(x, x_sv, w):
    w = w.T.unsqueeze(2)
    x_sv = x_sv.T.unsqueeze(2)
    x = x.unsqueeze(0)
    dis_mat = torch.norm(w * (x - x_sv), dim=1, keepdim=False)
    kernel_mat = torch.exp(-1 * dis_mat)
    return kernel_mat
