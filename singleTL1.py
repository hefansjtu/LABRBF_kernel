import torch
import torch.nn as nn
import numpy as np
import array
import torch.optim as optim
import argparse
import random
from model_adap_reg import AdaptiveKernel
from model_deep_reg import DeepKernel
from data_reg import CosineDataset, Yacht,  Airfoil, Tecator, \
    YearPredict, Comp_activ, Parkinson, SML, SkillCraft, WineQuality
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random
seed = 30
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
#
# old = torch.load('old/old.pth')['b']
# new = torch.load('new.pth')['b']
#
# print(old == new.cpu())
# print((old==new.cpu()).all())
# import pdb
# pdb.set_trace()


class TL1Regressor(nn.Module):
    def __init__(self, x_train, y_train, rho=1, lamda=0.1):
        super(TL1Regressor, self).__init__()

        self.num_samples = x_train.shape[1]
        self.feature_dim = x_train.shape[0]
        self.lamda = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.lamda.data = torch.tensor(lamda).float()
        self.alpha = torch.ones(1, self.num_samples).cuda()
        self.beta = torch.tensor(0.0).cuda()  # y_train.mean()
        self.x_train = x_train
        self.y_train = y_train.reshape(-1)
        self.rho = torch.tensor(rho).float() * self.feature_dim
        self.train_ker_norm = 0.0
        self.device = self.x_train.device

        # calculate the initial inner-product
        hat_x_train = self.x_train.repeat(self.num_samples, 1, 1)
        hat_x_train = torch.transpose(hat_x_train, 0, 2)
        hat_x_train = torch.transpose(hat_x_train, 1, 0)
        tmp = torch.transpose(hat_x_train, 2, 1) - hat_x_train
        self.train_gram = torch.norm(tmp, dim=0)

    def forward(self, x_val):
        assert x_val.shape[0] == self.x_train.shape[0], 'but found {} and {}'.format(x_val.shape[0],
                                                                                     self.x_train.shape[0])
        device = x_val.device
        N_val = x_val.shape[1]
        N = self.num_samples
        M = self.x_train.shape[0]

         # N, N
        x_train_kernel = torch.max(self.rho - self.train_gram, torch.tensor(0))

        # calculate the initial inner-product of the val data
        hat_x_train = self.x_train.repeat(N_val, 1, 1)
        hat_x_train = torch.transpose(hat_x_train, 0, 2)
        hat_x_train = torch.transpose(hat_x_train, 0, 1)
        hat_x_val = x_val.repeat(N, 1, 1)
        # hat_x_val = torch.reshape(x_val, (N, M, N_val))
        hat_x_val = torch.transpose(hat_x_val, 0, 1)
        x_val_gram = torch.norm(hat_x_val-hat_x_train, dim=0)  # N, N_val
        x_val_kernel = torch.max(self.rho - x_val_gram, torch.tensor(0))

        ele1 = self.y_train.reshape(1, N) - self.beta.repeat(1, N)  # y_mean
        ele2 = 1 * x_train_kernel + self.lamda * torch.eye(N, N, device=device, dtype=torch.double)
        # self.alpha = torch.linalg.solve(ele2, ele1.T).T   # 1, N
        self.alpha = torch.matmul(ele1, torch.inverse(ele2)) * 1
        self.train_ker_norm = torch.matmul(torch.matmul(self.alpha, x_train_kernel), torch.transpose(self.alpha, 1, 0))
        self.alpha_norm = torch.norm(self.alpha)
        y_pred_val = torch.matmul(self.alpha, x_val_kernel) + self.beta  # (1, N)

        return y_pred_val

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
        print("train kernel norm: ", format(self.train_ker_norm))
        print("alpha norm:  ", format(self.alpha_norm))
        print("weight norm:", format(self.weight.norm()))



def testFun(adp_model, data_x, data_y):
    #
    batch = 128
    adp_model.eval()
    pred_y_list = []
    cnt = 0
    data_y = data_y.view(len(data_y),1)
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(adp_model(x_val=data_x[:, cnt: cnt + batch]).detach())
        else:
            pred_y_list.append(adp_model(x_val=data_x[:, cnt:]).detach())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 1)
    pred_y = pred_y.view(max(pred_y.shape),1)
    print('\t\t the rsse loss: ', format(adp_model.rsse_loss(pred=pred_y, target=data_y)))
    # print('\t\t the mae loss:', format(adp_model.mad_loss(pred=pred_y, target=data_y))
    print('\t\t the rmse loss:', format(adp_model.rmse_loss(pred=pred_y, target=data_y)*delta_y/2))

    return adp_model.rmse_loss(pred=pred_y, target=data_y)*delta_y/2



if __name__ == '__main__':
    acc = []
    lam_range = [ 1, 0.1, 0.01, 0.001]  #[1, 0.1, 0.01, 0.001]
    rho_range = [10, 1,0.1]  # [10, 1,0.1]
    for iter in range(1):
        # build the dataset
        dataset_con = Comp_activ()#Airfoil()#SML() #Yacht()#Tecator()  # Parkinson() #SkillCraft() #Concrete() #Yacht()#WineQuality() #CosineDataset()  #

        data_train_x, data_train_y = dataset_con.get_sv_data()
        data_val_x, data_val_y = dataset_con.get_val_data()
        data_test_x, data_test_y = dataset_con.get_test_data()

        train_x = torch.cat([data_train_x, data_val_x], dim=1)
        train_y = torch.cat([data_train_y, data_val_y], dim=0)
        M = train_x.shape[0]

        train_x = train_x.cuda().double()
        train_y = train_y.cuda().double()
        data_test_x = data_test_x.cuda().double()
        data_test_y = data_test_y.cuda().double()

        global delta_y
        delta_y = 2
        # normalization
        # delta_y = train_y.max() - train_y.min()
        # print(delta_y)
        # mid_y = (train_y.max() + train_y.min()) / 2
        # train_y = (train_y - mid_y) / delta_y * 2
        # data_test_y = (data_test_y - mid_y) / delta_y * 2

        for fea_id in range(M):
            max_x = train_x[fea_id, :].max()#torch.quantile(train_x[fea_id,:], 0.75) #
            min_x = train_x[fea_id, :].min()#torch.quantile(train_x[fea_id, :], 0.25) #
            delta_x = max_x - min_x
            # print(delta_x)
            if delta_x == 0:
                delta_x = 1
            mid_x = (max_x - min_x) / 2
            train_x[fea_id, :] = (train_x[fea_id, :] - mid_x) / delta_x
            data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mid_x) / delta_x
        train_x = train_x * 1
        data_test_x = data_test_x * 1

        # for fea_id in range(M):
        #     mean_x = torch.mean(train_x[fea_id, :])
        #     std_x = torch.std(train_x[fea_id,:])
        #     if std_x == 0:
        #         std_x = 1
        #     train_x[fea_id, :] = (train_x[fea_id, :] - mean_x)/std_x
        #     data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mean_x)/std_x

        for rho_iter in rho_range:
            for lam_iter in lam_range:
                print('lambda:', lam_iter, 'rho: ', rho_iter)
                regressor = TL1Regressor(x_train=train_x, y_train=train_y, rho=rho_iter, lamda=lam_iter)
                regressor.cuda().double()
                regressor.eval()

                with torch.no_grad():
                    print('the train error rate loss: ')
                    testFun(regressor, train_x, train_y)

                    print('the test error rate loss: ')
                    acc.append(testFun(regressor, data_test_x, data_test_y).cpu())
    print('mean:', np.mean(acc), 'std:', np.std(acc))