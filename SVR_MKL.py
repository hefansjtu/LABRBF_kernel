import torch
import torch.nn as nn
import numpy as np
# from cvxopt import normal
# from cvxopt.modeling import variable, op, max, sum
from sklearn.svm import SVR
from data_reg import CosineDataset, Yacht, Concrete, Tecator, \
    YearPredict, Comp_activ, Parkinson, SML, SkillCraft, WineQuality,Airfoil
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random

#
# old = torch.load('old/old.pth')['b']
# new = torch.load('new.pth')['b']
#
# print(old == new.cpu())
# print((old==new.cpu()).all())
# import pdb
# pdb.set_trace()


class EasyMKL(nn.Module):
    def __init__(self, x_train, y_train):
        super(EasyMKL, self).__init__()

        self.device = x_train.device
        self.num_samples = x_train.shape[1]
        self.feature_dim = x_train.shape[0]
        self.x_train = x_train
        self.y_train = y_train.reshape(-1)
        self.sigmas = torch.tensor([100,10,5,1, 0.5, 0.1, 0.01,0.001], dtype=torch.float)/self.feature_dim
        self.num_kernels = len(self.sigmas)
        self.weight = torch.tensor(1) /self.num_kernels * torch.ones(self.num_kernels,1, device=self.device, dtype=torch.double)
        self.train_ker_norm = 0.0
        self.regressor = None

        # calculate the initial inner-product
        ele0 = torch.pow(torch.linalg.norm(self.x_train, dim=0), 2)
        ele1 = torch.matmul(torch.ones(self.num_samples, 1, device=self.device, dtype=torch.double),
                            ele0.view(1, self.num_samples))
        ele2 = -2 * torch.matmul(torch.transpose(self.x_train, 0, 1), self.x_train)
        self.train_gram = ele1 + ele2 + torch.transpose(ele1, 0, 1)

        self.opt_alpha()

    def opt_alpha(self):
        N = self.num_samples
        x_train_kernel = torch.zeros(N, N, device=self.device, dtype=torch.double)
        # N, N
        for k_iter in range(self.num_kernels):
            x_train_kernel = x_train_kernel + self.weight[k_iter] * torch.exp(-1.0 * self.sigmas[k_iter] * self.train_gram)
        self.regressor = SVR(C=1e1, kernel='precomputed', epsilon=0.01)
        self.regressor.fit(x_train_kernel, self.y_train)
        print('the support number is:', len(self.regressor.support_))
        alpha_tmp = np.zeros((N,))
        alpha_tmp[self.regressor.support_] = self.regressor.dual_coef_
        alpha = torch.from_numpy(alpha_tmp)
        for k_iter in range(self.num_kernels):
            k_tmp = torch.exp(-1.0 * self.sigmas[k_iter] * self.train_gram)
            self.weight[k_iter] = torch.matmul(torch.matmul(alpha.T, k_tmp), alpha)
        self.weight = self.weight / torch.sum(self.weight)
        print(self.weight)
        x_train_kernel = torch.zeros(N, N, device=self.device, dtype=torch.double)
        for k_iter in range(self.num_kernels):
            x_train_kernel = x_train_kernel + self.weight[k_iter] * torch.exp(-1.0 * self.sigmas[k_iter] * self.train_gram)
        self.regressor.fit(x_train_kernel, self.y_train)

    def forward(self, x_val):
        assert x_val.shape[0] == self.x_train.shape[0], 'but found {} and {}'.format(x_val.shape[0],
                                                                                     self.x_train.shape[0])
        device = x_val.device
        N_val = x_val.shape[1]
        N = self.num_samples
        M = self.x_train.shape[0]

        ele0 = torch.pow(torch.linalg.norm(self.x_train, dim=0), 2)
        ele1 = torch.matmul(torch.ones(N_val, 1, device=self.device, dtype=torch.double),
                            ele0.view(1, self.num_samples))
        ele2 = -2 * torch.matmul(torch.transpose(x_val, 0, 1), self.x_train)
        ele4 = torch.pow(torch.linalg.norm(x_val, dim=0), 2)
        ele3 = torch.matmul(ele4.view(N_val, 1),
                            torch.ones(1, self.num_samples, device=self.device, dtype=torch.double))
        x_val_kernel = torch.zeros(N_val, N, device=self.device, dtype=torch.double)
        for k_iter in range(self.num_kernels):
            x_val_kernel = x_val_kernel + self.weight[k_iter] * torch.exp(
                -1.0 * self.sigmas[k_iter] * (ele1 + ele2 + ele3))

        y_tmp = self.regressor.predict(x_val_kernel)
        y_tmp[y_tmp > 1] = 1
        y_tmp[y_tmp < -1] = -1

        y_pred_val = torch.from_numpy(y_tmp)

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
    #
    batch = 128
    adp_model.eval()
    pred_y_list = []
    cnt = 0
    data_y = data_y.reshape(-1)
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(adp_model(x_val=data_x[:, cnt: cnt + batch]).detach())
        else:
            pred_y_list.append(adp_model(x_val=data_x[:, cnt:]).detach())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list)
    pred_y = pred_y.reshape(-1)
    print('\t\t the rsse loss: ', format(adp_model.rsse_loss(pred=pred_y, target=data_y)))
    # print('\t\t the mae loss:', format(adp_model.mad_loss(pred=pred_y, target=data_y))
    print('\t\t the rmse loss:', format(adp_model.rmse_loss(pred=pred_y, target=data_y) * delta_y / 2))
    plt.figure(2)
    plt.plot(data_y, 'r+', label='test data', markersize=13)
    plt.plot(pred_y, 'b2', label='pred data', markersize=13)
    plt.show()
    return adp_model.rsse_loss(pred=pred_y, target=data_y)



if __name__ == '__main__':
    acc = []
    for iter in range(1):
        seed = iter
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        # build the dataset
        dataset_con = Comp_activ() #Airfoil()#SML() # Tecator()  #Yacht()#Parkinson() #Concrete() #SkillCraft() #WineQuality() #CosineDataset()  #

        data_train_x, data_train_y = dataset_con.get_sv_data()
        data_val_x, data_val_y = dataset_con.get_val_data()
        data_test_x, data_test_y = dataset_con.get_test_data()

        train_x = torch.cat([data_train_x, data_val_x], dim=1)
        train_y = torch.cat([data_train_y, data_val_y], dim=0)
        M = train_x.shape[0]
        train_x = train_x.double()
        train_y = train_y.double()
        data_test_x = data_test_x.double()
        data_test_y = data_test_y.double()

        global delta_y
        delta_y = 2
        # normalization
        delta_y = train_y.max() - train_y.min()
        print(delta_y)
        mid_y = (train_y.max() + train_y.min()) / 2
        train_y = (train_y - mid_y) / delta_y * 2
        data_test_y = (data_test_y - mid_y) / delta_y * 2

        # for fea_id in range(M):
        #     max_x = train_x[fea_id, :].max()#torch.quantile(train_x[fea_id,:], 0.75) #
        #     min_x = train_x[fea_id, :].min()#torch.quantile(train_x[fea_id, :], 0.25) #
        #     delta_x = max_x - min_x
        #     # print(delta_x)
        #     if delta_x == 0:
        #         delta_x = 1
        #     mid_x = (max_x - min_x) / 2
        #     train_x[fea_id, :] = (train_x[fea_id, :] - mid_x) / delta_x
        #     data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mid_x) / delta_x
        # train_x = train_x * 1
        # data_test_x = data_test_x * 1

        for fea_id in range(M):
            mean_x = torch.mean(train_x[fea_id, :])
            std_x = torch.std(train_x[fea_id,:])
            if std_x == 0:
                std_x = 1
            train_x[fea_id, :] = (train_x[fea_id, :] - mean_x)/std_x
            data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mean_x)/std_x

        regressor = EasyMKL(x_train=train_x, y_train=train_y)
        regressor.eval()

        with torch.no_grad():
            print('the train error rate loss: ')
            testFun(regressor, train_x, train_y)

            print('the test error rate loss: ')
            acc.append(testFun(regressor, data_test_x, data_test_y).cpu())

    print('mean:', np.mean(acc), 'std:', np.std(acc))