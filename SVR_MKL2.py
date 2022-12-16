import torch
import torch.nn as nn
import numpy as np
# from cvxopt import normal
# from cvxopt.modeling import variable, op, max, sum
from sklearn.svm import SVR
from data_reg import CosineDataset, Yacht, Tecator, \
    Comp_activ, Parkinson, SML,Airfoil
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
        self.kernelDic = {
            'RBF': torch.tensor([10,1,0.1,0.01], dtype=torch.float)/self.feature_dim,
            'Lap': torch.tensor([0.5,0.05], dtype=torch.float)/self.feature_dim,
            'Poly': torch.tensor([1], dtype=torch.float)}
        # self.sigmas = torch.tensor([100,10,5,1, 0.5, 0.1, 0.01,0.001], dtype=torch.float)/self.feature_dim
        self.num_kernels = len(self.kernelDic['RBF']) + len(self.kernelDic['Lap']) + len(self.kernelDic['Poly'])
        self.weight = torch.tensor(1) /self.num_kernels * torch.ones(self.num_kernels, 1, device=self.device, dtype=torch.double)
        self.train_ker_norm = 0.0
        self.regressor = None

        # calculate the initial inner-product
        self.train_inner = torch.matmul(self.x_train.T, self.x_train)
        self.train_gram = torch.norm(self.x_train[:, None].T-self.x_train.T, dim=2, p=2)

        self.opt_alpha()

    def comp_kernel(self):
        self.x_train_kernel = torch.zeros(self.num_samples, self.num_samples, device=self.device, dtype=torch.double)
        w_iter = 0
        # RBF
        for k_iter in range(len(self.kernelDic['RBF'])):
            self.x_train_kernel = self.x_train_kernel + self.weight[w_iter] * torch.exp(
                -1.0 * self.kernelDic['RBF'][k_iter] * torch.pow(self.train_gram, 2))
            w_iter = w_iter + 1
        # Laplace
        for k_iter in range(len(self.kernelDic['Lap'])):
            self.x_train_kernel = self.x_train_kernel + self.weight[w_iter] * torch.exp(
                -1.0 * self.kernelDic['Lap'][k_iter] * self.train_gram)
            w_iter = w_iter + 1
        # Polynomial
        for k_iter in range(len(self.kernelDic['Poly'])):
            k_tmp = torch.pow(self.train_inner + 1, self.kernelDic['Poly'][k_iter])
            dd = torch.matmul(torch.diag(k_tmp).view(self.num_samples, 1),
                              torch.ones(1, self.num_samples, dtype=torch.double))
            k_tmp = k_tmp / dd
            self.x_train_kernel = self.x_train_kernel + self.weight[w_iter] * k_tmp
            w_iter = w_iter + 1

    def comp_weight(self, alpha):
        w_iter = 0
        # RBF
        for k_iter in range(len(self.kernelDic['RBF'])):
            k_tmp = torch.exp(-1.0 * self.kernelDic['RBF'][k_iter] * torch.pow(self.train_gram, 2))
            self.weight[w_iter] = torch.matmul(torch.matmul(alpha.T, k_tmp), alpha)
            w_iter = w_iter + 1
        # Laplace
        for k_iter in range(len(self.kernelDic['Lap'])):
            k_tmp = torch.exp(-1.0 * self.kernelDic['Lap'][k_iter] * self.train_gram)
            self.weight[w_iter] = torch.matmul(torch.matmul(alpha.T, k_tmp), alpha)
            w_iter = w_iter + 1
        # Polynomial
        for k_iter in range(len(self.kernelDic['Poly'])):
            k_tmp = torch.pow(self.train_inner + 1, self.kernelDic['Poly'][k_iter])
            dd = torch.matmul(torch.diag(k_tmp).view(self.num_samples, 1),
                              torch.ones(1, self.num_samples, dtype=torch.double))
            k_tmp = k_tmp / dd
            self.weight[w_iter] = torch.matmul(torch.matmul(alpha.T, k_tmp), alpha)
            w_iter = w_iter + 1

    def opt_alpha(self):

        self.comp_kernel()
        self.regressor = SVR(C=1e1, kernel='precomputed', epsilon=0.01, max_iter=2000)
        self.regressor.fit(self.x_train_kernel, self.y_train)
        print('the support number is:', len(self.regressor.support_))
        alpha_tmp = np.zeros((self.num_samples,))
        alpha_tmp[self.regressor.support_] = self.regressor.dual_coef_
        alpha = torch.from_numpy(alpha_tmp)
        print('before:', torch.matmul(alpha.T, torch.matmul(self.x_train_kernel, alpha)))
        self.comp_weight(alpha)
        self.weight = self.weight / torch.sum(self.weight)
        print(self.weight)
        self.comp_kernel()
        self.regressor.fit(self.x_train_kernel, self.y_train)
        alpha_tmp = np.zeros((self.num_samples,))
        alpha_tmp[self.regressor.support_] = self.regressor.dual_coef_
        alpha = torch.from_numpy(alpha_tmp)
        print('after:', torch.matmul(alpha.T, torch.matmul(self.x_train_kernel, alpha)))

    def forward(self, x_val):
        assert x_val.shape[0] == self.x_train.shape[0], 'but found {} and {}'.format(x_val.shape[0],
                                                                                     self.x_train.shape[0])
        device = x_val.device
        N_val = x_val.shape[1]
        N = self.num_samples
        M = self.x_train.shape[0]

        test_train_inner = torch.matmul(x_val.T, self.x_train)
        test_train_norm = torch.norm(x_val[:, None].T-self.x_train.T, dim=2, p=2)
        x_val_kernel = torch.zeros(N_val, N, device=self.device, dtype=torch.double)
        w_iter = 0
        # RBF
        for k_iter in range(len(self.kernelDic['RBF'])):
            x_val_kernel = x_val_kernel + self.weight[w_iter] * torch.exp(
                -1.0 * self.kernelDic['RBF'][k_iter] * torch.pow(test_train_norm,2))
            w_iter = w_iter + 1
        # Laplace
        for k_iter in range(len(self.kernelDic['Lap'])):
            x_val_kernel = x_val_kernel + self.weight[w_iter] * torch.exp(
                -1.0 * self.kernelDic['Lap'][k_iter] * test_train_norm)
            w_iter = w_iter + 1
        # Polynomial
        for k_iter in range(len(self.kernelDic['Poly'])):
            k_tmp = torch.pow(test_train_inner + 1, self.kernelDic['Poly'][k_iter])
            dd = torch.matmul(torch.diag(k_tmp).view(N_val, 1),
                              torch.ones(1, self.num_samples, dtype=torch.double))
            k_tmp = k_tmp / dd
            x_val_kernel = x_val_kernel + self.weight[w_iter] * k_tmp
            w_iter = w_iter + 1

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
    # plt.figure(2)
    # plt.plot(data_y, 'r+', label='test data', markersize=13)
    # plt.plot(pred_y, 'b2', label='pred data', markersize=13)
    # plt.show()
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
        dataset_con = Comp_activ() #SML() # Airfoil() #Tecator()  #Yacht() #Parkinson() #CosineDataset()  #

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