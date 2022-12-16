import torch
from torch.utils.data import Dataset
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt


class CosineDataset(Dataset):

    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

        self.data_x = torch.rand(1, num_samples) * 2 * 3.1416
        self.data_y = (1 + torch.sin(4 * self.data_x))*(4+torch.sin(2*self.data_x)) /(3.5+torch.sin(self.data_x))
        # self.data_y = self.data_y - self.data_y.mean()
        # self.data_y = (1 + torch.sin(2*self.data_x[0] + 3*self.data_x[1])) /\
        #               (3.5 + torch.sin(self.data_x[0] - self.data_x[1]))

        self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        y_train = y_train.reshape(-1)
        # sorted, index = torch.sort(y_train)
        # self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::3]

        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class SinDataset(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.data_x = 3*torch.rand(1,num_samples)
        self.noise = 0.5*(torch.rand(num_samples)-0.5)
        self.data_y = torch.sin(torch.pow(self.data_x, 3)).reshape(-1)  + self.noise
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        x_train = self.data_x[:, self.train_id]
        x_train = x_train.reshape(-1)
        sorted, index = torch.sort(x_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::5]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)
        xx = torch.linspace(0,3,300)
        yy =  torch.sin(torch.pow(xx, 3))
        x_train = self.data_x[:,self.train_id]
        y_train = self.data_y[self.train_id]
        x_sv = self.data_x[:,self.sv_id]
        y_sv = self.data_y[self.sv_id]
        plt.figure(1)
        plt.subplot(2,2,1)
        plt.scatter(x_train, y_train, c='black', marker='+')
        plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')
        plt.plot(xx,yy,'k:')
        plt.subplot(2, 2, 2)
        plt.scatter(x_train, y_train, c='black', marker='+')
        plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')
        plt.subplot(2, 2, 3)
        plt.scatter(x_train, y_train, c='black', marker='+')
        plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')
        plt.subplot(2, 2, 4)
        plt.scatter(x_train, y_train, c='black', marker='+')
        plt.scatter(x_sv,  y_sv, marker='o',facecolor='white',edgecolors='black')

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class Yacht(Dataset):

    def __init__(self):
        self.dataFile = 'D:/learning kernel for PCA/v2/dataset/yacht'
        data_tmp = scio.loadmat(self.dataFile + '/yachthyX.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['yachthyX']), 0, 1).float()
        print(self.data_x.shape)
        for fea_id in range(self.data_x.shape[0]):
            self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))

        data_tmp = scio.loadmat(self.dataFile + '/yachthyY.mat')
        self.data_y = torch.from_numpy(data_tmp['yachthyY']).float()

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        y_train = y_train.reshape(-1)
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::10]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class Tecator(Dataset):

    def __init__(self):

        self.dataFile = 'D:/learning kernel for PCA/v2/dataset/tecator'
        data_tmp = scio.loadmat(self.dataFile + '/tecatorX.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['tecatorX']), 0, 1).float()
        print(self.data_x.shape)
        # for fea_id in range(self.data_x.shape[0]):
        #     self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))

        data_tmp = scio.loadmat(self.dataFile + '/tecatorY.mat')
        self.data_y = torch.from_numpy(data_tmp['tecatorY']).float()
        self.data_y = self.data_y[:, 0]
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::10]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max( self.data_y.shape), 1)
        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class Comp_activ(Dataset):

    def __init__(self):
        self.dataFile = 'D:/learning kernel for PCA/v2/dataset/comp-active'
        data_tmp = scio.loadmat(self.dataFile + '/comp.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['cpuactivX']), 0, 1).float()
        print(self.data_x.shape)
        # for fea_id in range(self.data_x.shape[0]):
        #     self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))
            # tmp  = self.data_x[fea_id, :]
            # self.data_x[fea_id, :] = (tmp - torch.mean(tmp)) / torch.std(tmp)

        data_tmp = scio.loadmat(self.dataFile + '/comp.mat')
        self.data_y = torch.from_numpy(data_tmp['cpuactivY']).float()
        self.data_y = torch.add(self.data_y, 1.0)
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num

        data_tmp = scio.loadmat(self.dataFile + '/order.mat')
        ind_tmp = (data_tmp['Randomorder']).astype(np.int32)
        ind_tmp = ind_tmp.reshape(-1)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        y_train = y_train.reshape(-1)
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::200]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class Parkinson(Dataset):

    def __init__(self):
        self.dataFile = 'D:/keyan/kernels/deep kernels/exp/realdata/parkinson'
        data_tmp = scio.loadmat(self.dataFile + '/parkinson.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['parkinsonX']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/parkinson.mat')
        self.data_y = torch.from_numpy(data_tmp['parkinsonY2']).double()
        self.data_y = self.data_y[:, 0]
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::45]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class SML(Dataset):

    def __init__(self):
        self.dataFile = 'D:/keyan/kernels/deep kernels/exp/realdata/SML'
        data_tmp = scio.loadmat(self.dataFile + '/SML.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['SMLX']), 0, 1).float()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/SML.mat')
        self.data_y = torch.from_numpy(data_tmp['SMLY2']).float()
        self.data_y = self.data_y[:, 0]
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::10]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]


class Airfoil(Dataset):

    def __init__(self):
        self.dataFile = 'D:/HDfor2LevelRegression/airfoil_self_noise'
        data_tmp = scio.loadmat(self.dataFile + '/airfoil_X.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['airfoil_X']), 0, 1).float()
        print(self.data_x.shape)
        # for fea_id in range(self.data_x.shape[0]):
        #     self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))

        data_tmp = scio.loadmat(self.dataFile + '/airfoil_Y.mat')
        self.data_y = torch.from_numpy(data_tmp['airfoil_Y']).float()
        self.data_y = self.data_y[:, 0]
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::10]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max( self.data_y.shape), 1)

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.num_samples

    def get_sv_data(self):
        return self.data_x[:, self.sv_id], self.data_y[self.sv_id]

    def get_val_data(self):
        return self.data_x[:, self.val_id], self.data_y[self.val_id]

    def get_test_data(self):
        return self.data_x[:, self.test_id], self.data_y[self.test_id]
