import torch
from torch.utils.data import Dataset
from sklearn import datasets
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"/usr/share/fonts/dejavu-serif-fonts/DejaVuMathTexGyre.ttf", size=14)
FONTSIZE = 18


class CosineDataset(Dataset):

    def __init__(self, num_samples=100, num_SV=50, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)

        # self.num_samples = num_samples
        # self.data_x = torch.rand(1, num_samples) * 2 * 3.1416
        # self.data_y = (1 + torch.sin(4 * self.data_x))*(4+torch.sin(2*self.data_x)) /(3.5+torch.sin(self.data_x))
        # # self.data_y = self.data_y - self.data_y.mean()
        # # self.data_y = (1 + torch.sin(2*self.data_x[0] + 3*self.data_x[1])) /\
        # #               (3.5 + torch.sin(self.data_x[0] - self.data_x[1]))
        # self.data_y = self.data_y.reshape(-1)
        # tmp_x = torch.linspace(0, 0.8 * 3.14156, 1000)
        # # tmp_x, _ = torch.sort(self.data_x)
        # data_gt_y = (1 + torch.sin(4 * tmp_x))*(4+torch.sin(2*tmp_x)) /(3.5+torch.sin(tmp_x))
        # # self.data_y = self.data_y + 0.5*(torch.rand(self.data_y.shape)-0.5)

        self.dataFile = '/volume1/scratch/fhe/dataset/yacht'
        data_tmp = scio.loadmat(self.dataFile + '/yachthyX.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['yachthyX']), 0, 1).float()
        print(self.data_x.shape)
        for fea_id in range(self.data_x.shape[0]):
            self.data_x[fea_id, :] = self.data_x[fea_id, :] / torch.max(torch.abs(self.data_x[fea_id, :]))

        data_tmp = scio.loadmat(self.dataFile + '/yachthyY.mat')
        self.data_y = torch.from_numpy(data_tmp['yachthyY']).float()
        self.data_y = self.data_y[:, 0]
        # self.data_y = self.data_y.reshape(-1)

        self.num_samples = self.data_x.shape[1]
        self.train_num = num_samples#int(np.ceil(0.2 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]

        x_train = self.data_x[0, self.train_id].reshape(-1)
        _, index = torch.sort(x_train)
        self.train_id = self.train_id[index]
        self.sv_id = []
        train_id_tmp = self.train_id
        while len(self.sv_id) < num_SV:
            tmp = num_SV - len(self.sv_id)
            indexes = (np.arange(0, tmp) * self.train_num / tmp).astype(int)
            self.sv_id = self.sv_id + list(train_id_tmp[indexes])
            train_id_tmp = list(set(train_id_tmp).difference(set(self.sv_id)))
        self.sv_id = list(set(self.sv_id))
        # self.sv_id = self.train_id[::ratio]
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
        self.data_y = torch.sin(1*torch.pow(self.data_x, 3)).reshape(-1) + self.noise
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num

        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        x_train = self.data_x[:, self.train_id]
        x_train = x_train.reshape(-1)
        sorted, index = torch.sort(x_train)
        self.train_id = self.train_id[index]
        idx = list(range(0, int(np.ceil(num_samples / 2)), 10)) + list(range(int(np.ceil(num_samples / 2)), self.train_num, 5))
        self.sv_id = self.train_id[idx]
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


class NoiseDataset(Dataset):

    def __init__(self, num_samples):
        self.num_samples = num_samples

        self.data_x = 3*torch.rand(1,num_samples)-1.5
        self.data_y = (torch.sin(3*torch.pow(self.data_x, 1))+ torch.pow(self.data_x,2)).reshape(-1)
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num

        ind_tmp = np.random.permutation(self.num_samples)
        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        x_train = self.data_x[:, self.train_id]
        x_train = x_train.reshape(-1)
        sorted, index = torch.sort(x_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::3]
        self.noise_id = self.sv_id[3::10]
        self.data_y[self.noise_id] = (self.data_y[self.noise_id]
                                      + 1.5*np.sign(self.data_y[self.noise_id]))
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)
        self.nonNoise = list(set(self.sv_id).difference(set(self.noise_id)))

        print('sv_num: ', self.sv_num, 'val_num: ', self.val_num, 'test_num: ',
              self.test_num)
        xx = torch.linspace(-1.5,1.5,300)
        yy =  (torch.sin(3*xx)+ torch.pow(xx,2))
        x_train = self.data_x[:,self.train_id]
        y_train = self.data_y[self.train_id]
        x_sv = self.data_x[:,self.sv_id]
        y_sv = self.data_y[self.sv_id]

        plt.figure(5)
        ax=plt.subplot(2, 2, 1)
        plt.plot(xx.reshape(-1), yy.reshape(-1), 'k:', label='Ground Truth')
        # plt.plot(self.data_x[:, self.sv_id].reshape(-1), self.data_y[self.sv_id].reshape(-1), 'ko',
        #          markerfacecolor='white', label='train data', markersize=13)
        plt.plot(self.data_x[:, self.noise_id].reshape(-1), self.data_y[self.noise_id].reshape(-1), 'k*',
                 label='Noised Data', markersize=10)
        plt.plot(self.data_x[:, self.nonNoise].reshape(-1), self.data_y[self.nonNoise].reshape(-1), 'k+',
                 markersize=10)
        plt.plot(self.data_x[:, self.val_id].reshape(-1), self.data_y[self.val_id].reshape(-1), 'k+',
                 label='Sample Data', markersize=10)
        # plt.plot(self.data_x[:,self.test_id].reshape(-1),  self.data_y[self.test_id].reshape(-1),  'g|', label='test data', markersize=13)
        plt.legend(loc="lower right", fontsize=FONTSIZE)
        ax.set_title('(a) Ground Truth', fontsize=FONTSIZE)

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
        self.dataFile = '/volume1/scratch/fhe/dataset/yacht'
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

        self.dataFile = '/volume1/scratch/fhe/dataset/tecator'
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
        self.dataFile = '/volume1/scratch/fhe/dataset/comp-active'
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
        self.sv_id = self.train_id[::100]
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
        self.dataFile = '/volume1/scratch/fhe/dataset/parkinson'
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


class SML(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset/SML'
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


class Airfoil(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset/airfoil_self_noise'
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


class YearPredict(Dataset):

    def __init__(self):
        self.dataFile = 'D:/keyan/kernels/deep kernels/exp/realdata/song'
        data_tmp = scio.loadmat(self.dataFile + '/YearPrediction.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).float()
        print(self.data_x.shape)

        y_tmp = data_tmp['Y']/1.0
        self.data_y = torch.from_numpy(y_tmp)

        self.num_samples = self.data_x.shape[1]
        self.train_num = 463715
        self.test_num = self.num_samples - self.train_num

        self.train_id = (np.arange(self.train_num))
        self.test_id = (np.arange(self.train_num, self.train_num+10000))
        y_train = self.data_y[self.train_id]
        y_train = y_train.reshape(-1)
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::1000]
        self.val_id = self.train_id[::100]
        # self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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


class Twitter(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset'
        data_tmp = scio.loadmat(self.dataFile + '/Twitter.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/Twitter.mat')
        self.data_y = torch.from_numpy(data_tmp['Y']).double()
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
        self.sv_id = self.train_id[::1000]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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


class Boston(Dataset):

    def __init__(self, seed=0):
        self.seed = seed
        torch.manual_seed(self.seed )
        np.random.seed(self.seed )
        random.seed(self.seed )
        X, Y = datasets.fetch_california_housing(return_X_y=True)
        self.data_x = torch.transpose(torch.from_numpy(X),0,1).double()
        self.data_y = torch.from_numpy(Y).double()
        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::30]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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


class Tomshardware(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset'
        data_tmp = scio.loadmat(self.dataFile + '/TomsHardware.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/TomsHardware.mat')
        self.data_y = torch.from_numpy(data_tmp['Y']).double()
        self.data_y = self.data_y[:, 0]

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

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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

class Protein(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset'
        data_tmp = scio.loadmat(self.dataFile + '/Protein.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/Protein.mat')
        self.data_y = torch.from_numpy(data_tmp['Y']).double()
        self.data_y = self.data_y[:, 0]

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::50]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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


class Electricity(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset'
        data_tmp = scio.loadmat(self.dataFile + '/Electricity.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/Electricity.mat')
        self.data_y = torch.from_numpy(data_tmp['Y']).double()
        self.data_y = self.data_y[:, 0]

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::50]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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

class KCprice(Dataset):

    def __init__(self):
        self.dataFile = '/volume1/scratch/fhe/dataset'
        data_tmp = scio.loadmat(self.dataFile + '/KCprice.mat')
        self.data_x = torch.transpose(torch.from_numpy(data_tmp['X']), 0, 1).double()
        print(self.data_x.shape)

        data_tmp = scio.loadmat(self.dataFile + '/KCprice.mat')
        self.data_y = torch.from_numpy(data_tmp['Y']).double()
        self.data_y = self.data_y[:, 0]

        self.num_samples = self.data_x.shape[1]
        self.train_num = int(np.ceil(0.8 * self.num_samples))
        self.test_num = self.num_samples - self.train_num
        ind_tmp = np.random.permutation(self.num_samples)

        self.train_id = ind_tmp[0: self.train_num]
        self.test_id = ind_tmp[self.train_num:]
        y_train = self.data_y[self.train_id]
        sorted, index = torch.sort(y_train)
        self.train_id = self.train_id[index]
        self.sv_id = self.train_id[::50]
        self.val_id = list(set(self.train_id).difference(set(self.sv_id)))
        self.sv_num = len(self.sv_id)
        self.val_num = len(self.val_id)

        self.data_y = self.data_y.view(max(self.data_y.shape), 1)
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