
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from data_reg import CosineDataset, Yacht, Concrete, Tecator, \
    YearPredict, Comp_activ, Parkinson, SML, SkillCraft,Airfoil

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)   # reproducible

class Net(torch.nn.Module):
    def __init__(self, n_feature):
        super(Net, self).__init__()
        # # 1
        # self.layer1 = torch.nn.Linear(n_feature, 1000)   # hidden
        # self.layer2 = torch.nn.Linear(1000, 500)  # hidden layer
        # self.layer3 = torch.nn.Linear(500, 50)  # hidden layer
        # # # self.layer4 = torch.nn.Linear(100, 50)  # hidden layer
        # self.predict = torch.nn.Linear(50, 1)   # output layer
        # # 2
        # self.layer1 = torch.nn.Linear(n_feature, 500)   # hidden
        # self.layer2 = torch.nn.Linear(500, 50)  # hidden layer
        # self.predict = torch.nn.Linear(50, 1)   # output layer
        # # # 3
        self.layer1 = torch.nn.Linear(n_feature, 500)   # hidden
        self.predict = torch.nn.Linear(500, 1)   # output layer

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')

    def forward(self, x):
        x = F.relu(self.layer1(x))    # activation function for hidden layer
        # x = F.relu(self.layer2(x))  # activation function for hidden layer
        # x = F.relu(self.layer3(x))  # activation function for hidden layer
        # x = F.relu(self.layer4(x))  # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x

def rsse_loss(pred, target):
    # RSSE
    tmp = ((target.reshape(-1) - target.mean()) ** 2).sum()
    loss = ((pred.reshape(-1) - target.reshape(-1)) ** 2).sum() / tmp
    return loss

def testFun(cnn, data_x, data_y):
    #
    batch = 128
    cnn.eval()
    pred_y_list = []
    loss_func = torch.nn.MSELoss()
    cnt = 0
    data_y = data_y.view(len(data_y),1)
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            tmp = torch.transpose(data_x[:, cnt: cnt + batch], 0, 1)
            pred_y_list.append(cnn(tmp).detach())
        else:
            tmp = torch.transpose(data_x[:, cnt:], 0, 1)
            pred_y_list.append(cnn(tmp).detach())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 0)
    pred_y = torch.squeeze(pred_y).view(len(pred_y), 1)
    print('\t\t the rsse loss:', format(rsse_loss(pred_y, data_y)))
    print('\t\t the rmse loss:', format(torch.sqrt(loss_func(pred_y, data_y))*delta_y))
    # plt.plot(pred_y.cpu())
    # plt.plot(data_y.cpu())
    # plt.show()

    return torch.sqrt(loss_func(pred_y, data_y))*delta_y

def train_rnn(rnn, train_x, train_y, data_test_x, data_test_y):
    # Hyper Parameters
    device = train_x.device
    lr = 1e-2  # learning rate
    max_iters = 3000
    batch_size = 128
    train_num = len(train_y)
    input_size = train_x.shape[0]
    loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)  # optimize all cnn parameters


    # calculate total training index
    train_indexes = []
    while len(train_indexes) < max_iters * batch_size:
        train_index = np.arange(0, train_num)
        train_index = np.random.permutation(train_index)
        train_indexes.extend(train_index.tolist())

    train_indexes = train_indexes[: max_iters * batch_size]
    loss_list = []

    for iter_id in range(max_iters):
        ind_tmp = train_indexes[iter_id*batch_size:(iter_id+1)*batch_size]
        x = torch.transpose(train_x[:, ind_tmp], 0, 1)
        y = train_y[ind_tmp].view(batch_size, 1)

        prediction = rnn(x).view(batch_size, 1)  # rnn output
        loss = torch.sqrt(loss_func(prediction, y))  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if iter_id == 0 or iter_id % 100 == 99: #iter_id >=0 :  #
            print('[{}] loss={}'.format(iter_id, loss*delta_y))
            # for name, params in rnn.named_parameters():
            #     print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())

        tmp = (loss*delta_y).detach().cpu()
        loss_list.append(tmp)
        # test

    # test
    print('the train error rate loss: ')
    acc = testFun(rnn, train_x, train_y)
    print('the test error rate loss: ')
    acc = testFun(rnn, data_test_x, data_test_y)

    #
    #again
    optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-3)

    for iter_id in range(3000):
        ind_tmp = train_indexes[iter_id*batch_size:(iter_id+1)*batch_size]
        x = torch.transpose(train_x[:, ind_tmp], 0, 1)
        y = train_y[ind_tmp].view(batch_size, 1)

        prediction = rnn(x)  # rnn output
        loss = torch.sqrt(loss_func(prediction, y))  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if iter_id == 0 or iter_id % 100 == 99: #iter_id >=0 :  #
            print('[{}] loss={}'.format(iter_id, loss*delta_y))
            # for name, params in rnn.named_parameters():
            #     print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())
        tmp = (loss*delta_y).detach().cpu()
        loss_list.append(tmp)

    # test
    print('the train error rate loss: ')
    acc = testFun(rnn, train_x, train_y)
    print('the test error rate loss: ')
    acc = testFun(rnn, data_test_x, data_test_y)

    plt.plot(loss_list)
    plt.show()

    return  acc

if __name__ == '__main__':
    acc = []
    for iter in range(1):
        # build the dataset
        dataset_con = Tecator()#Comp_activ() #Airfoil()#Parkinson() #SML() #Yacht()#Concrete() #SkillCraft() #CosineDataset()  #

        d_train_x, d_train_y = dataset_con.get_sv_data()
        data_val_x, data_val_y = dataset_con.get_val_data()
        data_test_x, data_test_y = dataset_con.get_test_data()

        train_x = torch.cat([d_train_x, data_val_x], dim=1)
        train_y = torch.cat([d_train_y, data_val_y], dim=0)
        M = train_x.shape[0]

        train_x = train_x.cuda().double()
        train_y = train_y.cuda().double()
        data_test_x = data_test_x.cuda().double()
        data_test_y = data_test_y.cuda().double()

        global delta_y
        delta_y = 1
        # normalization
        delta_y = train_y.max() - train_y.min()
        print(delta_y)
        mid_y = (train_y.max() + train_y.min()) / 2
        train_y = (train_y - mid_y) / delta_y
        data_test_y = (data_test_y - mid_y) / delta_y

        for fea_id in range(M):
            delta_x = train_x[fea_id, :].max() - train_x[fea_id, :].min()
            # print(delta_x)
            if delta_x == 0:
                delta_x = 1
            mid_x = (train_x[fea_id, :].max() + train_x[fea_id, :].min()) / 2
            train_x[fea_id, :] = (train_x[fea_id, :] - mid_x) / delta_x
            data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mid_x) / delta_x
        train_x = train_x * 2
        data_test_x = data_test_x * 2

        # for fea_id in range(M):
        #     mean_x = torch.mean(train_x[fea_id, :])
        #     std_x = torch.std(train_x[fea_id,:])
        #     if std_x == 0:
        #         std_x = 1
        #     train_x[fea_id, :] = (train_x[fea_id, :] - mean_x)/std_x
        #     data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mean_x)/std_x

        cnnNet = Net(n_feature=M)
        cnnNet.cuda().double()
        print(cnnNet)

        acc.append(train_rnn(cnnNet, train_x, train_y, data_test_x,data_test_y).cpu())

    print(np.mean(acc))
    print(np.std(acc))