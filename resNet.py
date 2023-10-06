import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from data_reg import CosineDataset, Yacht, Tecator, \
     Comp_activ, Parkinson, SML, Airfoil, \
     Twitter, Tomshardware, Electricity, KCprice
import numpy as np

class ResNetBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(ResNetBlock, self).__init__()

        self.dense1 = nn.Linear(in_feature, out_feature)
        self.bn1 = nn.BatchNorm1d(out_feature)

        self.dense2 = nn.Linear(out_feature, out_feature)
        self.bn2 = nn.BatchNorm1d(out_feature)

        self.downsample = None
        if in_feature != out_feature:
            self.downsample = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        identity = x

        out = self.dense1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out)

        out = self.dense2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.functional.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, in_feature):
        super(ResNet, self).__init__()

        self.Layer1 = ResNetBlock(in_feature, 1000)
        self.Layer2 = ResNetBlock(1000, 100)
        self.Layer3 = ResNetBlock(100, 100)
        self.Layer4 = ResNetBlock(100, 100)
        self.predict = torch.nn.Linear(100, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')

        print(sum(p.numel() for p in self.parameters() if p.requires_grad) )

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.Layer3(x)
        x = self.Layer4(x)
        x = self.predict(x)  # linear output
        return x


class ResNetSmall(nn.Module):
    def __init__(self, in_feature):
        super(ResNetSmall, self).__init__()

        self.Layer1 = ResNetBlock(in_feature, 500)
        self.Layer2 = ResNetBlock(500, 50)
        self.predict = torch.nn.Linear(50, 1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')

        print(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        x = self.Layer1(x)
        x = self.Layer2(x)
        x = self.predict(x)  # linear output
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
    data_y = data_y.view(len(data_y), 1)
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

    pred_y = pred_y / 2 * delta_y + mid_y
    data_y = data_y / 2 * delta_y + mid_y

    print('\t\t the rsse loss:', format(rsse_loss(pred_y, data_y)))
    print('\t\t the rmse loss:', format(torch.sqrt(loss_func(pred_y, data_y))))
    # plt.plot(pred_y.cpu())
    # plt.plot(data_y.cpu())
    # plt.show()

    return rsse_loss(pred_y, data_y)

def train_rnn(rnn, train_x, train_y, val_x, val_y, data_test_x, data_test_y):
    # Hyper Parameters
    device = train_x.device
    lr = 1e-3  # learning rate
    max_iters = 3000
    batch_size = 128
    train_num = len(train_y)
    input_size = train_x.shape[0]
    loss_func = torch.nn.MSELoss()
    # optimizer = torch.optim.SGD(rnn.parameters(), lr=0.01)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)  # optimize all cnn parameters

    # Define early stopping parameters
    best_val_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement
    counter = 0

    # calculate total training index
    train_indexes = []
    while len(train_indexes) < max_iters * batch_size:
        train_index = np.arange(0, train_num)
        train_index = np.random.permutation(train_index)
        train_indexes.extend(train_index.tolist())

    train_indexes = train_indexes[: max_iters * batch_size]
    loss_list = []

    for iter_id in range(max_iters):
        ind_tmp = train_indexes[iter_id * batch_size:(iter_id + 1) * batch_size]
        x = torch.transpose(train_x[:, ind_tmp], 0, 1)
        y = train_y[ind_tmp].view(batch_size, 1)

        prediction = rnn(x).view(batch_size, 1)  # rnn output
        loss = torch.sqrt(loss_func(prediction, y))  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if iter_id == 0 or iter_id % 100 == 99:  # iter_id >=0 :  #
            # Validation
            rnn.eval()
            with torch.no_grad():
                val_outputs = rnn(val_x.T)
                val_loss = torch.sqrt(loss_func(val_outputs, val_y))

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                rnn_best = rnn
            # elif val_loss > 1.05*best_val_loss:
            #     counter += 1
            #     if counter >= patience:
            #         print(f'Early stopping after {iter_id + 1} epochs.')
            #         rnn = rnn_best
            #         break

        #     print('[{}] loss={}'.format(iter_id, loss*delta_y))
        # for name, params in rnn.named_parameters():
        #     print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())

        tmp = (loss).detach().cpu()
        loss_list.append(tmp)
        # test

    rnn = rnn_best
    # test
    print('the train error rate loss: ')
    acc = testFun(rnn, train_x, train_y)
    print('the test error rate loss: ')
    acc = testFun(rnn, data_test_x, data_test_y)

    #
    # again
    optimizer = torch.optim.SGD(rnn.parameters(), lr=1e-4)

    for iter_id in range(1000):
        ind_tmp = train_indexes[iter_id * batch_size:(iter_id + 1) * batch_size]
        x = torch.transpose(train_x[:, ind_tmp], 0, 1)
        y = train_y[ind_tmp].view(batch_size, 1)

        prediction = rnn(x)  # rnn output
        loss = torch.sqrt(loss_func(prediction, y))  # calculate loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        # if iter_id == 0 or iter_id % 100 == 99: #iter_id >=0 :  #
        #     print('[{}] loss={}'.format(iter_id, loss*delta_y))
        # for name, params in rnn.named_parameters():
        #     print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())
        tmp = (loss).detach().cpu()
        loss_list.append(tmp)

    # test
    print('the train error rate loss: ')
    acc1 = testFun(rnn, train_x, train_y)
    print('the test error rate loss: ')
    acc2 = testFun(rnn, data_test_x, data_test_y)

    # plt.plot(loss_list)
    # plt.show()

    return acc1, acc2

def preprocessX(data_sv_x, data_train_x, data_test_x):
    for fea_id in range(data_sv_x.shape[0]):
        max_x = max([data_sv_x[fea_id, :].max(), data_train_x[fea_id, :].max()])
        min_x = min([data_sv_x[fea_id, :].min(), data_train_x[fea_id, :].min()])
        delta_x = max_x - min_x
        # print(delta_x)
        if delta_x == 0:
            delta_x = 1
        mid_x = (max_x + min_x) / 2
        data_sv_x[fea_id, :] = (data_sv_x[fea_id, :] - mid_x) / delta_x
        data_train_x[fea_id, :] = (data_train_x[fea_id, :] - mid_x) / delta_x
        data_test_x[fea_id, :] = (data_test_x[fea_id, :] - mid_x) / delta_x
    data_sv_x = data_sv_x * 2
    data_train_x = data_train_x * 2
    data_test_x = data_test_x * 2

    return data_sv_x, data_train_x, data_test_x

def preprocessY(data_sv_y, data_train_y, data_test_y):

    # normalization
    max_y = max([data_sv_y.max(), data_train_y.max()])
    min_y = min([data_sv_y.min(), data_train_y.min()])
    global delta_y, mid_y
    delta_y = max_y - min_y
    print(delta_y)
    mid_y = (max_y + min_y) / 2
    data_sv_y = (data_sv_y - mid_y) / delta_y * 2
    data_train_y = (data_train_y - mid_y) / delta_y * 2
    data_test_y = (data_test_y - mid_y) / delta_y * 2

    return data_sv_y, data_train_y, data_test_y

if __name__ == '__main__':
    acc_test = []
    acc_train = []
    for iter in range(1):
        seed = iter
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build the dataset
        dataset_con = Yacht()#Tecator()# Airfoil()
        #Comp_activ()
        #SML() #Electricity() # KCprice()  #Parkinson() # #Yacht()# Protein() # Tomshardware() # # SkillCraft() #Concrete() #CosineDataset()  #

        data_sv_x, data_sv_y = dataset_con.get_sv_data()
        data_train_x, data_train_y = dataset_con.get_val_data()
        data_test_x, data_test_y = dataset_con.get_test_data()

        data_sv_y, data_train_y, data_test_y = preprocessY(data_sv_y, data_train_y, data_test_y)
        data_sv_x, data_train_x, data_test_x = preprocessX(data_sv_x, data_train_x, data_test_x)
        #
        # train_x = torch.cat([data_sv_x, data_train_x], dim=1)
        # train_y = torch.cat([data_sv_y, data_train_y], dim=0)
        M = data_train_x.shape[0]

        data_sv_x = data_sv_x.double()
        data_sv_y = data_sv_y.double()
        data_train_x = data_train_x.double()
        data_train_y = data_train_y.double()
        data_test_x = data_test_x.double()
        data_test_y = data_test_y.double()

        rnnNet = ResNetSmall(in_feature=M)
        # rnnNet = ResNet(in_feature=M)

        rnnNet.double()
        # print(cnnNet)
        acc_tr, acc_te = train_rnn(rnnNet, data_train_x, data_train_y, data_sv_x, data_sv_y, data_test_x,
                                   data_test_y)
        acc_train.append(acc_tr.cpu())
        acc_test.append(acc_te.cpu())

    print(np.mean(acc_train))
    print(np.std(acc_train))
    print(np.mean(acc_test))
    print(np.std(acc_test))