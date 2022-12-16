import torch
import torch.nn as nn
import numpy as np
import array
import torch.optim as optim
import argparse
import random
from clsLABRBF import LABRBF
from data_reg import CosineDataset, Yacht, Tecator, \
     Comp_activ, Parkinson, SML, Airfoil
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import random

# old = torch.load('old/old.pth')['b']
# new = torch.load('new.pth')['b']
#
# print(old == new.cpu())
# print((old==new.cpu()).all())
# import pdb
# pdb.set_trace()

First_Layer_LR = 1e-3
First_Layer_BS = 128



def testFun(LABRBF_model, data_x, data_y):
    #
    batch = 128
    LABRBF_model.eval()
    pred_y_list = []
    cnt = 0
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt: cnt + batch]).detach())
        else:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt:]).detach())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 1)
    pred_y = pred_y.view(max(pred_y.shape),1)
    print('\t\t the rsse loss: ', format(LABRBF_model.rsse_loss(pred=pred_y, target=data_y)))
    print('\t\t the rmse loss:', format(LABRBF_model.rmse_loss(pred=pred_y, target=data_y)*delta_y/2))

    # plt.plot(pred_y.cpu())
    # plt.plot(data_y.cpu())
    # plt.show()

    return LABRBF_model.rsse_loss(pred=pred_y, target=data_y), pred_y


def TrainKernel(LABRBF_model, train_x, train_y, optFlag=0):

    # initialize
    # saved_weights = torch.load('weight0adam.pth')
    # adp_model.weight.data = saved_weights


    LABRBF_model.train()
    # build the optimizer
    if optFlag < 1:
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=First_Layer_LR)
    else:
        optimizer = torch.optim.SGD(LABRBF_model.parameters(), lr=1e-3)
    # scheduler = lr_scheduler.StepLR(optimizer, 500, 0.5)
    optimizer.zero_grad()

    # train the DeepKernel
    alpha_list = []
    loss_list = []
    tt_list = []
    LABRBF_model.train()
    max_iters = 3000
    batch_size = First_Layer_BS
    train_num = len(train_y)

    # calculate total training index
    train_indexes = []
    while len(train_indexes) < max_iters * batch_size:
        train_index = np.arange(0, train_num)
        train_index = np.random.permutation(train_index)
        train_indexes.extend(train_index.tolist())

    train_indexes = train_indexes[: max_iters*batch_size]

    for iter_id in range(max_iters):

        # ind_tmp = np.random.randint(0, val_y.shape[0], size=min(batch_size, train_num ))
        ind_tmp = train_indexes[iter_id*batch_size:(iter_id+1)*batch_size]
        # ind_tmp = train_indexes[0:batch_size]
        val_pred = LABRBF_model(x_train=train_x[:, ind_tmp])
        val_loss = LABRBF_model.mse_loss(pred=val_pred, target=train_y[ind_tmp]) # or rsse_loss
        optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(LABRBF_model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        tmp = (torch.sqrt(val_loss)*delta_y/2).detach().cpu()
        loss_list.append(tmp)

        # if iter_id == 0 or iter_id % 100 == 99: #iter_id >=0 :  #
        #     print('[{}] loss={}'.format(iter_id, val_loss))
        #     for name, params in adp_model.named_parameters():
        #         print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())
        #
        #     print('min weight:', torch.min(adp_model.weight.data))
        #     adp_model.norm_print()
        #     tmp = adp_model.alpha_norm.detach().cpu().numpy()
        #     alpha_list.append(tmp)
        #     tt_list.append(testFun(adp_model, test_x, test_y).cpu())
        # scheduler.step()

    plt.plot(loss_list)


    weight_mat = LABRBF_model.weight.data

    return weight_mat


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

    return  data_sv_x, data_train_x, data_test_x


def preprocessY(data_sv_y, data_train_y, data_test_y):

    # normalization
    max_y = max([data_sv_y.max(), data_train_y.max()])
    min_y = min([data_sv_y.min(), data_train_y.min()])
    delta_y = max_y - min_y
    print(delta_y)
    mid_y = (max_y + min_y) / 2
    data_sv_y = (data_sv_y - mid_y) / delta_y * 2
    data_train_y = (data_train_y - mid_y) / delta_y * 2
    data_test_y = (data_test_y - mid_y) / delta_y * 2

    return data_sv_y, data_train_y, data_test_y

if __name__ == '__main__':
    test_err_list = []
    train_err_list = []
    train2_err_list = []
    global delta_y
    delta_y = 2

    for repeat in range(1):
        err_thes = 1e-3
        max_sv = 400

        max_err = 1
        WEIGHT = 0.01

        global seed
        seed = repeat
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build the dataset
        dataset_con = Comp_activ()#Airfoil() #Tecator()  #Comp_activ() # YearPredict()
        #Yacht()#Concrete() #SML() #Parkinson() #CosineDataset()  #

        data_sv_x, data_sv_y = dataset_con.get_sv_data()
        data_train_x, data_train_y = dataset_con.get_val_data()
        data_test_x, data_test_y = dataset_con.get_test_data()

        data_sv_y, data_train_y, data_test_y = preprocessY(data_sv_y,data_train_y, data_test_y)
        data_sv_x, data_train_x, data_test_x = preprocessX(data_sv_x, data_train_x, data_test_x)

        data_sv_x = data_sv_x.cuda().double()
        data_sv_y = data_sv_y.cuda().double()
        data_train_x = data_train_x.cuda().double()
        data_train_y = data_train_y.cuda().double()
        data_test_x = data_test_x.cuda().double()
        data_test_y = data_test_y.cuda().double()

        weight_last = torch.sqrt(torch.tensor(WEIGHT).float() / data_sv_x.shape[0]) * torch.ones(data_sv_x.shape)

        inner_iter = 0
        while inner_iter < 2:
            sv_num = data_sv_x.shape[1]
            print('sv: ', sv_num)
            add_num = sv_num - weight_last.shape[1]
            weight_mean = torch.mean(weight_last).cpu()
            weight_ini = torch.cat((weight_last.cpu(), weight_mean*torch.ones(data_sv_x.shape[0], add_num)), 1)
            LABRBF_model = LABRBF(x_sv=data_sv_x, y_sv=data_sv_y, weight_ini=weight_ini)
            LABRBF_model.cuda().double()
            LABRBF_model.eval()

            weight_last = TrainKernel(LABRBF_model, data_train_x, data_train_y, inner_iter)

            print('the train error rate loss: ')
            train_err, pred_y = testFun(LABRBF_model, data_train_x, data_train_y)

            last_max_err = max_err
            err_tmp = torch.abs(data_train_y - pred_y)
            tt = max_sv- sv_num
            k = min(max(50, int(tt/3)), tt)
            _, idx = torch.topk(err_tmp, k, dim=0)
            max_err = torch.max(err_tmp) / torch.std(data_train_y)
            print(max_err)
            data_sv_x = torch.cat((data_sv_x, data_train_x[:, idx.reshape(-1)]), 1)
            data_sv_y = torch.cat((data_sv_y, data_train_y[idx.reshape(-1),:]), 0)

            train_err_list.append(train_err.cpu())
            if sv_num < max_sv:
                inner_iter = 0
            else:
                inner_iter = inner_iter+1
            # if sv_num == max_sv and train_err_list[-2] - train_err <= 1e-5:
            #     break

            print('the test error rate loss: ')
            test_err, pred_y = testFun(LABRBF_model, data_test_x, data_test_y)

        train2_err_list.append(train_err.cpu())
        test_err_list.append(test_err.cpu())


    print('train err mean: ', np.mean(train2_err_list), 'err1 std: ', np.std(train2_err_list))
    print('test err mean: ', np.mean(test_err_list), 'err1 std: ',np.std(test_err_list))

    plt.show()
    print('finish')