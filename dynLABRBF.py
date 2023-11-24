import torch
import numpy as np
import torch.optim as optim
from clsLABRBF import LABRBF
import matplotlib.pyplot as plt


def testFun(LABRBF_model, data_x, data_y):
    #
    batch = 128
    LABRBF_model.eval()
    pred_y_list = []
    cnt = 0
    while cnt < data_x.shape[1]:
        if cnt + batch < data_x.shape[1]:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt: cnt + batch].to(LABRBF_model.device)).detach().cpu())
        else:
            pred_y_list.append(LABRBF_model(x_train=data_x[:, cnt:].to(LABRBF_model.device)).detach().cpu())
        cnt = cnt + batch

    pred_y = torch.cat(pred_y_list, 1)

    pred_y = pred_y / 2 * delta_y + mid_y
    data_y = data_y / 2 * delta_y + mid_y

    pred_y = torch.max(torch.min(pred_y, (delta_y+2*mid_y)/2), (2*mid_y-delta_y)/2)
    acc = 1 - LABRBF_model.rsse_loss(pred=pred_y, target=data_y).detach().cpu()

    print('\t\t the MAE loss: ', format(LABRBF_model.mae_loss(pred=pred_y, target=data_y)))
    print('\t\t the R2 loss: ', format(acc))
    print('\t\t the rmse loss:', format(LABRBF_model.rmse_loss(pred=pred_y, target=data_y)))

    # plt.plot(pred_y.cpu())
    # plt.plot(data_y.cpu())
    # plt.show()

    return acc, pred_y


def TrainKernel(LABRBF_model, train_x, train_y, optFlag=0, LR=1e-2, BS=64):


    # build the optimizer
    if optFlag < 1:
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=LR)
        max_iters = 500
    else:
        # optimizer = torch.optim.SGD(LABRBF_model.parameters(), lr=1e-4)
        optimizer = optim.Adam(LABRBF_model.parameters(), lr=1e-2)
        max_iters = 5000
    # scheduler = lr_scheduler.StepLR(optimizer, 500, 0.5)
    optimizer.zero_grad()

    # train the Kernel
    loss_list = []

    batch_size = BS
    train_num = len(train_y)

    # calculate total training index
    train_indexes = []
    while len(train_indexes) < max_iters * batch_size:
        train_index = np.arange(0, train_num)
        train_index = np.random.permutation(train_index)
        train_indexes.extend(train_index.tolist())

    train_indexes = train_indexes[: max_iters*batch_size]

    for iter_id in range(max_iters):
        LABRBF_model.train()
        # ind_tmp = np.random.randint(0, val_y.shape[0], size=min(batch_size, train_num ))
        ind_tmp = train_indexes[iter_id*batch_size:(iter_id+1)*batch_size]
        # ind_tmp = train_indexes[0:batch_size]
        val_pred = LABRBF_model(x_train=train_x[:, ind_tmp].to(LABRBF_model.device))
        val_loss = LABRBF_model.mse_loss(pred=val_pred, target=train_y[ind_tmp].to(LABRBF_model.device)) # or rsse_loss
        optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(LABRBF_model.parameters(), max_norm=10, norm_type=2)
        optimizer.step()

        tmp = (torch.sqrt(val_loss)).detach().cpu()
        loss_list.append(tmp)

        if iter_id == 0 or iter_id % 100 == 99: #iter_id >=0 :  #
            print('[{}] loss={}'.format(iter_id, val_loss))
            for name, params in LABRBF_model.named_parameters():
                print('-->name:', name, ' -->grad_value:',  params.grad.data.norm(), '-->weight_value:', params.data.norm())


    plt.plot(loss_list)


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
    global delta_y, mid_y
    delta_y = max_y - min_y
    print(delta_y)
    mid_y = (max_y + min_y) / 2
    data_sv_y = (data_sv_y - mid_y) / delta_y * 2
    data_train_y = (data_train_y - mid_y) / delta_y * 2
    data_test_y = (data_test_y - mid_y) / delta_y * 2

    return data_sv_y, data_train_y, data_test_y


def dynTrainLABRBF(dataset, max_sv=100, WEIGHT=1, LR=1e-2, BS=64):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_err_list = []

    data_sv_x, data_sv_y = dataset.get_sv_data()
    data_train_x, data_train_y = dataset.get_val_data()
    data_test_x, data_test_y = dataset.get_test_data()

    data_sv_y, data_train_y, data_test_y = preprocessY(data_sv_y,data_train_y, data_test_y)
    data_sv_x, data_train_x, data_test_x = preprocessX(data_sv_x, data_train_x, data_test_x)

    data_sv_x = data_sv_x.float().to(device)
    data_sv_y = data_sv_y.float().to(device)
    data_train_x = data_train_x.float()
    data_train_y = data_train_y.float()
    data_test_x = data_test_x.float()
    data_test_y = data_test_y.float()

    weight_last = torch.sqrt(torch.tensor(WEIGHT).float() / data_sv_x.shape[0]) * torch.ones(data_sv_x.shape)

    inner_iter = 0
    while inner_iter < 2:
        sv_num = data_sv_x.shape[1]
        print('sv: ', sv_num)
        add_num = sv_num - weight_last.shape[1]
        weight_mean = torch.mean(weight_last).cpu()
        weight_ini = torch.cat((weight_last.cpu(), weight_mean*torch.ones(data_sv_x.shape[0], add_num)), 1)
        LABRBF_model = LABRBF(x_sv=data_sv_x, y_sv=data_sv_y, weight_ini=weight_ini)
        LABRBF_model.float().to(device)
        LABRBF_model.eval()

        TrainKernel(LABRBF_model, data_train_x, data_train_y,
                                  optFlag=inner_iter, LR=LR, BS=BS)
        weight_last = LABRBF_model.weight.data.detach().cpu()

        print('the train error rate loss: ')
        train_err, pred_y = testFun(LABRBF_model, data_train_x, data_train_y)

        err_tmp = torch.abs(data_train_y.reshape(-1) - pred_y.reshape(-1))
        tt = max_sv - sv_num
        k = min(max(10, int(sv_num/10)), tt)
        if k<=0:
            inner_iter = inner_iter+1
            continue
        _, idx = torch.topk(err_tmp, k, dim=0)

        data_sv_x = torch.cat((data_sv_x, data_train_x[:, idx.reshape(-1)].to(device)), 1)
        data_sv_y = torch.cat((data_sv_y, data_train_y[idx.reshape(-1),:].to(device)), 0)

        train_err_list.append(train_err.cpu())

    print('the final test error rate loss: ')
    test_err, pred_y = testFun(LABRBF_model, data_test_x, data_test_y)

    return test_err