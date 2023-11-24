import torch
from dynLABRBF import dynTrainLABRBF
import numpy as np
import random
from data_reg import Yacht, Tecator, \
     Comp_activ, Parkinson, SML, Airfoil, \
     Tomshardware, Electricity, KCprice


if __name__ == '__main__':
    test_err_list = []

    for repeat in range(1):
        # parameter for LAB RBF
        max_sv = 350
        ini_sv = max_sv // 3
        WEIGHT = 30
        # parameter for SGD
        LR = 1e-2
        BS = 128

        global seed
        seed = repeat
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # build the dataset
        dataset = Parkinson(required_sv=ini_sv) # KCprice() #Tecator()
        # Yacht() #KCprice() #Tomshardware() #Electricity() #Tecator()
        # #Comp_activ() #SML() # Airfoil()

        test_err = dynTrainLABRBF(dataset, max_sv=max_sv, WEIGHT=WEIGHT, LR=LR, BS=BS)
        test_err_list.append(test_err)

    print('test err mean: ', np.mean(test_err_list), 'err1 std: ',np.std(test_err_list))

    # plt.show()
    print('finish')