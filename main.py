import random
from comet_ml import Experiment
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from train import training
from utils.data import load_dataset
from utils.delta import setup_delta_tracking
from utils.helpers import load_model


def run_experiment(epochs, model_name, training_type, configs):
    """ Runs the basic experiment"""

    print(epochs, "CONFIGS: ", configs)

    # set seed for reproducibility.
    seed = configs.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # gpu training specific seed settings.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load data
    loaders = load_dataset(configs)

    # load model
    model = load_model(model_name, training_type, configs)

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = optim.SGD(model.parameters(), configs.lr,
                          momentum=configs.momentum,
                          weight_decay=configs.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # get tracking dictionaries
    model_weights, layer_dict = setup_delta_tracking(model)

    # train model
    rmae_delta_dict, train_acc_arr, test_acc_arr = training(
        epochs, loaders, model, optimizer, scheduler, criterion, model_weights, layer_dict, configs)

    return rmae_delta_dict, train_acc_arr, test_acc_arr
