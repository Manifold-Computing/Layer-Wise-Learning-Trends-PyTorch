import numpy as np
import json
import pickle
from types import SimpleNamespace
from main import run_experiment
from comet_ml import Experiment


def multi_train(configs_path="./configs.json"):

    configs = SimpleNamespace(**json.load(open(configs_path)))
    print(configs)

    average_rmae_dict, average_train_acc, average_test_acc = {
    }, np.zeros(configs.epochs), np.zeros(configs.epochs)

    og_config_exp_name = configs.exp_name

    for _, seed in enumerate(configs.seed_list):

        configs.exp_name = f"{seed}_{og_config_exp_name}"
        print(f"Training: {configs.exp_name}")

        configs.seed = seed
        rmae_dict, train_acc_arr, test_acc_arr = run_experiment(
            configs.epochs, configs.model_name, "untrained", configs)

        with open(f"{configs.save_directory}{configs.exp_name}_rmae.pkl", 'wb') as f:
            pickle.dump(rmae_dict, f, pickle.HIGHEST_PROTOCOL)

    #     for layer in rmae_dict:
    #         if layer not in average_rmae_dict:
    #             average_rmae_dict[layer] = np.array(rmae_dict[layer])
    #         else:
    #             average_rmae_dict[layer] += np.array(rmae_dict[layer])

    #     average_train_acc += train_acc_arr
    #     average_test_acc += test_acc_arr

    # for layer in average_rmae_dict:
    #     average_rmae_dict[layer] /= 5

    # average_train_acc /= len(configs.seed_list)
    # average_test_acc /= len(configs.seed_list)

    print("Done!")

    # PLOT THE GRAPHS USING THE AVERAGE ARRAYS


if __name__ == "__main__":
    multi_train()
