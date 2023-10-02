import json
import os
from copy import deepcopy

from main import main, parse_args
from utils import get_stats


# {
#     "arch": ["hierarchical", "global"],
#     "hidden": [16, 32, 64, 128],
#     "pool_ratio": [0.25, 0.5],
#     "lr": [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4],
#     "weight_decay": [1e-2, 1e-3, 1e-4, 1e-5],
#     "dataset": ["dataset1"]
# }

def load_config(path="./grid_search_config.json"):
    with open(path, "r") as f:
        return json.load(f)


def run_experiments(args):
    res = []
    for i in range(args.num_trials):
        print("Trial {}/{}".format(i + 1, args.num_trials))
        acc, _, values = main(args, i , False)
        #acc = min(acc, min(min(values[2]), min(values[1])))
        res.append(acc)

    mean, err_bd = get_stats(res, conf_interval=True)
    return mean, err_bd


def grid_search(config: dict):
    args = parse_args()
    args.dataset = config['dataset']
    results = {}
    cnt = save_cnt = 0
    best_acc, err_bd = 0.0, 0.0
    best_args = vars(args)
    for arch in config["arch"]:
        args.architecture = arch
        for hidden in config["hidden"]:
            args.hid_dim = hidden
            for pool_ratio in config["pool_ratio"]:
                args.pool_ratio = pool_ratio
                for conv_layers in config["conv_layers"]:
                    args.conv_layers = conv_layers
                    for lr in config["lr"]:
                        args.lr = lr
                        for weight_decay in config["weight_decay"]:
                                args.weight_decay = weight_decay
                            #for k in config["k"]:
                                #args.k = k
                                acc, bd = run_experiments(args)
                                cnt += 1
                                if acc > best_acc:
                                    best_acc = acc
                                    err_bd = bd
                                    best_args = deepcopy(vars(args))
                                    save_cnt = cnt
    args.output_path = "./output"
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    args.output_path = "./output/model_{}_{}.log".format(save_cnt,args.dataset)
    result = {
        "params": best_args,
        "result": "{:.4f}({:.4f})".format(best_acc, err_bd),
    }
    with open(args.output_path, "w") as f:
        json.dump(result, f, sort_keys=True, indent=4)


grid_search(load_config())
