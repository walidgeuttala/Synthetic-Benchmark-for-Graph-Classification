# Imports required libraries
import os
import torch
# Sets torch as the environment library
os.environ['TORCH'] = torch.__version__
# Imports the math library for mathematical functions
import math
# Fixes random seed
import random
# Imports libraries required for training and using the DGL library
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(42)
# Imports DGL libraries for generating and training the model on Torch
import dgl
# Imports libraries for calculations in the model
import numpy as np
import pandas as pd
# Imports networkx for generating graphs
import networkx as nx
# Imports libraries for handling plots
import matplotlib.pyplot as plt
import pylab
# Imports the statistics library for calculating mean
from statistics import mean
# Using binary search method
import bisect
import gzip
from data import GraphDataset
from create_dataset import *
from utils import *
import os
import json
import re
import shutil
import numpy as np
import torch
import numpy as np
from sklearn.decomposition import PCA
import subprocess


torch.cuda.empty_cache()
# name of the folder of the data training
data_path = 'data'
# starting the naming of the outputs folders
cnt = 1

config = {
    "architecture": ['gin', 'gat', 'global', 'hierarchical'],
    "hidden_dim": [1, 2 , 4, 8, 16, 32],
    "feat_type": ['identity_feat', 'degree_feat', 'noise_feat', 'ones_feat', 'norm_degree_feat'],
    "lr": [1e-2],
    "num_layers":[4],
    "weight_decay": [1e-3],
    "k": [4],
    "num_trials": [1]
}

keys = list(config.keys())
values = [config[key] for key in keys]
combinations = list(itertools.product(*values))

combination_dicts = []

# Iterate through the combinations and create dictionaries
for combo in combinations:
    combination_dict = {keys[i]: combo[i] for i in range(len(keys))}
    combination_dicts.append(combination_dict)

for combo in combination_dicts:
        output_path = "output{}/".format(cnt)
        if os.path.exists(output_path):
            files_names = os.listdir(output_path)
            args_file_name = [file for file in files_names if "Data_dataset_Hidden_" in file][0]
            args_path = output_path+args_file_name
            with open(args_path, 'r') as f:
                args = json.load(f)
            args = args['hyper-parameters']
            if args['architecture'] == combo['architecture'] and args['hidden_dim'] == combo['hidden_dim'] and args['feat_type'] == combo['feat_type']:
                continue
            else:
                shutil.rmtree(output_path)
        print('{}-------------------------------architecture : {} feat_type: {} hidden_dim: {} num_layers: {}---------------------'
              .format(cnt, combo['architecture'], combo['feat_type'], combo['hidden_dim'], combo['num_layers']))
        formatted_string = " ".join([f"--{key} {value}" for key, value in combo.items()])
        print()
        script = "python main.py " + formatted_string
        script = script.split()
        subprocess.run(script)
        torch.cuda.empty_cache()
        os.rename("output", "output"+str(cnt))
        cnt += 1