import os
import json
import pandas as pd
import numpy as np

def count_output_folders():
    current_directory = os.getcwd()
    all_items = os.listdir(current_directory)
    output_folders = [folder for folder in all_items if os.path.isdir(folder) and folder.startswith('output')]
    return len(output_folders)

length = count_output_folders()
print(length)
current_path = ''
keys = ['architecture', 'hidden_dim', 'num_layers', 'feat_type', 'train_loss', 'train_loss_error', 'train_acc', 'train_acc_error', 'valid_acc', 'valid_acc_error', 'test_acc', 'test_acc_error']
df = []
for i in range(1, length+1):
    args_path = [f for f in os.listdir(current_path+'output{}/'.format(i)) if 'Data_dataset' in f][0]
    with open(current_path+'output{}/'.format(i)+args_path) as f:
        data = json.load(f)
    results = np.array(data['results'])
    ans = np.ones(results.shape[1]*2)
    mean_results = np.mean(results, axis=0)
    var_results = np.var(results, axis=0)
    ans[::2] = mean_results
    ans[1::2] = var_results
    values = [data['hyper-parameters']['architecture'], data['hyper-parameters']['hidden_dim'], data['hyper-parameters']['num_layers'], data['hyper-parameters']['feat_type']]+ans.tolist()

    my_dict = {keys[i]: values[i] for i in range(len(keys))}
    df.append(my_dict)

df = pd.DataFrame(df)
df.to_csv('test_resutls_small_networks.csv')
print(df)