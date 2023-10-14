import logging
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import t
import dgl
import random
import os
import re
import numpy as np
import math
import pandas as pd

def get_stats(
    array, conf_interval=False, name=None, stdout=False, logout=False
):
    """Compute mean and standard deviation from an numerical array

    Args:
        array (array like obj): The numerical array, this array can be
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal.
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n - 1)
        err_bound = t_value * se
    else:
        err_bound = std

    # log and print
    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound

def boxplot(accs, output_path, name, feat_type):
    # Plot the distribution of average degree for each type of graph
    fig, ax = plt.subplots(figsize=(14, 10), dpi=80)
    ax.set_title('Test Accurcy of diffrente trail models')
    ax.boxplot(accs)
    ax.set_xticklabels([feat_type])
    plt.show()
    plt.savefig('{}/boxplot_{}_{}.png'.format(output_path, name, feat_type))

def mean_of_n_values(arr, smoth):
    arr = arr.reshape(arr.shape[0], -1, smoth)
    mean = arr.mean(axis=2)
    return mean.reshape(mean.shape[0], -1)

def acc_loss_plot(results, epochs, smoth, num_trials, output_path, feat_type, name):
    results = np.array(results)

    # generate sample data
    plt.figure(figsize=(14, 10), dpi=80)

    x = np.arange(0,epochs,smoth)

    loss_train_results = mean_of_n_values(results[:, 0, :], smoth)
    acc_train_results = mean_of_n_values(results[:, 1, :], smoth)
    acc_valid_results = mean_of_n_values(results[:, 2, :], smoth)
        
    plt.plot(x, np.mean(acc_train_results,axis=0), label='Train Accurcy')
    plt.fill_between(x, np.min(acc_train_results,axis=0), np.max(acc_train_results,axis=0), alpha=0.2, label='Train Accurcy Noise')
    plt.plot(x, np.mean(acc_valid_results,axis=0), label='Valid Accurcy')
    plt.fill_between(x, np.min(acc_valid_results,axis=0), np.max(acc_valid_results,axis=0), alpha=0.2, label='Valid Accurcy Noise')
    plt.title("train plot and valid plot over over {:4d} models ploted the mean with min noise and max noise smothed every {:4d} epochs".format(num_trials, smoth))
    plt.legend()
    plt.savefig('{}/accurcy_plot_valid_train_{}_{}.png'.format(output_path, name, feat_type))

        
    plt.figure(figsize=(14, 10), dpi=80)

    x = np.arange(0,epochs,smoth)

    plt.plot(x, np.mean(loss_train_results,axis=0), label='Train Loss')
    plt.fill_between(x, np.min(loss_train_results,axis=0), np.max(loss_train_results,axis=0), alpha=0.2, label='Train Loss Noise')

    plt.title(" loss plot over over {:4d} models ploted the mean with min noise and max noise smothed every {:4d} epochs".format(num_trials, smoth))
    plt.legend()
    plt.show()
    plt.savefig('{}/loss_plot_train_{}_{}.png'.format(output_path, name, feat_type))

def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)


def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full(
            (num_nodes[i],), i, dtype=torch.long, device=num_nodes.device
        )
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(
    x: torch.Tensor,
    ratio: float,
    batch_id: torch.Tensor,
    num_nodes: torch.Tensor,
):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full(
        (batch_size * max_num_nodes,), torch.finfo(x.dtype).min
    )
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device)
        + i * max_num_nodes
        for i in range(batch_size)
    ]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k

def find(str2):
  # Set the directory path
  dir_path = "output/"

  files = os.listdir(dir_path)
  file_names = [os.path.basename(file) for file in files]

  matching_names = list(filter(lambda x: re.search(str2, x, re.IGNORECASE), file_names))

  # Pass the folder name to the command line
  return matching_names[0]

def calculate_avg_shortest_path(graph):
    matrix = dgl.shortest_dist(graph)
    # Get the dimensions of the matrix
    rows, cols = matrix.size()
    # Create a mask for the upper half (above the diagonal)
    mask = torch.triu(torch.ones(rows, cols, dtype=torch.uint8), diagonal=1)
    # Calculate the sum of elements in the upper half
    sum_upper_half = torch.sum(matrix * mask)
    # Calculate the number of elements in the upper half
    count = torch.sum(mask)
    # Calculate the average of the upper half
    average_upper_half = sum_upper_half / count

    return float(average_upper_half)


def generate_factors(n, min, max):
    
    # Initialize a list to store the factors of n
    factors = []
    
    # Find factors of n
    for i in range(n//3+1, n // 2):
        if n % i == 0:
            factors.append(i)
    
    # Randomly choose one of the factors as 'a'
    if len(factors) < 2:
      x = np.random.randint(min, max + 1)
      return generate_factors(x, min, max)
    a = random.choice(factors)
    
    # Calculate 'b' as 'n' divided by 'a'
    b = n // a
    
    return a, b


def is_prime(n):
    if n == 2:
        return 1
    if n%2 == 0 or n == 1:
        return 0
    i = 3
    while i*i<= n:
        if n % i == 0:
            return 0
        i += 2
        
    return 1

def generate_uniform_array_non_prime(n, min, max):
    ans = []
    while n != 0:
        x = 2
        while is_prime(x):
            x = np.random.randint(min, max + 1)
            if is_prime(x):
              print(x)
        ans.append(x)
        n-= 1
    return np.array(ans)

def merge_dataframes(length, input_path):
    df = pd.read_csv('/content/{}1/stanford_output_testing.csv'.format(input_path), index_col=[0, 1, 2])
    for i in range(1, length):
        df_rem = pd.read_csv('/content/{}{}/stanford_output_testing.csv'.format(input_path, i+1), index_col=[0, 1, 2])
        df = pd.concat([df, df_rem])

    df = df * 100
    pd.set_option('display.float_format', '{:.2f}'.format)
    return df
