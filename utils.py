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
import h5py
from sklearn.decomposition import KernelPCA, PCA 
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from itertools import combinations
import seaborn as sns
import json

output_combinations = {
    1: ("identity feat", "gat"),
    2: ("identity feat", "hierarchical"),
    3: ("identity feat", "gin"),
    4: ("identity feat", "global"),
    5: ("degree feat", "gat"),
    6: ("degree feat", "hierarchical"),
    7: ("degree feat", "gin"),
    8: ("degree feat", "global"),
    9: ("noise feat", "gat"),
    10: ("noise feat", "hierarchical"),
    11: ("noise feat", "gin"),
    12: ("noise feat", "global"),
    13: ("ones feat", "gat"),
    14: ("ones feat", "hierarchical"),
    15: ("ones feat", "gin"),
    16: ("ones feat", "global")
}

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
    #matrix[matrix == -1] = 0
    # Get the dimensions of the matrix
    rows, cols = matrix.size()
    # Create a mask for the upper half (above the diagonal)
    mask = torch.triu(torch.ones(rows, cols, dtype=torch.uint8), diagonal=1)
    # Calculate the sum of elements in the upper half
    #mask[matrix == 0] = 0
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
    for i in range(4, n // 2):
        if n % i == 0 and n/i > 3:
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
    df = pd.read_csv('{}1/stanford_output_testing.csv'.format(input_path), index_col=[0, 1, 2])
    for i in range(1, length):
        df_rem = pd.read_csv('{}{}/stanford_output_testing.csv'.format(input_path, i+1), index_col=[0, 1, 2])
        df = pd.concat([df, df_rem])

    df = df * 100
    pd.set_option('display.float_format', '{:.2f}'.format)
    return df

def update_args_with_dict(args, arg_dict):
    for key, value in arg_dict.items():
        if hasattr(args, key):
            setattr(args, key, value)

# return numpy array
def read_hidden_feat(folder_path):
    # Specify the path to your HDF5 file
    file_path = "{}/save_hidden_output_test_trial0.h5".format(folder_path)
    
    # Open the HDF5 file for reading
    with h5py.File(file_path, 'r') as file:
        # List all datasets in the file
        dataset_names = list(file.keys())
        # Iterate through all datasets and read them
        for dataset_name in dataset_names:
            data = file[dataset_name][:]
    
    return torch.tensor(data)

def read_hidden_feat2(folder_path):
    # Specify the path to your HDF5 file
    file_path = "{}/save_hidden_output_test_trial0.h5".format(folder_path)
    
    # Open the HDF5 file for reading
    with h5py.File(file_path, 'r') as file:
        # List all datasets in the file
        dataset_names = list(file.keys())
        # Iterate through all datasets and read them
        data = []
        for dataset_name in dataset_names:
            data.append(file[dataset_name][:])
    
    return torch.tensor(data)

def apply_pca(data, n_components):
    data = data.numpy()
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    
    # Inverse transform to get the reconstructed data
    reconstructed_data = pca.inverse_transform(reduced_data)
    reconstruction_error = np.mean(np.square(data - reconstructed_data))
    print(f"Reconstruction error PCA: {reconstruction_error}")

    return reduced_data 

def apply_kernel_pca(data, n_components):
    pca = KernelPCA(kernel='rbf', n_components=n_components)
    data = pca.fit_transform(data)

    return data 

def apply_t_sne(data, n_components):
    tsne = TSNE(n_components=n_components)
    data = tsne.fit_transform(data)

    return data

def min_max_norm(tensor):
    # Calculate the minimum and maximum values along each feature
    min_values, _ = tensor.min(dim=0)
    max_values, _ = tensor.max(dim=0)

    # Perform Min-Max scaling
    normalized_tensor = (tensor - min_values) / (max_values - min_values)

    return normalized_tensor

def min_max_normalize(column):
    min_val = column.min()
    max_val = column.max()
    return (column - min_val) / (max_val - min_val)

# doing dim reduction with changing the data from numpy array into tensor
def dim_reduction(data, method_num, n_components=2):
    names_methods = ['apply_pca', 'apply_kernel_pca', 'apply_t_sne']
    data = globals()[names_methods[method_num]](data, n_components)
    return torch.tensor(data)

def comparing_hidden_feat(data_path, output_path, number_samples_for_type_graph, method_num):
    names_methods = ['PCA', 'Kernel_PCA', 'T-SNE']
    data = dim_reduction(read_hidden_feat(output_path), method_num, 2)
    data = min_max_norm(data)

    indices = torch.load('{}/test_indices.pt'.format(output_path))
    classes = (indices/number_samples_for_type_graph).floor().to(torch.int64)

    output_path = output_path + '/analysis_plots'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = output_path + '/{}'.format(names_methods[method_num])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    scatter_plot_classes(data, classes, output_path, '{} Scatter Plot for classes'.format(names_methods[method_num]))    

    df1 = pd.read_csv('./{}/info_about_graphs.csv'.format(data_path), header=[0, 1])
    networks_names = df1.columns.get_level_values(0).unique()
    df = pd.DataFrame()
    for name in networks_names:
        if df.empty:
            df = df1[name]
        else:
            df = pd.concat([df, df1[name]], ignore_index=True)

    df = df.apply(min_max_normalize)
    n = df.shape[1]  # Change this to your desired 'n'

    combinations_list = list(combinations(range(n), 2))
    for i in range(n):
        combinations_list.insert(i, (i, i))

    for i in combinations_list:
        array = min_max_norm(torch.tensor(df.iloc[indices.tolist(), list(i)].values))

        scatter_plot_classes_given_feat(data, array, classes, output_path, title="{} Scatter Plot ({}, {}) where circle is hidden_feat and triangle is the properties"
        .format(names_methods[method_num], df.iloc[:, i[0]].name, df.iloc[:, i[1]].name), name_feat1=df.iloc[:, i[0]].name, name_feat2=df.iloc[:, i[1]].name)

def comparing_hidden_feat2(data_path, output_path, number_samples_for_type_graph, method_num, idx):
    names_methods = ['PCA', 'Kernel_PCA', 'T-SNE']
    data = dim_reduction(read_hidden_feat(output_path), method_num, 2)
    data = min_max_norm(data)

    indices = torch.load('{}/test_indices.pt'.format(output_path))
    classes = (indices/number_samples_for_type_graph).floor().to(torch.int64)

    output_path = output_path + '/analysis_plots'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    output_path = output_path + '/{}'.format(names_methods[method_num])
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df1 = pd.read_csv('./{}/info_about_graphs.csv'.format(data_path), header=[0, 1])
    networks_names = df1.columns.get_level_values(0).unique()
    df = pd.DataFrame()
    for name in networks_names:
        if df.empty:
            df = df1[name]
        else:
            df = pd.concat([df, df1[name]], ignore_index=True)

    n = df.shape[1] 
    networks_names = df1.columns.get_level_values(0).unique()
    scatter_plot_classes(data, classes, output_path,networks_names, '{} {} Scatter Plot for classes'.format(output_combinations[idx], names_methods[method_num]))    
    

    for i in range(n):
        array = df.iloc[indices.tolist(), i].values
        
        scatter_plot_classes_given_feat2(data, array, classes, output_path, "{} {} Scatter Plot of the hidden features downsampled into 2 dimensions, with the heatmap coloring representing the {}"
        .format(output_combinations[idx], names_methods[method_num], df.iloc[:, i].name), df.iloc[:, i].name, networks_names)

def scatter_plot_classes_given_feat2(X1, X2, y, output_path, title, column_name, networks_names):
    plt.figure(figsize=(14, 10))
    # Define a colormap for X2
    colormap = plt.get_cmap('viridis')
    max_X2 = X2.max()
    # Normalize the float values in X2 to fit within the [0, 1] range
    X2_normalized = (X2 - X2.min()) / (X2.max() - X2.min())
    
    # Define marker styles for y
    marker_styles = ['o', 's', 'D', '^', 'v', '>', '<', 'P']
    
    # Create a scatter plot
    for class_id in range(8):
        class_mask = y == class_id  # Mask for points in the current class
        class_x = X1[class_mask]
        class_y = y[class_mask]
        class_color = [colormap(X2_normalized[i]) for i in range(class_x.shape[0])]  # Map X2 values to colors using the colormap
        class_marker = marker_styles[class_id]  # Assign marker style based on class

        plt.scatter(class_x[:, 0], class_x[:, 1], c=class_color, label=f'Class {networks_names[class_id]}', marker=class_marker)

    # Add colorbar for the heatmap
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(0, max_X2))
    sm.set_array([])  # An empty array is required for the colorbar to work
    cbar = plt.colorbar(sm, ax=plt.gca())
    cbar.set_label('{} Value'.format(column_name))

    # Add labels and legend
    plt.xlabel('Hidden_Feat1')
    plt.ylabel('Hidden_Feat2')
    plt.legend()

    plt.title(title)

    plt.savefig('{}/{}.png'.format(output_path, title))
    plt.show()


def scatter_plot_classes(X, y, output_path, networks_names, title="Scatter Plot", name_feat1='Hidden_Feat1', name_feat2='Hidden_Feat2'):
    plt.figure(figsize=(14, 10))
    
    # Determine the unique class labels
    unique_classes = torch.unique(y)

    # Create a color map with a sufficient number of distinct colors
    num_colors = max(unique_classes) + 1  # Maximum class label + 1
    colors = plt.cm.viridis(torch.linspace(0, 1, num_colors))

    # Create a scatter plot
    for class_label in range(num_colors):
        mask = (y == class_label)
        plt.scatter(X[mask, 0], X[mask, 1], color=colors[class_label], label=f'Class {networks_names[class_label]}')

    # Add labels and legend
    plt.xlabel(name_feat1)
    plt.ylabel(name_feat2)
    plt.legend()
    plt.title(title)

    # Show the plot
    plt.savefig('{}/{}.png'.format(output_path, title))
    plt.show()



def scatter_plot_classes_given_feat(X1, X2, y, output_path, title="Scatter Plot", name_feat1='Feature 1', name_feat2='Feature 2'):
    plt.figure(figsize=(14, 10))
    
    # Determine the unique class labels
    unique_classes = torch.unique(y)

    # Create a color map with a sufficient number of distinct colors
    num_colors = max(unique_classes) + 1  # Maximum class label + 1
    colors = plt.cm.viridis(torch.linspace(0, 1, num_colors))

    # Create a scatter plot
    for class_label in unique_classes:
        mask = (y == class_label)
        plt.scatter(X1[mask, 0], X1[mask, 1], color=colors[class_label], marker='o', label=f'Class {class_label}')
        plt.scatter(X2[mask, 0], X2[mask, 1], color=colors[class_label], marker='^', label=f'Class {class_label}')
    # Add labels and legend
    plt.xlabel(name_feat1)
    plt.ylabel(name_feat2)
    plt.legend()
    plt.title(title)

    # Show the plot
    plt.savefig('{}/{}.png'.format(output_path, title))
    plt.show()

def network_metrics():
    df = pd.read_csv('/data/info_about_graphs.csv', header=[0, 1])
    df_min = df.min().round(2)
    df_mean = df.mean().round(2)
    df_mean = df.max().round(2)
    combined_df = pd.concat([df_min, df_mean, df_mean], axis=1, keys=['Min', 'Mean', 'Max'], )

    # Transpose the DataFrame to have the desired structure
    combined_df = combined_df.transpose()
    print(combined_df)
    # Reset the index to include the dataset names
    combined_df = combined_df.rename_axis(['Dataset', 'Metric'])

    # Sort the index for a cleaner presentation
    combined_df = combined_df.sort_index()
    combined_df.rename(columns=list(df_mean.index.get_level_values(1).unique()))
    df = pd.DataFrame(combined_df.values,index=combined_df.index, columns= list(df_min.index.get_level_values(1).unique()))
    return df


def merge_stanfrod_prediction_and_properties():
    df1 = pd.read_csv('stanford_output.csv')
    for col in df1.columns:
        if df1[col].dtype == 'float64':
            df1[col] = df1[col].round(2)

    df = pd.read_csv('metrics_of_stanford_networks.csv')

    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].round(2)


    # Merge the DataFrames based on the specified columns
    result = pd.merge(df1, df, left_on='network_name', right_on='Name', how='inner')
    result.to_csv('stanfrod_prediction_with_properites.csv', index=False)

    return result


def plot_frequency_distributions(data):
    num_columns = data.shape[1]

    # Calculate the number of rows and columns for the grid
    num_rows = (num_columns + 3) // 4  # Assuming 4 plots per row

    # Set up subplots in a grid layout
    fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(16, 4 * num_rows))

    for i in range(num_columns):
        row, col = divmod(i, 4)  # Calculate the row and column for each plot
        ax = axes[row, col]
        column_data = data[:, i]

        ax.hist(column_data, bins=20, color='skyblue', edgecolor='black')
        ax.set_title(f'Column {i + 1} Frequency Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    # Remove any empty subplots
    for i in range(num_columns, num_rows * 4):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()
    plt.show()

def correlation_heatmap(data, output_path, name_model, name_feat):
    corr_matrix = np.abs(np.corrcoef(data, rowvar=False))

    # Create a heatmap with color bars
    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, linewidths=0.5, cbar=True)

    # Add color bars
    cax = plt.gcf().axes[-1]
    cax.set_ylabel('Correlation', size=14)

    plt.title('Correlation Matrix Heatmap {} {}'.format(name_model, name_feat), size=16)
    plt.show()
    plt.savefig('{}/Correlation Matrix Heatmap {} {}.png'.format(output_path, name_model, name_feat))



@torch.no_grad()
def test2(model: torch.nn.Module, loader, device, args, trial, e, if_test):
    model.eval()
    correct = 0.0
    loss = 0.0
    num_graphs = 0
    list_hidden_output = []
    args['current_batch'] = 1
    args['current_trial'] = trial
    args['current_epoch'] = e
    for batch in loader:
        batch_graphs, batch_labels = batch
        num_graphs += batch_labels.size(0)
        batch_graphs = batch_graphs.to(device)
        batch_labels = batch_labels.long().to(device)

        # Calculate hidden features
        if args['save_hidden_output_test'] == True and if_test and (args['save_last_epoch_hidden_output'] == False or e == args['epochs']-1):
            out, hidden_feat = model(batch_graphs, args)
            hidden_feat = hidden_feat.cpu().detach().numpy()
            list_hidden_output.append(hidden_feat)
            del hidden_feat
        # Calculate predictions and loss
        out, _ = model(batch_graphs)
        pred = out.argmax(dim=1)
        loss += torch.nn.F.nll_loss(out, batch_labels, reduction="sum").item()
        correct += pred.eq(batch_labels).sum().item()

        # Delete variables that are no longer needed
        del out
        del pred

        args['current_batch'] += 1

    # Clear the list after it's no longer needed
    if args['save_hidden_output_test'] == True and if_test and (args['save_last_epoch_hidden_output'] == False or e == args['epochs']-1):
        with h5py.File("{}/save_hidden_output_test_trial{}.h5".format(args['output_path'], trial), 'a') as hf:
            hf.create_dataset('epoch_{}'.format(e), data=np.concatenate(list_hidden_output))
        list_hidden_output.clear()

    return correct / num_graphs, loss / num_graphs



def independent_test2():
    combination_dicts = []
    number_folders = len(combination_dicts)+1


    results = []
    selected_keys = ["architecture", "feat_type", "hidden_dim", "num_layers", "test_loss", "test_loss_error", "test_acc", "test_acc_error"]
    for i in range(1, number_folders):
        #read the model
        output_path = "output{}/".format(i)
        files_names = os.listdir(output_path)
        models_path = [file for file in files_names if  "last_model_weights_trail" in file]
        args_file_name = [file for file in files_names if "Data_dataset_Hidden_" in file][0]
        args_path = output_path+args_file_name


        with open(args_path, 'r') as f:
            args = json.load(f)
        args = args['hyper-parameters']
        def get_network():
            pass
        accuracies = []
        dataset1 = dataset2 = dataset3 = dataset4 = None
        test_loader1 = test_loader2 = test_loader3 = test_loader4 = None
        losses = []
        device = None
        for num_trial, model_path in enumerate(models_path):
            model_op = get_network(args['architecture'])
            if args['feat_type'] == 'ones_feat':
                dataset = dataset1
                test_loader = test_loader1
            elif args['feat_type'] == 'degree_feat':
                dataset = dataset2
                test_loader = test_loader2
            elif args['feat_type'] == 'noise_feat':
                dataset = dataset3
                test_loader = test_loader3
            else:
                dataset = dataset4
                test_loader = test_loader4

            num_feature, num_classes, _ = dataset.statistics()

            model = model_op(
                    in_dim=num_feature,
                    hidden_dim=args['hidden_dim'],
                    out_dim=num_classes,
                    num_layers=args['num_layers'],
                    pool_ratio=args['pool_ratio'],
                    dropout=args['dropout'],
                    output_activation = args['output_activation']
            ).to(device)
            model.load_state_dict(torch.load(output_path+model_path))
            model.eval()
            accuracy, loss = test2(model, test_loader, device, args, num_trial, 1, False)
            accuracies.append(accuracy)
            losses.append(loss)

        accuracies = np.array(accuracies)
        losses = np.array(losses)

        result = [args["architecture"], args["feat_type"], args["hidden_dim"], args["num_layers"], losses.mean(), losses.var(), accuracies.mean(), accuracies.var()]
        result = dict(zip(selected_keys, result))
        results.append(result)

    test_results = pd.DataFrame(results)
    test_results.to_csv('test_resutls_large_networks.csv')



def draw_figures_scatters(df):
    # Create a list of unique feat_type values to assign different colors
    feat_types = df['feat_type'].unique()
    palette = sns.color_palette('husl', n_colors=len(feat_types))
    length = len(df['hidden_dim'].unique())
    # Create a separate plot for each architecture
    x_axis = np.arange(1, 7)
    for arch in df['architecture'].unique():
        plt.figure(figsize=(10, 6))
        plt.title(f'Architecture {arch} Test Accuracy vs Hidden Dimension')
        x = 0
        cnt = 1
        for idx, feat_type in enumerate(feat_types):
            subset = df[(df['architecture'] == arch) & (df['feat_type'] == feat_type)]            
            x += 0.2
            plt.scatter(x_axis+x, subset['test_acc'], label=f'{feat_type} - test_acc', marker='x', color=palette[idx])
            plt.scatter(x_axis+x, subset['second_test_acc'], label=f'{feat_type} - second_test_acc', marker='s', facecolor='none', edgecolor=palette[idx])
            
        for idx in range(length):
            plt.axvline(idx+1, color='red')
        
        plt.axvline(length+1, color='red')

        min_hidden_dim = df['hidden_dim'].min()
        max_hidden_dim = df['hidden_dim'].max()
        x_ticks = [2 ** i for i in range(int(np.log2(min_hidden_dim)), int(np.log2(max_hidden_dim) + 1))]
        plt.xticks(x_axis+0.5, labels=[str(val) for val in x_ticks])
        plt.xlabel('Hidden Dimension')
        
        plt.ylabel('Accuracy')
        plt.legend()
        
        # Save or display the plot
        plt.savefig(f'architecture_{arch}_plot.png')  # Save the plot with a unique name
        plt.show()  # Display the plot (remove this line if you want to save only)


def draw_figures_scatters_2(df):
    # Assuming your DataFrame is named 'df'
    sns.set_style("whitegrid")

    # Define the size of the plot
    plt.figure(figsize=(14, 10))

    # Get unique architecture values
    unique_architectures = df['architecture'].unique()
    unique_feat_types = df['feat_type'].unique()

    # Get a color palette with different colors for each feat_type
    palette = sns.color_palette('husl', n_colors=len(unique_feat_types))
    line_styles = [False, (3, 3), (1, 1), (7, 1)]

    # Create a separate plot for each architecture
    for arch in unique_architectures:
        filtered_data = df[df['architecture'] == arch]
        plt.figure(figsize=(14, 10))  # Set size for the current plot

        for i, feat in enumerate(unique_feat_types):
            label = f'{arch} {feat}'

            # Use a different color for each feat_type
            color = palette[i]

            sns.lineplot(x='hidden_dim', y='test_acc', data=filtered_data[filtered_data['feat_type'] == feat], marker='o', label=f'{label} (test_acc)', color=color, markersize=10, dashes=line_styles[i])
            sns.lineplot(x='hidden_dim', y='second_test_acc', data=filtered_data[filtered_data['feat_type'] == feat], marker='^', label=f'{label} (second_test_acc)', color=color, markersize=10, dashes=line_styles[i])

        plt.xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2
        plt.xlabel('Hidden Dimension (log2 scale)')

        # Set the x-tick labels to powers of 2
        min_hidden_dim = df['hidden_dim'].min()
        max_hidden_dim = df['hidden_dim'].max()
        x_ticks = [2 ** i for i in range(int(np.log2(min_hidden_dim)), int(np.log2(max_hidden_dim) + 1))]
        plt.xticks(x_ticks, labels=[str(val) for val in x_ticks])

        plt.ylabel('Accuracy')
        plt.title(f'Architecture: {arch}')
        plt.legend(title='Legend')
        filename = f'{arch}_plot.png'
        plt.savefig(filename)
        plt.show()


