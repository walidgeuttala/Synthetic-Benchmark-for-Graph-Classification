import argparse
import gzip
import subprocess
import numpy as np
import dgl
from network import get_network
import networkx as nx
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import os 
import json 
import torch
from identity import compute_identity
from utils import calculate_avg_shortest_path
import logging

def parse_args():
    parser = argparse.ArgumentParser(description="Testing Stanford Networks")
    parser.add_argument(
        "--model_weights_path",
        type=str,
        default="dataset",
        help="path for weights of the model",
    )
    parser.add_argument(
        "--args_file",
        type=str,
        default=False,
        help="args json file for the model that we trained with",
    )
    parser.add_argument(
        "--dist_draw",
        type=bool,
        default=False,
        help="dist draw for networks list",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='data',
        help="dataset path",
    )
    parser.add_argument(
        "--feat_type",
        type=str,
        default='ones_feat',
        help="feature type",
    )

    return parser.parse_args()

def stanford_degree_dist_plots(draw = True):
    # Open the file in read mode
    names = []
    density = []
    transitivity = []
    avg_short_path = []
    with open('links.txt', 'r') as file:
        # Iterate over each line in the file
        f = 0
        for line in file:
            # Print the line
            
            if len(line) == 1:
                f = 1
            else :
                if f == 1:
                    f = 0
                else:
                    graph, name = download_Stanford_network(line[:-1])
                    avg_short_path.append(calculate_avg_shortest_path(graph))
                    graph = dgl.to_networkx(graph)
                    graph = nx.Graph(graph)
                    names.append(name[:-7])
                    density.append(nx.density(graph))
                    transitivity.append(nx.transitivity(graph))
                    
                    if draw == True:
                        print("network name : "+names[-1])
                        print()
                        print('number of nodes : ',graph.number_of_nodes())
                        print('nuler of edges : ', graph.number_of_edges())
                        print('average_shortest_path', avg_short_path[-1])
                        print("transitivity : ",transitivity[-1])
                        print("density : ",density[-1])
                        graph = nx.Graph(graph)
                        
                        hist = nx.degree_histogram(graph)
                        plt.plot(hist)
                        plt.xlabel("Degree")
                        plt.ylabel("Frequency")
                        plt.title("Degree Distribution")
                        plt.show()
    return names, density, transitivity, avg_short_path     



def download_Stanford_network(url, save_as = "/txt.txt"):
    
    input_file = url.split("/")[-1]
    subprocess.run(['wget', url, '-O', input_file])
    
    with gzip.open(input_file, 'rb') as f_in:
        with open(save_as, 'wb') as f_out:
            f_out.write(f_in.read())
    
    os.remove(input_file)
    
    data = np.loadtxt(save_as, dtype=int)

    unique_values = np.unique(data)

    # map the IDs to a new range of values between 0 and k-1
    data = np.searchsorted(unique_values, data)

    data = data.tolist()
    data = [x for x in data]

    graph = dgl.graph(data)

    
    return graph, input_file

@torch.no_grad()
def test_network_diff_nfeat(model, graph, name, device, feat_type, k, param):
    model.eval()
    ans = [] 
    softmax = torch.nn.Softmax(dim=1)
    graph = dgl.add_self_loop(graph)
    graph = graph.to(device)
    print(name + ' number of nodes is : ', graph.num_nodes())
    print(name + ' number of edges is : ', graph.num_edges())
    if feat_type == 'ones_feat':
        graph.ndata['feat'] = torch.ones(graph.num_nodes(), 1).float().to(device)
    elif feat_type == 'noise_feat':
        graph.ndata['feat'] = torch.randn(graph.num_nodes(), 1).float().to(device)
    elif feat_type == 'identity_feat':
        graph.ndata['feat'] = compute_identity(torch.stack(graph.edges(), dim=0), graph.number_of_nodes(), k).float().to(device)
    else:
        graph.ndata['feat'] = graph.in_degrees().unsqueeze(1).float().to(device)
    
    logits = model(graph)
    result = softmax(logits)
    ans.append(result[0])
    print('{} {} : '.format(name, feat_type), end='')
    for i, key in enumerate(list(param.keys())):
        print('{}. {:.4f} '.format(key, result[0][i]), end='')
    print()
    
    return ans

def radar_plot(ans, names, output_path, feat_type, param):
    ans = torch.tensor(ans)
    col =  list(param.keys())
    df = pd.DataFrame(ans.numpy()*100.0, index=names, columns=col)
    df.reset_index(inplace=True)

    # Plot data
    for i in range(len(df)):
        # number of variable
        categories=list(df)[1:]
        N = len(categories)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        # Initialise the spider plot
        fig, ax = plt.subplots(nrows=1, ncols=1, subplot_kw=dict(projection='polar'),figsize=(15, 6))
        
        # Draw one axe per variable + add labels
        ax.set_xticks(angles[:-1], categories, color='grey', size=8)
        # Draw ylabele
        values=df.loc[i].drop('index').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid')
        ax.fill(angles, values, 'b', alpha=0.1)
        ax.set_rlabel_position(0)
        ax.set_yticks([25], ["25"], color="grey", size=7)
        ax.set_ylim(0,100)
        ax.set_title('{} model1'.format(feat_type))
        fig.suptitle(names[i])
        plt.show()
        plt.savefig('{}/radar{}_{}.png'.format(output_path, names[i], feat_type))

def test_networks(model, args, param):
    # Open the file in read mode
    ans = []
    names = []
    with open('links.txt', 'r') as file:
        # Iterate over each line in the file
        f = 0
        for line in file:
            # Print the line
            if len(line) <= 1:
                f = 1
            else :
                if f == 1:
                    print()
                    print("field : "+line)
                    f = 0
                else:
                    graph, name = download_Stanford_network(line[:-1])
                    ans.append((test_network_diff_nfeat(model, graph, name, args['device'], args['feat_type'], args['k'], param))[0].tolist())
                    names.append(name[:-7])
                    print()
            torch.cuda.empty_cache()
    df = pd.DataFrame(ans, columns=list(param.keys()))
    df.index = index
    print(df)
    df.to_csv("{}/stanford_output_testing.csv".format(args['output_path']))
    index = [(args['architecture'], args['feat_type'], name) for name in names]
    
    radar_plot(ans, names, args['output_path'], args['feat_type'], param)
               

if __name__ == "__main__":
    
    args2 = parse_args()
    param = torch.load('{}/parameters_generated_data.pth'.format(args2.dataset_path))

    if args2.dist_draw == True:
        stanford_degree_dist_plots()
    else:
        with open(args2.args_file, 'r') as f:
            args = json.load(f)
        args = args['hyper-parameters']
        model_op = get_network(args['architecture'])
        
        model = model_op(
            in_dim=args['num_feature'],
            hidden_dim=args['hidden_dim'],
            out_dim=args['num_classes'],
            num_layers=args['num_layers'],
            pool_ratio=args['pool_ratio'],
            dropout=args['dropout'],
        ).to(args['device'])
        model.load_state_dict(torch.load(args2.model_weights_path))
        args['feat_type'] = args2.feat_type
        test_networks(model, args, param)