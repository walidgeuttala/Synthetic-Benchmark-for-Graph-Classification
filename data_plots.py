import dgl
from data import GraphDataset
import networkx as nx
from test_stanford_networks import stanford_degree_dist_plots
import pandas as pd
import torch 
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset plot statistics")
    parser.add_argument(
        "--types",
        type=str,
        default="degree_dist",
        help="list of names of plots that you want to do",
    )
    parser.add_argument(
        "--draw_stanford_points",
        type=bool,
        default=False,
        help="draw stanford networks points in the plots",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data",
        help="dataset path",
    )

    return parser.parse_args()

def degree_dist(pick, names , args):
    dataset = GraphDataset(device='cpu')
    dataset.load(args.dataset_path)
    data = []
    length = len(dataset) // len(names)
    for i in range(len(names)):
        graphs = dataset.graphs[i*length:(i+1)*length]
        graphs = sorted(graphs, key=lambda g: g.number_of_nodes())
        graphs = graphs[len(graphs)//pick-1::len(graphs)//pick]
        graphs = [dgl.to_networkx(g.to('cpu')) for g in graphs]
        data.append(graphs)

    for i in range(len(names)):
        print('networks are of type  '+names[i])
        for j in range(pick):
            data[i][j] = nx.Graph(data[i][j])
            print('number of nodes : ',data[i][j].number_of_nodes())
            hist = nx.degree_histogram(data[i][j])
            plt.plot(hist)
            plt.xlabel("Degree")
            plt.ylabel("Frequency")
            plt.title("Degree Distribution of "+names[i])
            plt.savefig('{}/Degree Distribution of {}.png'.format(args.dataset_path, names[i]))
            plt.show()

def draw_distribution(data, args):
    # Create a histogram to visualize the distribution
    plt.hist(data, bins=100, color='blue', edgecolor='black')

    # Add labels and a title
    plt.xlabel('X-axis label')
    plt.ylabel('Frequency')
    plt.title('Distribution of {}'.format(data.name))

    # Show the plot
    plt.savefig('{}/Distribution for {} density.png'.format(args.dataset_path, data.name))
    plt.show()

def draw_dist_density(df, length, args):
    density = df.iloc[:,3::df.shape[1]//length]
    for column in density.columns:
        draw_distribution(density[column], args)

def draw_dist_edges(df, length, args):
    density = df.iloc[:,1::df.shape[1]//length]
    for column in density.columns:
        draw_distribution(density[column], args)

def draw_dist_nodes(df, length, args):
    density = df.iloc[:,0::df.shape[1]//length]
    for column in density.columns:
        draw_distribution(density[column], args)

def density_boxplot(df, density, names, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,3::df.shape[1]//length].boxplot()
    n = 1
    if density == None:
        n = 0
    else:
        n = len(density)
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')
    color = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']
    if args.draw_stanford_points == True:
        for i in range(n):
            for j in range(length):
                plt.scatter(j+1, density[i], color=color[i%len(color)],label=names[i])
    # Show the plot
    plt.legend()
    plt.savefig('{}/Boxplot for each column density.png'.format(args.dataset_path))
    plt.show()


def transitivity_boxplot(df, transitivity, names, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,4::df.shape[1]//length].boxplot()
    n = 1
    if transitivity == None:
        n = 0
    else:
        n = len(transitivity)
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')
    color = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']
    if args.draw_stanford_points == True:
        for i in range(n):
            for j in range(length):
                plt.scatter(j+1, transitivity[i], color=color[i%len(color)],label=names[i])
        
    # Show the plot
    plt.legend()
    # Show the plot
    plt.savefig('{}/Boxplot for each column transitivity.png'.format(args.dataset_path))
    plt.show()

def transitivity_by_density(df, transitivity, density, names, length, args):
    plt.figure(figsize=(14, 10))

    multi_index = df.iloc[:,4::df.shape[1]//length].columns
    # Modify the second level by adding "_density"
    multi_index = multi_index.set_levels([level + '_density' for level in multi_index.levels[1]], level=1)
    tran_density = pd.DataFrame(np.where(df.iloc[:,3::df.shape[1]//length].values != 0, df.iloc[:,4::df.shape[1]//length].values / df.iloc[:,3::df.shape[1]//length].values, -1), columns = multi_index)

    tran_density.boxplot()
    n = 1
    if transitivity == None:
        n = 0
    else:
        n = len(transitivity)
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')
    color = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']
    if args.draw_stanford_points == True:
        for i in range(n):
            for j in range(length):
                if density[i] == 0:
                    plt.scatter(j+1, -1, color=color[i%len(color)],label=names[i])
                else:
                    plt.scatter(j+1, transitivity[i]/density[i], color=color[i%len(color)],label=names[i])
    
    # Show the plot
    plt.legend()
    # Show the plot
    plt.savefig('{}/Boxplot for each column transitivity_density.png'.format(args.dataset_path))
    plt.show()

def probability_of_rewiring(param, df, dataset_path):
    probability_of_rewiring = param['WS'][:, 2]
    transitivity = df['WS']['Transitivity'].values
    sorted_indices = np.argsort(probability_of_rewiring)

    probability_of_rewiring = probability_of_rewiring[sorted_indices]
    transitivity = transitivity[sorted_indices]



    plt.figure(figsize=(14, 10), dpi=80)
    plt.scatter(probability_of_rewiring, transitivity, label='Transitivity')
    plt.xlabel('The probability of rewiring each edge')
    plt.ylabel('Transitivity')
    plt.title(" Transitivity plot over over WS networks ploted the mean with min and max noise")
    plt.legend()
    plt.savefig('{}/plots of probability_of_rewiring.png'.format(dataset_path))
    plt.show()

def average_degree_boxplot(df, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,2::df.shape[1]//length].boxplot()

    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')

    # Show the plot
    plt.show()
    plt.savefig('{}/boxplot_of_average_degree.png'.format(args.dataset_path))

def num_edge_boxplot(df, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,1::df.shape[1]//length].boxplot()

    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')

    # Show the plot
    plt.show()
    plt.savefig('{}/boxplot_of_edges.png'.format(args.dataset_path))

def num_nodes_boxplot(df, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,::df.shape[1]//length].boxplot()

    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')

    # Show the plot
    plt.show()
    plt.savefig('{}/boxplot_of_nodes.png'.format(args.dataset_path))

def avg_shortest_path_boxplot(df, avg_shortest_path, length, args):
    plt.figure(figsize=(14, 10))
    df.iloc[:,5::df.shape[1]//length].boxplot()
    n = 1
    if avg_shortest_path == None:
        n = 0
    else:
        n = len(avg_shortest_path)
    # Add labels and title
    plt.xlabel('Columns')
    plt.ylabel('Values')
    plt.title('Boxplot for each column')
    color = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink']
    if args.draw_stanford_points == True:
        for i in range(n):
            for j in range(length):
                plt.scatter(j+1, avg_shortest_path[i], color=color[i%len(color)],label=names[i])
        
    # Show the plot
    plt.legend()
    # Show the plot
    plt.savefig('{}/Boxplot for each column avg_shortest_path.png'.format(args.dataset_path))
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    names2 = args.types.split(',')
    names, density, transitivity, avg_shortest_path = None, None, None, None
    args.draw_stanford_points = False
    if args.draw_stanford_points == True:
        names, density, transitivity, avg_shortest_path = stanford_degree_dist_plots(False)
    df = pd.read_csv('{}/info_about_graphs.csv'.format(args.dataset_path), header=[0,1])
    param = torch.load('{}/parameters_generated_data.pth'.format(args.dataset_path))

    for name in names2:
        if name == "degree_dist":
            degree_dist(10, list(param.keys()), args)
        elif name == "density_boxplot":
            density_boxplot(df, density, names, len(param.keys()), args)
        elif name == "transitivity_boxplot":
            transitivity_boxplot(df, transitivity, names, len(param.keys()), args)
        elif name == "transitivity_by_density":
            transitivity_by_density(df, transitivity, density, names, len(param.keys()), args)
        elif name == "probability_of_rewiring":
            probability_of_rewiring(param, df, args)
        elif name == "average_degree_boxplot":
            average_degree_boxplot(df, len(param.keys()), args)
        elif name == "num_edge_boxplot":
            num_edge_boxplot(df, len(param.keys()), args)
        elif name == "avg_shortest_path_boxplot":
            avg_shortest_path_boxplot(df, avg_shortest_path, len(param.keys()), args)
        elif name == "num_nodes_boxplot":
            num_nodes_boxplot(df, len(param.keys()), args)
        elif name == "draw_dist_density":
            draw_dist_density(df, len(param.keys()), args)
        elif name == "draw_dist_nodes":
            draw_dist_nodes(df, len(param.keys()), args)
        elif name == "draw_dist_edges":
            draw_dist_edges(df, len(param.keys()), args)
        print()

