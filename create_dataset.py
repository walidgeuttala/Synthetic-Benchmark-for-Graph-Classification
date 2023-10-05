
import networkx as nx
import numpy as np
import itertools
import torch
import pandas as pd 
import subprocess
import networkit as nk
import random 

# Generates the parameters for the data generating function
# The seed value is incremented each time so that we can get the same dataset next time we only set the seed parameter
# and other seeds will be seed+=1
def generate_parameters(data_dist = [250] * 5, networks="all", seed=42):
  # Dictionary to save the parameters generated for our data generating function
  param = dict()
  idx = 0
  # General parameters for the 4 graphs
  # Range of generated nodes
  min_n = 25
  max_n = 1024

  # Graph ER parameters
  # Probability for edge creation.
  max_p = 0.05
  # Graph WS parameters
  # Each node is joined with its k nearest neighbors in a ring topology
  min_k = 4
  max_k = 6
  # The probability of rewiring each edge
  min_w = 0.01
  max_w = 0.05
  # Graph BA parameters
  # Number of edges to attach from a new node to existing nodes
  min_m = 1
  max_m = 4 #2

  # Graph HB parameters
  # degree 
  min_degree = 5
  max_degree = 10

  # Graph powerlaw_cluseter
  proba = 1

  # number of nodes x and y
  nodes_min = 12
  nodes_max = 35
  # Graph ER parameters
  seed = np.int64(seed)
  np.random.seed(seed)
  # Generates an array of random integers between min_n and max_n with the size of data_dist[0]
  saved_seed = seed
  n = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  seed += 1
  #Probability for edge creation.
  def my_random_function(x):
    return np.random.uniform(2/x, 4/x, 1)
  
  p = np.vectorize(my_random_function)(n)
  # Store the parameters in a 2D numpy array
  if networks == "all" or "ER" in networks:
    param['ER'] = np.column_stack((n, p))
    idx += 1

  # Graph WS parameters
  np.random.seed(saved_seed)
  # Generates an array of random integers between min_n and max_n with the size of data_dist[idx]
  n = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  seed += 1
  np.random.seed(seed)
  # Generates an array of random integers between min_k and max_k even with the size of data_dist[idx]
  k = np.array(np.random.randint(min_k, max_k+1, data_dist[idx]))
  k = np.floor_divide(k, 2) * 2
  seed += 1
  np.random.seed(seed)
  # Generates an array of random floating point numbers between min_w and max_w with the size of data_dist[idx]
  p = np.random.uniform(min_w, max_w, data_dist[idx])
  seed+=1

  # Store the parameters in a 2D numpy array
  if networks == "all" or "WS" in networks:
    param['WS'] = np.column_stack((n, k, p))
    idx += 1
  # graph BA parameters
  np.random.seed(saved_seed)
  n = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  seed += 1

  np.random.seed(seed)
  m = np.array(np.random.randint(min_m, max_m, data_dist[idx]))
  seed += 1

  # Store the parameters for the BA graph in a dictionary
  if networks == "all" or "BA" in networks:
    param['BA'] = np.column_stack((n, m))

  # graph GRID parameters
  np.random.seed(saved_seed)
  x = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  x = np.array([find_random_close_a_b(n) for n in x])
  np.random.seed(seed)
  
  seed += 1
  # Store the parameters for the SC graph in a dictionary
  if networks == "all" or "grid_low" in networks:
    param['grid_tr_low'] = np.column_stack((x[:, :0], x[:, :1]))
    idx += 1
  np.random.seed(saved_seed)
  x = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  x = np.array([find_random_close_a_b(n) for n in x])
  np.random.seed(seed)
  seed += 1
  if networks == "all" or "grid_high" in networks:
    param['grid_tr_high'] = np.column_stack((x[:, :0], x[:, :1]))
    idx += 1

  
  np.random.seed(seed)
  x = np.array(np.random.randint(nodes_min, nodes_max, data_dist[idx]))
  np.random.seed(seed)
  seed += 1
  y = np.array(np.random.randint(min_degree, max_degree, data_dist[idx]))
  seed += 1
  if networks == "all" or "HB" in networks:
     param["HB"] = np.column_stack((x, y))
     idx += 1
  
  np.random.seed(seed)
  np.random.seed(saved_seed)
  n = np.array(np.random.randint(min_n, max_n, data_dist[idx]))
  seed += 1

  np.random.seed(seed)
  m = np.array(np.random.randint(min_m, max_m, data_dist[idx]))
  seed += 1

  if networks == "all" or "PC" in networks:
     param["PC"] = np.column_stack((n, m))
     idx += 1

  # Reset the seed for random number generation
  np.random.seed(None)

  # Return the graph parameters as a dictionary
  return param

def generate_data(param, data_dist, networks="all"):
    '''
    generate_data: This function generates data for 4 types of graphs given the parameters

    Parameters:
        - param: Dictionary that contains the parameters for each type of graph
        - data_dist: List that contains the count of each type of graph to generate
        - networks: all, or choose the networks you want
    Returns:
        - graphs: List of generated graphs
        - classes: List of 0s, 1s, 2s, and 3s indicating the class of each graph
    '''
    graphs = []
    # ER Graphs
    idx = 0
    if networks == "all" or "ER" in networks:
      for i in range(data_dist[idx]):
          np.random.seed(i)
          graphs.append(nx.gnp_random_graph(int(param['ER'][i, 0]), param['ER'][i, 1], seed=i, directed=False))
      idx += 1
    np.random.seed(None)
    # WS Graphs
    if networks == "all" or "WS" in networks:
      for i in range(data_dist[idx]):
          edges = int(param['WS'][i,0]*param['WS'][i,1])
          graphs.append(nx.watts_strogatz_graph(int(param['WS'][i,0]), int(param['WS'][i,1]), param['WS'][i,2], seed=i))
      idx += 1
    # BA Graphs
    if networks == "all" or "BA" in networks:
      for i in range(data_dist[idx]):
          graphs.append(nx.barabasi_albert_graph(int(param['BA'][i,0]), int(param['BA'][i,1]), seed=i, initial_graph=None))
      idx += 1
    # 2D Grid using manhattan distance low transitivity
    if networks == "all" or "grid_low" in networks:
      for i in range(data_dist[idx]):
        graphs.append(create_manhattan_2d_grid_graph(param['grid_tr_low'][i, 0], param['grid_tr_low'][i, 1], 1))
      idx += 1
    # 2D Grid using moore distance high transitivity
    if networks == "all" or "grid_high" in networks:
      for i in range(data_dist[idx]):
        graphs.append(create_moore_2d_grid_graph(param['grid_tr_high'][i, 0], param['grid_tr_high'][i, 1], 2))
      idx += 1
    # Resetting the seed for numpy
    if networks == "all" or "HB" in networks:
      for i in range(data_dist[idx]):
        graphs.append(hyperbolic_graph(int(param['HB'][i,0]), int(param['HB'][i,1])))
      idx += 1

    if networks == "all" or "PC" in networks:
      for i in range(data_dist[idx]):  
        graphs.append(nx.powerlaw_cluster_graph(int(param['PC'][i,0]), int(param['PC'][i,1]), 1, seed=i))
      idx += 1

    np.random.seed(None)

    return graphs, torch.LongTensor([i for i, x in enumerate(data_dist) for _ in range(x)])


def find_random_close_a_b(n):
  combination = list()
  for a in range(2, n+1):
      if n%a == 0:
         b = n/a
         for x in range(a-10, a+11):
            combination.append((x, b))

  random_index = random.randint(0, len(combination) - 1)
  random_number = random.randint(0, 1)
  a, b = combination[random_index]
  if random_number == 1:
     a, b = b, a
  return np.array([a, b])

def hyperbolic_graph(N,deg,gamma = 3.5):
  G_Nk = nk.generators.HyperbolicGenerator(n = N,k = deg,gamma = 3.5).generate()
  G = convertNkToNx(G_Nk)
  return G

def generate_combinations(choices):
    for combo in itertools.product(*[range(choice) for choice in choices]):
        yield combo

def create_moore_2d_grid_graph(n, m, r):
    if r == 0:
        return nx.Graph()
    G = nx.Graph()
    
    combinations_generator = generate_combinations([n, m])
    for combo in combinations_generator:
        G.add_node(combo)
    
    for node in G.nodes():
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if i != 0 or j != 0:
                    node2 = ((node[0] + i + n) % n, (node[1] + j + m) % m )
                    if node == node2:
                        continue
                    G.add_edge(node, node2)
  
    return G

def create_manhattan_2d_grid_graph(n, m, r):
    if r == 0:
        return nx.Graph()
    G = nx.Graph()
    
    combinations_generator = generate_combinations([n, m])
    for combo in combinations_generator:
        G.add_node(combo)
    
    for node in G.nodes():
        for i in range(-r, r+1):
            for j in range(-r, r+1):
                if 0 < abs(i) + abs(j) <= r:
                    node2 = ((node[0] + i + n) % n, (node[1] + j + m) % m )
                    if node == node2:
                        continue
                    G.add_edge(node, node2)

    return G

def create_DF_transtivity_density(param, graphs, data_dist):
  length = sum(data_dist) // len(param)
  dataFrames = []

  # Add data to the list `edges`
  edges = dict()
  # Index for slicing the list of graphs
  idx = 0

  # Loop over n types of graphs
  for i, key in enumerate(param.keys()):
    # Append the number of edges for a type of graph to `edges`
    edges[key] = np.array([g.number_of_edges() for g in graphs[idx:idx+data_dist[i]]]) * 2
    # Increase the index by the number of graphs of this type
    idx += data_dist[i]

  for key in param.keys():
    if 'grid' in key:
      dataFrames.append(pd.DataFrame({'Num_nodes': param[key][:, 0]*param[key][:, 1], 'Num_edges':edges[key]}))
    else:
      dataFrames.append(pd.DataFrame({'Num_nodes': param[key][:, 0], 'Num_edges':edges[key]}))


  # Concatenate the dataframes along axis 1 (i.e., horizontally)
  df = pd.concat(dataFrames, axis=1)

  # Create a hierarchical column index with the top level being the column names of the original dataframes
  df.columns = pd.MultiIndex.from_product([param.keys(), ['Num_nodes', 'Num_edges']])


  for i, key in enumerate(param):
    ave_degree = df[key]['Num_edges']/df[key]['Num_nodes']
    df.insert(i+i*2+2, column=(key, 'Average_degree'), value=ave_degree)
  density = 2 * df[key]['Num_edges'] / (df[key]['Num_nodes'] * (df[key]['Num_nodes'] - 1))


  for i, key in enumerate(param):
    transitivity = []
    for j in range(length):
      transitivity.append(nx.density(graphs[i*length+j]))
    df.insert(i+i*3+3, column=(key, 'Density'), value=transitivity)

  for i, key in enumerate(param):
    transitivity = []
    for j in range(length):
      transitivity.append(nx.transitivity(graphs[i*length+j]))
    df.insert(i+i*4+4, column=(key, 'Transitivity'), value=transitivity)

  return df
    

def add_summary(df):
  df.to_csv('data/info_about_graphs.csv', index=False)

  summary = df.describe()

  # save the summary statistics to a file
  with open('data/summary.txt', 'w') as f:
      f.write(summary.to_string())

  command = "touch data/README.md"
  subprocess.run(command, shell=True)
  with open('data/README.md', 'w') as f:
    f.write('In the data folder, you will find several components that contain information about the dataset. These components are described below:\n\n1. dgl_graph.bin and info.pkl: These are the two main files that contain the structure of the graphs and additional information about the dataset. The dgl_graph.bin file contains the graph structure, and the info.pkl file contains the number of features and the number of classes for the data. To load these files, you can use the `load()` method from the GraphDataset class in the DGL library.\n\n2. info_about_graphs.csv and summary.txt: These files contain additional information about the graphs in the dataset, including the number of nodes, edges, degree, and density. Note that our graphs are undirected.\n\n3. Box plots: Lastly, you will find three box plots that provide a visual representation of the dataset. These plots can help you to better understand the distribution of the data and identify any potential outliers.\n\n4. parameters_generated_data.pth: This file contains a dictionary of parameters used to generate the data. You can load it using the torch.load() method.\n\nTo use this dataset, you can load the dgl_graph.bin and info.pkl files using the GraphDataset class in the DGL library. You can also refer to the info_about_graphs.csv and summary.txt files to get additional information about the graphs. Finally, you may find it helpful to review the box plots to gain a better understanding of the data distribution.')

    
def get_nk_lcc_undirected(G):
    G2 = max(nx.connected_component_subgraphs(G), key=len)
    tdl_nodes = G2.nodes()
    nodeListMap = dict(zip(tdl_nodes, range(len(tdl_nodes))))
    G2 = nx.relabel_nodes(G2, nodeListMap, copy=True)
    return G2, nodeListMap


def convertNkToNx(G_nk):
    G_nx = nx.Graph()
    for i, j in G_nk.iterEdges():
        G_nx.add_edge(i,j)
    return G_nx
    

