# On the Power of Graph Neural Networks and Feature Augmentation Strategies to Classify Social Networks

## Overview
This paper studies four Graph Neural Network architectures (GNNs) for a graph classification task on a  synthetic dataset created using classic generative models of Network Science. Since the  synthetic networks do not contain (node or edge) features, five different augmentation strategies (artificial feature types) are applied to nodes. All combinations of the 4 GNNs (GCN with Hierarchical and Global aggregation, GIN and GATv2) and the 5 feature types (constant 1, noise, degree, normalized degree and ID -- a vector of the number of cycles of various lengths) are studied and their performances compared as a function of the hidden dimension of artificial neural networks used in the GNNs. The generalisation ability of these models is also analysed using a second synthetic network dataset (containing networks of different sizes). Our results point towards the balanced importance of the computational power of the GNN architecture and the the information level provided by the artificial features. GNN architectures with higher computational power, like GIN and GATv2, perform well for most augmentation strategies. On the other hand, artificial features with higher information content, like ID or degree, not only consistently outperform other augmentation strategies, but can also help GNN architectures with lower computational power to achieve good performance.

## Synthetic-Benchmark-for-Graph-Classification Dataset Description
Two datasets were created, both containing synthetic networks produced by abstract generative models from Network Science:

- The first dataset was used to train the studied GNN models and to test their performance on unseen samples.
- The second dataset was solely used for testing the generalization ability of the trained models.

### Network Generation Details

The networks in these datasets were generated using the Erdős-Rényi (ER), Watts-Strogatz (WS), and Barabási-Albert (BA) models. The parameters were chosen to emphasize the characteristic features of each network family while maintaining similar basic network statistics across the dataset.

The characteristic features considered:
- Average path length ($\ell$)
- Transitivity ($T$)
- Shape of the degree distribution

For average path length and transitivity, two cases were considered: low ($\ell < \log(N)$ and $T < d$) and high (otherwise), where $N$ represents the number of nodes, and $d$ denotes density.

### Network Types

There are 8 possible combinations of the high and low cases of these three properties. The dataset covers both scale-free (SF) and non-scale-free (NSF) distributions across various network types.

### Dataset Sizes and Splitting

Two datasets were created:
- **Small-Sized Graphs Dataset (Small Dataset)**: Contains 250 samples from each of the 8 network types, totaling 2000 synthetic networks. Network sizes ($N$) were randomly selected from the $[250, 1024]$ interval.
- **Medium-Sized Graphs Dataset (Medium Dataset)**: Also contains 250 samples from each of the 8 network types, totaling 2000 synthetic networks. Network sizes ($N$) were randomly selected from the $[1024, 2048]$ interval.

During training, the Small Dataset was split into 1600 training graphs, 200 validation graphs, and 200 testing graphs (80%/10%/10% ratio). The Medium Dataset serves as additional test data to evaluate the models' generalizing ability.

### Parameters and Statistics

For each network type in both datasets, specific parameters were chosen to achieve the desired average degree. Main network statistics are reported in tables for reference.

### Network Types Summary

The table below summarizes the 8 network types included in the synthetic datasets and their main features. These labels correspond to the generating models (ER=Erdős-Rényi, WS=Watts-Strogatz, BA=Barabási-Albert, and GRID=regular lattice) along with their average degree values and characteristics related to degree distribution, average path length ($\ell$), and transitivity ($T$).

| **Label**   | **Degree Distribution** | **$\ell$** | **$T$** | **Avg Degree** |
|-------------|-------------------------|------------|---------|----------------|
| $ER_{low}$  | NSF                     | Low        | Low     | 4              |
| $ER_{high}$ | NSF                     | Low        | Low     | 8              |
| $WS_{low}$  | NSF                     | Low        | High    | 4              |
| $WS_{high}$ | NSF                     | Low        | High    | 8              |
| $BA_{low}$  | SF                      | Low        | Low     | 4              |
| $BA_{high}$ | SF                      | Low        | Low     | 8              |
| $GRID_{low}$| NSF                     | High       | Low     | 4              |
| $GRID_{high}$| NSF                    | High       | High    | 8              |

This table helps understand the different network types generated and their key characteristics, aiding in the analysis and understanding of the datasets.

- **DataSet used**: The dataset is available in the 'datasets.zip' file within this repository. Alternatively, you can access it via the ([Link to the dataset](https://www.kaggle.com/datasets/geuttalawalid/synthetic-benchmark-for-graph-classification/data)).

## Data Generation Code

- **File**: `create_dataset.py`
- **Description**: This Python script (`create_dataset.py`) contains functions to generate synthetic datasets for the study. The code utilizes various libraries such as NetworkX, NumPy, Torch, Pandas, DGL (Deep Graph Library), and others to create synthetic graphs with specific characteristics.

The script includes functions like `generate_parameters`, `generate_data`, `create_moore_2d_grid_graph`, and `create_manhattan_2d_grid_graph`, which collectively generate different types of graphs (Erdős-Rényi, Watts-Strogatz, Barabási-Albert, and grid-based graphs) based on specified parameters.

- `generate_parameters`: Generates parameters for different graph types based on specific distributions and characteristics.
- `generate_data`: Uses the generated parameters to create synthetic graphs for various types such as Erdős-Rényi, Watts-Strogatz, Barabási-Albert, and grid-based graphs.
- `create_moore_2d_grid_graph` and `create_manhattan_2d_grid_graph`: Functions to create 2D grid graphs with different topologies.

The script also contains functionalities to calculate graph statistics like average degree, density, transitivity, average shortest path, max degree, and degree variance for the generated graphs. These statistics are then compiled into a Pandas DataFrame (`create_DF`) and saved as CSV files and text files (`add_summary`).

The generated data includes files such as `info_about_graphs.csv`, `summary.txt`, box plots, and a file (`parameters_generated_data.pth`) containing a dictionary of parameters used for data generation. Additionally, a README file (`README.md`) is created within the `data` folder, providing guidance on how to use and interpret the dataset components.

To reproduce the dataset generation, one can execute this script and follow the instructions provided within the README file in the `data` folder.


## Model Generation Code

- **File**: `model_generation_code.py`
- **Description**: Detail the code used to generate and train the models using the dataset. Include libraries, parameters, and explanations of the model architecture or algorithms used.

### SAGNetworkHierarchical

This class defines a Self-Attention Graph Pooling Network with hierarchical readout based on the paper [Self Attention Graph Pooling](https://arxiv.org/pdf/1904.08082.pdf). It consists of several graph convolution layers (`ConvPoolBlock`) and an MLP aggregator (`MLP`). The `forward` method performs the forward pass through these layers.

### SAGNetworkGlobal

Another variation of the Self-Attention Graph Pooling Network, this one utilizes global readout and graph convolution layers (`GraphConv`). It also uses pooling methods (`SAGPool`, `AvgPooling`, `MaxPooling`) and an MLP aggregator (`MLP`) for computation.

### GAT

This class implements a Graph Attention Network (GAT) using multi-head attention (`GATv2Conv`) and sum pooling over all nodes in each layer. It contains multiple layers and uses BatchNorm, ReLU activation, and dropout for regularization.

### GIN

The GIN (Graph Isomorphism Network) class implements a model with GINConv layers, batch normalization, and MLP aggregators for sum pooling.

### get_network

This function acts as a factory to retrieve the desired network architecture based on the specified type, including hierarchical (`SAGNetworkHierarchical`), global (`SAGNetworkGlobal`), GAT (`GAT`), or GIN (`GIN`). It raises a ValueError for unsupported network types.

## Usage Instructions

### Requirements
- **Software Versions**:
  - `dgl` (with specific installation links for CUDA versions if applicable)
  - `dglgo`
  - `torch_geometric`
  - `torch`, `torch-scatter`, `torch-sparse`
  - `pytorch_lightning`
  - `networkit`
  - `networkx`
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `sklearn`

### Instructions

1. **Environment Setup**:
   - If using CUDA:
     - Install `dgl` and `dglgo`:
       ```bash
       pip install --pre dgl -f https://data.dgl.ai/wheels/cu116/repo.html
       pip install --pre dglgo -f https://data.dgl.ai/wheels-test/repo.html
       ```
   - If using CPU:
     - Install `dgl`:
       ```bash
       pip install dgl
       ```
   - Install other dependencies:
     ```bash
     pip install torch_geometric torch-scatter torch-sparse pytorch_lightning networkit networkx numpy pandas matplotlib sklearn
     ```

2. **Code Usage**:
   - Initialize the device:
     ```python
     device = 'cuda'  # or 'cpu'
     ```
   - Install necessary packages based on the selected device.
   - Import required libraries and set the necessary environment variables:
     ```python
     import os
     import torch

     os.environ['TORCH'] = torch.__version__
     # Other imports and settings
     ```
   - Use the provided code for graph generation, dataset creation (`GraphDataset`), and model training (`create_dataset.py`).

### Example Usage

To generate and train models, you can use the `main.py` script with various parameters to customize the process:

- **Dataset Configuration**:
  - `--dataset`: Name of the dataset, used for labeling information post-training.
  - `--feat_type`: Choice of feature types (`ones_feat`, `noise_feat`, `degree_feat`, `identity_feat`, `norm_degree_feat`).

- **Training Parameters**:
  - `--batch_size`: Batch size for training.
  - `--lr`: Learning rate for optimization.
  - `--weight_decay`: Weight decay applied to the learning rate during epochs.
  - `--hidden_dim`: Hidden size, determining the number of neurons in each hidden layer.
  - `--dropout`: Dropout ratio used in the model.
  - `--epochs`: Maximum number of training epochs.
  - `--patience`: Patience for early stopping (`-1` for no early stopping).

- **Device and Model Configuration**:
  - `--device`: Device choice between `cuda` or `cpu`.
  - `--architecture`: Model architecture (`gin`, `gatv2`, `gcn`, `sage`, `gcnv2`, `cheb`).
  - `--num_layers`: Number of convolutional layers.

- **Output and Saving**:
  - `--output_path`: Path to store the model output.
  - `--save_hidden_output_train`: Option to save output before applying activation function during training.
  - `--save_hidden_output_test`: Option to save output before applying activation function during testing/validation.
  - `--save_last_epoch_hidden_output`: Save last epoch hidden output only (applies to both train and test if set to `True`).
  - `--save_last_epoch_hidden_features_for_nodes`: Save last epoch hidden features of nodes (applies to both train and test if set to `True`).

- **Additional Configurations**:
  - `--k`: Depth control for generating ID features (for ID-GNN).
  - `--output_activation`: Output activation function.
  - `--optimizer_name`: Optimizer type (default is `Adam`).
  - `--loss_name`: Loss function correlated to the optimization function.

- **Tracking Progress**:
  - `--print_every`: Log training details every `k` epochs (`-1` for silent training).

- **Miscellaneous Parameters**:
  - `--num_trials`: Number of trials.
  - `--current_epoch`: The current epoch.
  - `--current_trial`: The current trial.
  - `--activate`: Activate saving the learned node features in the test dataset.
  - `--current_batch`: The current batch.

Utilize these parameters to customize the dataset generation, model training, and testing processes according to your requirements.


4. **Notes**:
   - Ensure that the required datasets are available or generated as needed before training the models.
   - Adjust hyperparameters and model architectures in the code as necessary for specific experiments or tasks.
   - Consult the provided `utils.py` file for additional utility functions and manipulations.

## Citation
- **Cite the Paper**:
- @article{guettala2024power,
  title={On the Power of Graph Neural Networks and Feature Augmentation Strategies to Classify Social Networks},
  author={Guettala, Walid and Guly{\'a}s, L{\'a}szl{\'o}},
  journal={arXiv preprint arXiv:2401.06048},
  year={2024}
}

## Contact Information
- **Author**: guettalawalid@inf.elte.hu

