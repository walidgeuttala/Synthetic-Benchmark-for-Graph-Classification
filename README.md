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

- **Kaggle Link**: [Link to the dataset](https://www.kaggle.com/datasets/geuttalawalid/synthetic-benchmark-for-graph-classification/data)

## Data Generation Code
- **File**: `data_generation_code.py`
- **Description**: Explain the code used to generate the dataset. Include necessary instructions or dependencies needed to run this code.

## Model Generation Code
- **File**: `model_generation_code.py`
- **Description**: Detail the code used to generate and train the models using the dataset. Include libraries, parameters, and explanations of the model architecture or algorithms used.

## Model Testing Code
- **File**: `model_testing_code.py`
- **Description**: Describe the code used for testing the models. Explain how the performance metrics were calculated and how the models were evaluated.

## Usage Instructions
- **Requirements**: List any specific software versions, libraries, or dependencies required to reproduce your results.
- **Instructions**: Provide step-by-step instructions on how to generate the dataset, train the models, and test them using the provided code.

## Citation
- **Cite the Paper**: Include the citation for your paper so that others can properly reference your work.

## References
- **List of References**: Include citations for any external sources or libraries used in your code or dataset creation.

## Contact Information
- **Author**: Provide contact details or a way for other researchers to reach out for questions or clarifications.

