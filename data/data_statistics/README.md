In the data folder, you will find several components that contain information about the dataset. These components are described below:

1. dgl_graph.bin and info.pkl: These are the two main files that contain the structure of the graphs and additional information about the dataset. The dgl_graph.bin file contains the graph structure, and the info.pkl file contains the number of features and the number of classes for the data. To load these files, you can use the `load()` method from the GraphDataset class in the DGL library.

2. info_about_graphs.csv and summary.txt: These files contain additional information about the graphs in the dataset, including the number of nodes, edges, degree, and density. Note that our graphs are undirected.

3. Box plots: Lastly, you will find three box plots that provide a visual representation of the dataset. These plots can help you to better understand the distribution of the data and identify any potential outliers.

4. parameters_generated_data.pth: This file contains a dictionary of parameters used to generate the data. You can load it using the torch.load() method.

To use this dataset, you can load the dgl_graph.bin and info.pkl files using the GraphDataset class in the DGL library. You can also refer to the info_about_graphs.csv and summary.txt files to get additional information about the graphs. Finally, you may find it helpful to review the box plots to gain a better understanding of the data distribution.
