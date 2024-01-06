import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


var = ''
df1 = pd.read_csv(var+'test_resutls_small_networks.csv', index_col=False)
df2 = pd.read_csv(var+'test_resutls_large_networks.csv', index_col=False)
df2.drop(df2.columns[0], axis=1, inplace=True)
df1.drop(df1.columns[0], axis=1, inplace=True)

merged_df = pd.merge(df1, df2, on=['architecture', 'feat_type', 'hidden_dim', 'num_layers'], how='inner')
merged_df = merged_df[['architecture', 'feat_type', 'hidden_dim', 'num_layers', 'test_acc_x', 'test_acc_y']]
merged_df = merged_df[merged_df['num_layers'] == 4]
merged_df = merged_df.rename(columns={'test_acc_x': 'test_acc', 'test_acc_y': 'second_test_acc'})

df = pd.read_csv('dafa.csv', index_col=False)

def func(df):
    # Assuming your DataFrame is named 'df'
    sns.set_style("whitegrid")

    # Define the size of the plot
    plt.figure(figsize=(14, 10))

    # Get unique architecture values
    unique_architectures = df['architecture'].unique()
    unique_feat_types = df['feat_type'].unique()
    unique_feat_types2 = ['identity','degree', 'noise', 'ones', 'norm_degree']
    # Get a color palette with different colors for each feat_type
    palette = sns.color_palette('husl', n_colors=len(unique_feat_types))
    line_styles = ['-', '--', ':', '-.', (8, 2)]
    markers = ['o', 's', '^', 'P', 'X']

    # Create a separate plot for each architecture
    for arch in unique_architectures:
        filtered_data = df[df['architecture'] == arch]
        plt.figure(figsize=(14, 10))  # Set size for the current plot

        if arch == 'gin':
            archd = 'GIN'
        elif arch == 'gat':
            archd = 'GATv2'
        elif arch == 'global':
            archd = 'Global'
        else:
            archd = 'Hierarchical'
        for i, feat in enumerate(unique_feat_types):
            label = f'{archd} {feat}'

            # Use a different color for each feat_type
            color = palette[i]
            if isinstance(line_styles[i], tuple):
                sns.lineplot( x='hidden_dim', y='test_acc', label=f'{unique_feat_types2[i]} (Small Dataset)', data=filtered_data[filtered_data['feat_type'] == feat], marker=markers[i], linewidth=2.0, color='black', markersize=20, dashes=line_styles[i])
                sns.lineplot( x='hidden_dim', y='second_test_acc', label=f'{unique_feat_types2[i]} (Medium Dataset)', data=filtered_data[filtered_data['feat_type'] == feat], marker=markers[i], linewidth=2.0, color='gray', markersize=20, dashes=line_styles[i])
            else:
                sns.lineplot(x='hidden_dim', y='test_acc', label=f'{unique_feat_types2[i]} (Small Dataset)', data=filtered_data[filtered_data['feat_type'] == feat], marker=markers[i], linewidth=2.0, color='black', markersize=20, linestyle=line_styles[i])
                sns.lineplot(x='hidden_dim', y='second_test_acc', label=f'{unique_feat_types2[i]} (medium Dataset)', data=filtered_data[filtered_data['feat_type'] == feat], marker=markers[i], linewidth=2.0, color='gray', markersize=20, linestyle=line_styles[i])


        plt.xscale('log', base=2)  # Set x-axis to logarithmic scale with base 2
        plt.xlabel('Hidden Dimension (log2 scale)', fontsize=24)

        # Set the x-tick labels to powers of 2
        min_hidden_dim = df['hidden_dim'].min()
        max_hidden_dim = df['hidden_dim'].max()
        x_ticks = [2 ** i for i in range(int(np.log2(min_hidden_dim)), int(np.log2(max_hidden_dim) + 1))]
        plt.xticks(x_ticks, labels=[str(val) for val in x_ticks], fontsize=20)

        plt.ylabel('Accuracy', fontsize=24)
        plt.yticks([i/10 for i in range(11)], fontsize=20)
        plt.title(f'Architecture: {archd}', fontsize=24)
        plt.legend(title='Legend', prop={'size': 18})
        filename = f'{arch}_plot.pdf'
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.show()
func(df)
print(merged_df)
merged_df.to_csv('all_outputs.csv', index=False)