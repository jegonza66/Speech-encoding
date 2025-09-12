import matplotlib.pyplot as plt
import numpy as np, os

from utils.general_functions import load_pickle

# =========
# Load data
backbone = "wav2vec2"
n_components = 18
n_layers = 24
model = 'ridge'
# model = 'ridge-laplacian'

DnnsCorrelationPath = lambda x: rf"output\mtrf-{model}\External-External\correlations\same_alpha\tmin-0.2_tmax0.6\Broad\{n_components}DNNs{x}-{backbone}.pkl"
# DnnsCorrelationPath = lambda x: rf"output\mtrf-{model}\External\correlations\distinct_alpha\tmin-0.2_tmax0.6\Broad\{n_components}DNNs{x}-{backbone}.pkl"

correlations = {
    n: load_pickle(DnnsCorrelationPath(n))['average_correlation_subjects']
    for n in range(1, n_layers)
}

# ====================
# Correlation analysis
correlation_mean = [
    correlations[n].mean()
    for n in range(1, n_layers)
]
correlation_std = [
    correlations[n].std()/np.sqrt(128*18)
    for n in range(1, n_layers)
]
data = [correlations[n].mean(axis=1) for n in range(1, n_layers)]
# Plot correlations
plt.figure(figsize=(10, 5))
quantiles_to_show = [[0.25, 0.75] for _ in data]

parts = plt.violinplot(
    data,
    positions=range(1, n_layers),
    showmeans=True,
    showmedians=False,
    quantiles=quantiles_to_show,
    widths=0.8
)
parts['cmeans'].set_color('black')
plt.xticks(range(1, n_layers))
plt.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
plt.xlabel("DNN Layer")
plt.ylabel("Inter-Subject Correlation")
plt.title(f"DNN Layer {backbone.capitalize()} Correlation Analysis - {n_components} Components")
os.makedirs("figures/analysis/dnn_layer_correlation", exist_ok=True)

# Connect means with black lines
means = [np.mean(d) for d in data]
plt.plot(range(1, n_layers), means, color='black', linewidth=1, alpha=1, zorder=10, label='Means')
plt.legend()
plt.savefig(f"figures/analysis/dnn_layer_correlation/{backbone}_{n_components}_layer_correlation_violin.png")
# plt.show()

plt.figure(figsize=(10, 5))
plt.errorbar(range(1, n_layers), correlation_mean, yerr=correlation_std, capsize=5, color='black', linewidth=1, alpha=1, zorder=10)
plt.scatter(range(1, n_layers), correlation_mean, color='black', s=10,alpha=1, zorder=11, label='Means')
plt.xticks(range(1, n_layers))
plt.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
plt.xlabel("DNN Layer")
plt.ylabel("Inter-Subject Correlation")
plt.title(f"DNN Layer {backbone.capitalize()} Correlation Analysis - {n_components} Components")
os.makedirs("figures/analysis/dnn_layer_correlation", exist_ok=True)

# Connect means with black lines
plt.legend()
plt.savefig(f"figures/analysis/dnn_layer_correlation/{backbone}_{n_components}_layer_correlation.png")
# plt.show()
