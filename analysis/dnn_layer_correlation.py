import matplotlib.pyplot as plt
import numpy as np, os

from utils.general_functions import load_pickle

# =========
# Load data
DnnsCorrelationPath = lambda x: rf"output\mtrf-ridge\External\correlations\distinct_alpha\tmin-0.2_tmax0.6\Broad\DNNs{x}.pkl"
DnnsTrfsPath = lambda x: rf"output\mtrf-ridge\External\weights\stims_Standarize_EEG_Standarize\distinct_alpha\tmin-0.2_tmax0.6\Broad\DNNs{x}\total_weights_per_subject.pkl"

trfs = {
    n: load_pickle(DnnsTrfsPath(n))['average_weights_subjects']
    for n in range(1, 24)
}
correlations = {
    n: load_pickle(DnnsCorrelationPath(n))['average_correlation_subjects']
    for n in range(1, 24)
}

# ====================
# Correlation analysis
correlation_mean = [
    correlations[n].mean()
    for n in range(1, 24)
]
correlation_std = [
    correlations[n].std()/np.sqrt(128*18)
    for n in range(1, 24)
]

# Plot correlations
plt.figure(figsize=(10, 5))
plt.errorbar(range(1, 24), correlation_mean, yerr=correlation_std, capsize=5, fmt='o', color='black')
plt.xticks(range(1, 24))
plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
plt.xlabel("DNN Layer")
plt.ylabel("Average Correlation")
plt.title("DNN Layer Wav2Vec Correlation Analysis")
os.makedirs("figures/analysis/dnn_layer_correlation", exist_ok=True)
plt.savefig("figures/analysis/dnn_layer_correlation/wav2vec2_layer_correlation.png")
# plt.show()

# ================
# Weights analysis
import mne
import config

# Parámetros fijos de tus TRFs
WIN_MS = (-75.0, 175.0)      # ventana de interés [ms]

use_abs = True  # usa |w|. Si quieres signo: False

values_by_layer = {}
vmax = 0.0

for layer, W in trfs.items():

    # Promedio sobre sujetos y componentes DNN -> (channels=128, time=104)
    W_mean = W.mean(axis=(0, 2))  # (128, 104)

    # Ventana temporal
    mask = (config.times >= WIN_MS[0]) & (config.times <= WIN_MS[1])
    if not np.any(mask):
        raise ValueError(f"Ventana {WIN_MS} ms fuera de rango [{config.times[0]:.1f},{config.times[-1]:.1f}] ms")

    # Vector por canal en la ventana
    W_win = W_mean[:, mask]  # (128, nwin)
    vec = (np.abs(W_win).mean(axis=1) if use_abs else W_win.mean(axis=1))  # (128,)
    values_by_layer[layer] = vec
    vmax = max(vmax, float(np.max(np.abs(vec))))

# Grid de topomapas por capa
layers_sorted = sorted(values_by_layer.keys())
n_layers = len(layers_sorted)
n_rows, n_cols = 4, 6  # hasta 24 capas
fig, axes = plt.subplots(n_rows, n_cols, figsize=(8, 8))
axes = axes.flatten()

for i, layer in enumerate(layers_sorted):
    ax = axes[i]
    data = values_by_layer[layer]
    vlim = (0.0, vmax) if use_abs else (-vmax, vmax)
    mne.viz.plot_topomap(
        data, config.info_mne, axes=ax, contours=0,
        vlim=vlim, cmap=("viridis" if use_abs else "RdBu_r"),
        sensors=True, show=False
    )
    ax.set_title(f"L{layer}", fontsize=9)

# Ocultar celdas sobrantes
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(f"TRF weights topography per layer [{WIN_MS[0]:.0f},{WIN_MS[1]:.0f}] ms", fontsize=12)
# os.makedirs("figures/analysis/dnn_layer_correlation", exist_ok=True)
out_png = rf"figures\analysis\dnn_layer_correlation\wav2vec2_topos_{int(WIN_MS[0])}_{int(WIN_MS[1])}ms.png"
# plt.savefig(out_png, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {out_png}")