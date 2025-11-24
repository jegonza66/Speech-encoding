"""
This script runs Hubert, WavLm and Wav2Vec2 representations.
For each DNN, it varies n_components and registers the average correlation value at each layer.
The idea is to plot many curves (one for each n_components), where the average correlation per layer is plotted.
"""
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import scienceplots
import numpy as np
import imageio
import shutil
import json
import mne
import os

plt.style.use(['science'])
rc('text', usetex=True)

import config
from utils.general_functions import (
    load_pickle, dump_pickle, convert_numpy_keys
)

from load import main_parallel as main_load
from validation import main as main_val
from main import main as main_main

SAVE_PATH = Path("output/mtrf-ridge/analysis/DNN_component_analysis")
BACKBONES = ["hubert", "wav2vec2", "wavlm"]#, 'whisper'] 

EXTRA_COMPONENTS = np.array([64, 128, 200])
COMPONENTS = np.array([12, 16, 21, 24, 32])
LAYERS = np.arange(24) # 23 

N_COMPONENTS_TO_PLOT = 32

correlations = {
    backbone: {
        n_components: {
                layer: None for layer in LAYERS
            } for n_components in COMPONENTS
    } for backbone in BACKBONES
}
for backbone in BACKBONES:
    if backbone == 'wav2vec2':
        for n_components in EXTRA_COMPONENTS:
            correlations[backbone][n_components] = {
                6: None
            }
    # elif backbone != 'whisper':
    else:
        for n_components in EXTRA_COMPONENTS:
            correlations[backbone][n_components] = {
                10: None
            }

# Load computed data and create missing entries if so
try:
    checkpoint_path = SAVE_PATH / "checkpoint_DNN_component_correlations.pkl"
    if not checkpoint_path.is_file():
        raise FileNotFoundError
    data = load_pickle(path=checkpoint_path)
    correlations = data["correlations"]
    for backbone in BACKBONES:
        if backbone not in correlations:
            correlations[backbone] = {
                n_components: {
                    layer: None for layer in LAYERS
                } for n_components in COMPONENTS
            }
        else:
            for n_components in COMPONENTS:
                if n_components not in correlations[backbone]:
                    correlations[backbone][n_components] = {
                        layer: None for layer in LAYERS
                    }
                else:
                    for layer in LAYERS:
                        if layer not in correlations[backbone][n_components]:
                            correlations[backbone][n_components][layer] = None
        # Special case for wav2vec2
        if backbone == 'wav2vec2':
            for n_components in EXTRA_COMPONENTS:
                if n_components not in correlations[backbone]:
                    correlations[backbone][n_components] = {
                        6: None
                    }
        # elif backbone != 'whisper':
        for n_components in EXTRA_COMPONENTS:
            if n_components not in correlations[backbone]:
                correlations[backbone][n_components] = {
                    10: None
                }
except FileNotFoundError:
    print("No previous checkpoint found, starting from scratch.")

for backbone in BACKBONES:
    components_to_use = np.concatenate(
        (COMPONENTS.copy(), EXTRA_COMPONENTS.copy())
    )
    for n_components in components_to_use:  
        if n_components > 32:
            if backbone == 'wav2vec2':
                layers_to_use = [6]
            # elif backbone != 'whisper':
            #     layers_to_use = [10]
            else:
                # continue
                layers_to_use = [10]
        else:
            layers_to_use = LAYERS.copy()
        for layer in layers_to_use:

            # Check if already computed
            stimuli = f'{n_components}DNNs{layer}-{backbone}'
            if correlations[backbone][n_components][layer] is not None:
                print(f"Skipping already computed {stimuli}")
                continue
            else:
                print(
                    f'\n\n\n\tProcessing {n_components} COMPONENTS, layer {layer}, backbone {backbone}\n',
                    f'\n\tStimuli:\t{stimuli}\n'
                )
            
            # Load data (this will preprocess and save the data if not found)
            _ = main_load(
                situations=['External'],
                bands=['Broad'],
                stimuli=[stimuli],
                save_results=True,
                number_of_workers=8
            )
            # Validation to get optimal alpha
            alphas = main_val(
                situations=['External'],
                stimuli=[stimuli],
                bands=['Broad'],
                save_results=True,
                no_figures=True,
                n_folds=2 if n_components > 200 else 10,
                recompute=False
            )['External']['Broad'][stimuli]
            
            # Main results with optimal alpha
            main_results = main_main(
                situations=['External'],
                stimuli=[stimuli],
                bands=['Broad'],
                save_results=False,
                set_alpha=None,
                same_validation_subjects=False,
                no_figures=True
            )['External']['Broad'][stimuli]

            # Store correlations
            correlations[backbone][n_components][layer] = main_results['average_correlation_subjects']

            # Save checkpoint
            SAVE_PATH.mkdir(parents=True, exist_ok=True)
            dump_pickle(
                path=SAVE_PATH / "checkpoint_DNN_component_correlations.pkl",
                obj={
                    "correlations": correlations,
                },
                rewrite=True,
                verbose=True
            )
            # Legible format
            with open(SAVE_PATH / "checkpoint_DNN_component_correlations.json", 'w') as f:
                json_data = {
                    "correlations": convert_numpy_keys(correlations),
                }
                json.dump(json_data, f, indent=4)

            # Remove preprocessed data to save space
            try:
                dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{stimuli}')
                shutil.rmtree(dir_to_remove, ignore_errors=True)
            except Exception as e:
                raise(f"Could not remove directory {dir_to_remove}: {e}")

# Make plots comparing different n_components for each backbone
fig, axes = plt.subplots(
    nrows=1, ncols=3, 
    figsize=(12, 5), 
    sharey=True, 
    tight_layout=True
)
# fig.suptitle("DNN Layer Correlation Analysis: Hubert vs WavLM vs Wav2Vec2", fontsize=16)
for idx, backbone in enumerate(['hubert', 'wavlm', 'wav2vec2']):#, 'whisper']):
    ax = axes[idx]

    for n_components in COMPONENTS:
        means = [correlations[backbone][n_components][layer].mean() for layer in LAYERS]
        sems = [correlations[backbone][n_components][layer].mean(axis=1).std(ddof=1)/np.sqrt(18) for layer in LAYERS]
        linewidth = 3 if n_components == 32 else 1
        alpha = 0.2 if n_components == 32 else 0.1
        # alpha=.1
        if backbone == 'wavlm':
            ax.plot(LAYERS, means, label=f'{n_components} components', linewidth=linewidth)
        else:
            ax.plot(LAYERS, means, linewidth=linewidth)
        ax.fill_between(LAYERS, np.array(means)-np.array(sems), np.array(means)+np.array(sems), alpha=alpha)
        ax.set_xticks(LAYERS[1::2])
        ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
        ax.set_xlabel("DNN Layer")
        ax.set_title(f"{backbone.capitalize()}")
    if backbone == 'wav2vec2':
        for n_components in EXTRA_COMPONENTS:
            means = correlations[backbone][n_components][6].mean()
            sems = correlations[backbone][n_components][6].mean(axis=1).std(ddof=1)/np.sqrt(18)
            ax.scatter(6, means, label=f'{n_components} components')
            ax.errorbar(6, means, yerr=sems, capsize=5) 
            ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
    else:
        for n_components in EXTRA_COMPONENTS:
            means = correlations[backbone][n_components][10].mean()
            sems = correlations[backbone][n_components][10].mean(axis=1).std(ddof=1)/np.sqrt(18)
            ax.scatter(10, means)#, label=f'{n_components} components')
            ax.errorbar(10, means, yerr=sems, capsize=5) 
            ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
axes[0].set_ylabel("Average Correlation")
legend = fig.legend(
    title="Number of components", 
    # loc='lower center', 
    fontsize=12,
    frameon=True, 
    bbox_to_anchor=(0.679, 0.32),
    ncol=4
)
fig_save_path = Path("figures/analysis/DNN_layer_correlation")
fig_save_path.mkdir(parents=True, exist_ok=True)
fig.savefig(
    fig_save_path / "hubert_wavlm_wav2vec2_all_components_layer_correlation.png",
    dpi=600,
    transparent=True
)

# =========
# TOPOPLOTS

# ========================================================================
# Make matrix correlations for all layers and channels at fix n_components
fig, axes = plt.subplots(
    nrows=1, ncols=len(BACKBONES),
    figsize=(12, 5), 
    constrained_layout=True,
    sharey=True, 
)
correlations_matrices = []
for idx, backbone in enumerate(BACKBONES):
    axes[idx].set_title(f"{backbone.capitalize()}")
    axes[idx].tick_params(axis='both', labelsize=15)
    correlation_per_layer_matrix = np.zeros((128, len(LAYERS)))
    for layer in LAYERS:
        correlation_per_layer_matrix[:, layer] = correlations[backbone][N_COMPONENTS_TO_PLOT][layer].mean(axis=0)
    correlations_matrices.append(correlation_per_layer_matrix)
max_ = max([cm.max() for cm in correlations_matrices])
min_ = min([cm.min() for cm in correlations_matrices])
for idx, (backbone, correlation_per_layer_matrix) in enumerate(zip(BACKBONES, correlations_matrices)):
    im = axes[idx].imshow(
        correlation_per_layer_matrix,
        vmin=min_, vmax=max_,
        cmap='viridis',
        aspect='auto'
    )
    axes[idx].set_xlabel("DNN Layer", fontsize=15)

# Add one shared horizontal colorbar for all matrices
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.15, pad=0.07)
cbar.set_label("Correlation", rotation=0, labelpad=10, fontsize=15)
cbar.ax.tick_params(labelsize=15)
axes[0].set_ylabel("EEG Channels", fontsize=15)

fig.savefig(
    fig_save_path / f"matrix_topoplots_{N_COMPONENTS_TO_PLOT}.png",
    dpi=600,
    transparent=True
)

# =================================================================================
# Make GIF per backbone, mne correlation topomap through layers at fix n_components
for backbone in BACKBONES:
    images = []
    for layer in LAYERS:
        # Get correlation values for this layer
        data = correlations[backbone][N_COMPONENTS_TO_PLOT][layer].mean(axis=0)  # shape: (128,)
        # Plot topomap
        fig, ax = plt.subplots(figsize=(5, 5))
        mne.viz.plot_topomap(
            data, config.info_mne, axes=ax, show=False, cmap='Reds', vlim=(data.min(), data.max()),
            contours=0, sensors=True
        )
        ax.set_title(f"{backbone.capitalize()} - Layer {layer} - Avg. Correlation {data.mean():.2f}", fontsize=14)
        fname = fig_save_path / "aux" / f"{backbone}_layer{layer}_{N_COMPONENTS_TO_PLOT}.png"
        fname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fname, dpi=120, bbox_inches='tight')
        plt.close(fig)
        images.append(imageio.imread(fname))
    # Save GIF
    gif_file = fig_save_path / f"{backbone}_{N_COMPONENTS_TO_PLOT}_topomap_layers.gif"
    imageio.mimsave(
        gif_file, images, fps=1.5, format='GIF'
    )

# ======================================================================
# Matrix comparison with classic attributes correlations across channels
classic_attributes = ['Envelope', 'Spectrogram-21', 'Phonemes-Discrete']

classic_correlations = {
    attr: load_pickle(
        path=Path(f"output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/{attr}.pkl")
    )['average_correlation_subjects'].mean(axis=0) for attr in classic_attributes
}

fig, axes = plt.subplots(
    nrows=1, ncols=len(BACKBONES),
    figsize=(12, 5), 
    constrained_layout=True,
    sharey=True, 
)
correlations_matrices = []
for idx, backbone in enumerate(BACKBONES):
    axes[idx].set_title(f"{backbone.capitalize()}")
    axes[idx].tick_params(axis='both', labelsize=15)
    correlation_per_layer_matrix = np.zeros((128, len(LAYERS)))
    for layer in LAYERS:
        correlation_per_layer_matrix[:, layer] = correlations[backbone][N_COMPONENTS_TO_PLOT][layer].mean(axis=0)
    
    # Add classic attributes as new columns
    n_classic = len(classic_attributes)
    extended_matrix = np.zeros((128, len(LAYERS) + n_classic))
    extended_matrix[:, :len(LAYERS)] = correlation_per_layer_matrix
    for i, attr in enumerate(classic_attributes):
        extended_matrix[:, len(LAYERS) + i] = classic_correlations[attr]
    correlations_matrices.append(extended_matrix)
max_ = max([cm.max() for cm in correlations_matrices])
min_ = min([cm.min() for cm in correlations_matrices])
for idx, (backbone, correlation_per_layer_matrix) in enumerate(zip(BACKBONES, correlations_matrices)):
    im = axes[idx].imshow(
        correlation_per_layer_matrix,
        vmin=min_, vmax=max_,
        cmap='viridis',
        aspect='auto'
    )
    axes[idx].set_xlabel("DNN Layer + Classic Attributes", fontsize=15)
    axes[idx].set_xticks(
        ticks=np.arange(len(LAYERS) + n_classic),
        labels=[*[str(layer) for layer in LAYERS], *classic_attributes],
        rotation=90
    )

# Add one shared horizontal colorbar for all matrices
cbar = fig.colorbar(im, ax=axes, orientation='horizontal', fraction=0.15, pad=0.07)
cbar.set_label("Correlation", rotation=0, labelpad=10, fontsize=15)
cbar.ax.tick_params(labelsize=15)
axes[0].set_ylabel("EEG Channels", fontsize=15)

fig.savefig(
    fig_save_path / f"matrix_topoplots_{N_COMPONENTS_TO_PLOT}_extended.png",
    dpi=600,
    transparent=True
)

# ===================
# Explicit comparison
classic_attributes = ['Envelope', 'Spectrogram-21', 'Phonemes-Discrete']
mapping_attributes = {
    'Envelope': "#D87272",
    'Spectrogram-21': '#87CEEB',
    'Phonemes-Discrete': 'orange'
}
name_map = {
    'Envelope': "Envelope",
    'Spectrogram-21': "Spectrogram",
    'Phonemes-Discrete': "Phonemes"
}
classic_correlations = {
    attr: load_pickle(
        path=Path(f"output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/{attr}.pkl")
    )['average_correlation_subjects'].mean(axis=0) for attr in classic_attributes
}

fig, axes = plt.subplots(
    nrows=1, ncols=len(BACKBONES),
    figsize=(12, 5), 
    constrained_layout=True,
    sharey=True, 
)
correlations_matrices = []
for idx, backbone in enumerate(BACKBONES):
    axes[idx].set_title(f"{backbone.capitalize()}")
    axes[idx].tick_params(axis='both', labelsize=15)
    correlation_per_layer_matrix = np.zeros((128, len(LAYERS)))
    for layer in LAYERS:
        correlation_per_layer_matrix[:, layer] = correlations[backbone][N_COMPONENTS_TO_PLOT][layer].mean(axis=0)

    correlations_matrices.append(correlation_per_layer_matrix)

for idx, (backbone, correlation_per_layer_matrix) in enumerate(zip(BACKBONES, correlations_matrices)):
    with_att_corr = {
        key: [] for key in classic_attributes
    }
    for key in classic_attributes:
        for layer in LAYERS:
            with_att_corr[key].append(
                np.corrcoef(
                    correlation_per_layer_matrix[:, layer],
                    classic_correlations[key]
                )[0, 1]
            )
        axes[idx].plot(
            LAYERS,
            with_att_corr[key],
            label=name_map[key],
            color=mapping_attributes[key],
            marker='o'
        )
    axes[idx].set_xlabel("DNN Layer", fontsize=15)
    axes[idx].set_ylabel("Correlation through channels", fontsize=15)
    axes[idx].legend()
fig.savefig(
    fig_save_path / f"matrix_topoplots_{N_COMPONENTS_TO_PLOT}_correlations_with_att.png",
    dpi=600,
    transparent=True
)
print(f"Topoplot figures saved in {fig_save_path}")