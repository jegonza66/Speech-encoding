"""
This script runs Hubert and Wav2Vec2 representations.
For each DNN, it varies n_components and registers the average correlation value at each layer.
The idea is to plot many curves (one for each n_components), where the average correlation per layer is plotted.
"""
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import shutil
import json
import os

import config
from utils.general_functions import load_pickle, dump_pickle

from load import main_parallel as main_load
from validation import main as main_val
from main import main as main_main

def convert_numpy_keys(obj):
    """Convert numpy integers to Python integers for JSON serialization"""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_numpy_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_keys(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# Total: 2*6*23 = 276 analyses
backbones = ["hubert", "wav2vec2", "wavlm"] # 2
components = np.concatenate(
    [np.arange(12, 24, 2), np.array([21])]
    ) # 6
layers = np.arange(1, 24) # 23 

correlations = {
    backbone: {
        n_components: {
                layer: None for layer in layers
            } for n_components in components
    } for backbone in backbones
}
correlations_std = {
    backbone: {
        n_components: {
                layer: None for layer in layers
            } for n_components in components
    } for backbone in backbones
}
save_path = Path("output/mtrf-ridge/analysis/DNN_component_analysis")
try:
    data = load_pickle(path=save_path / "checkpoint_DNN_component_correlations.pkl")
    correlations = data["correlations"]
    correlations_std = data["correlations_std"]
except FileNotFoundError:
    pass


for backbone in backbones:
    for n_components in components:  
        for layer in layers:
            stimuli = f'{n_components}DNNs{layer}-{backbone}'

            if correlations[backbone][n_components][layer] is not None:
                print(f"Skipping already computed {stimuli}")
                continue
            
            print(
                f'\n\n\n\tProcessing {n_components} components, layer {layer}, backbone {backbone}\n',
                f'\n\tStimuli:\t{stimuli}\n'
            )
            # Run the validation script with arguments for backbone and n_components
            _ = main_load(
                situations=['External'],
                bands=['Broad'],
                stimuli=[stimuli],
                save_results=True,
                number_of_workers=9
            )

            validation_path = Path(rf'output\mtrf-ridge\External\validation\stims_Standarize_EEG_Standarize\tmin-0.2_tmax0.6\Broad\{stimuli}')
            if not validation_path.exists():
                alphas = main_val(
                    situations=['External'],
                    stimuli=[stimuli],
                    bands=['Broad'],
                    save_results=True,
                    no_figures=True
                )['External']['Broad'][stimuli]
            else:
                alphas = load_pickle(path=validation_path / 'corr_limit_0.01.pkl')
                print(f"Validation found for {stimuli}, loading from disk.")

            alphas_total = []
            for session in config.sessions:
                for subject in [1, 2]:
                    alphas_total.append(alphas[session][subject])
            alphas_total = np.array(alphas_total)
            set_alpha = 10**(np.median(np.log10(alphas_total)))
            
            main_results = main_main(
                situations=['External'],
                stimuli=[stimuli],
                bands=['Broad'],
                save_results=False,
                set_alpha=set_alpha,
                no_figures=True
            )['External']['Broad'][stimuli]
            correlations[backbone][n_components][layer] = main_results['average_correlation_subjects'].mean(axis=1)
            correlations_std[backbone][n_components][layer] = main_results['average_correlation_subjects'].std(axis=1)/np.sqrt(128)

            # Save checkpoint
            save_path.mkdir(parents=True, exist_ok=True)
            dump_pickle(
                path=save_path / "checkpoint_DNN_component_correlations.pkl",
                obj={
                    "correlations": correlations,
                    "correlations_std": correlations_std
                },
                rewrite=True,
                verbose=True
            )
            # Save json to legible format
            with open(save_path / "checkpoint_DNN_component_correlations.json", 'w') as f:
                json_data = {
                    "correlations": convert_numpy_keys(correlations),
                    "correlations_std": convert_numpy_keys(correlations_std)
                }
                json.dump(json_data, f, indent=4)
            # Remove saved data to save space
            try:
                dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{stimuli}')
                shutil.rmtree(dir_to_remove, ignore_errors=True)
            except Exception as e:
                raise(f"Could not remove directory {dir_to_remove}: {e}")

# # Make DNNs plots
# fig, axes = plt.subplots(
#     nrows=1, ncols=3, 
#     figsize=(18, 6), 
#     sharey=True, 
#     # tight_layout=True
# )
# fig.suptitle("DNN Layer Correlation Analysis: Hubert vs Wav2Vec2 vs WavLM", fontsize=16)
# for idx, backbone in enumerate(['hubert', 'wav2vec2', 'wavlm']):
#     ax = axes[idx]
#     for n_components in components:
#         means = [correlations[backbone][n_components][layer].mean() for layer in layers]
#         stds = [correlations_std[backbone][n_components][layer].mean() for layer in layers]
#         ax.plot(layers, means, label=f'{n_components} components')
#         ax.fill_between(layers, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
#     ax.set_xticks(layers)
#     ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
#     ax.set_xlabel("DNN Layer")
#     ax.set_title(f"{backbone.capitalize()}")
# axes[0].set_ylabel("Inter-Subject Correlation")
# axes[1].legend(title="Number of components", bbox_to_anchor=(1.05, 1), loc='upper left')
# fig.tight_layout(rect=[0, 0, 1, 0.97])
# fig_save_path = Path("figures/analysis/dnn_layer_correlation")
# fig_save_path.mkdir(parents=True, exist_ok=True)
# fig.savefig(
#     fig_save_path / "hubert_wav2vec2_all_components_layer_correlation.png"
# )

# Make DNNs plots
fig, ax = plt.subplots(
    nrows=1, ncols=1, 
    figsize=(7, 5), 
    tight_layout=True
)
fig.suptitle(
    "DNN Layer Correlation Analysis", 
    fontsize=16
)
for idx, backbone in enumerate(['hubert', 'wav2vec2', 'wavlm']):
    means = np.array([correlations[backbone][21][layer].mean() for layer in layers]) # 23, 18
    stds = np.array([correlations_std[backbone][21][layer].mean() for layer in layers]) # 23, 18
    # make boxplot per layer hue by backbone
    ax.plot(layers, means, label=f'{backbone.capitalize()}')
    ax.fill_between(layers, np.array(means)-np.array(stds), np.array(means)+np.array(stds), alpha=0.2)
    ax.set_xticks(layers)
    ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
    ax.set_xlabel("Layer")
    # ax.set_title(f"{backbone.capitalize()}")
ax.set_ylabel("Average Correlation")
ax.legend(title="Model", loc='lower left', framealpha=0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig_save_path = Path("figures/analysis/dnn_layer_correlation")
fig_save_path.mkdir(parents=True, exist_ok=True)
fig.savefig(
    fig_save_path / "hubert_wavlm_wav2vec2_all_components_layer_correlation.png",
    dpi=600,
    transparent=True
)

