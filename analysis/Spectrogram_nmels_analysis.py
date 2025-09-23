"""
This script performs an analysis of the effect of varying the number of mel bins in spectrogram representations
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

# Spectrogram-12, -13, ..., -26
n_mels = np.arange(12, 22) 
correlations = []
correlations_std = []

save_path = Path("output/mtrf-ridge/analysis/Spectrogram_nmels_analysis")
try:
    data = load_pickle(path=save_path / "checkpoint_correlations.pkl")
    correlations = data["correlations"]
    correlations_std = data["correlations_std"]
except FileNotFoundError:
    pass

for j, n in enumerate(n_mels):
    stimuli = f'Spectrogram-{n}'
    if len(correlations) > j:
        print(f"Skipping already computed {stimuli}")
        continue
    print(
        f'\n\n\n\tProcessing {n}-mel Spectrogram\n',
    )
    # Run the validation script with arguments for n-mels spectrogram
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
    
    correlations.append(main_results['average_correlation_subjects'].mean(axis=1))
    correlations_std.append(main_results['average_correlation_subjects'].std(axis=1)/np.sqrt(128))

    # Save checkpoint
    save_path.mkdir(parents=True, exist_ok=True)
    dump_pickle(
        path=save_path / "checkpoint_correlations.pkl",
        obj={
            "correlations": correlations,
            "correlations_std": correlations_std
        },
        rewrite=True,
        verbose=True
    )
    # Save json to legible format
    with open(save_path / "checkpoint_correlations.json", 'w') as f:
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

# Make Spectrogram plot
fig, ax = plt.subplots(
    nrows=1, ncols=1, 
    figsize=(10, 6), 
    # tight_layout=True
)
fig.suptitle("Spectrogram n-mels Correlation Analysis")
mean_correlations = [c.mean() for c in correlations]
mean_correlations_std = [c.mean() for c in correlations_std]
ax.plot(n_mels, mean_correlations)
ax.fill_between(
    n_mels, 
    np.array(mean_correlations)-np.array(mean_correlations_std), 
    np.array(mean_correlations)+np.array(mean_correlations_std), 
    alpha=0.2
)
ax.set_xticks(n_mels)
ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
ax.set_xlabel("N-Mels")
ax.set_ylabel("Inter-Subject Correlation")
ax.legend(title="Number of mel bins", bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout(rect=[0, 0, 1, 0.97])  
fig_path = Path("figures/analysis/spectrogram_nmels_correlation")
fig_path.mkdir(parents=True, exist_ok=True)
fig.savefig(
    fig_path / "spectrogram_nmels_correlation.png"
)