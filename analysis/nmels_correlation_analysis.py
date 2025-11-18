"""
This script performs an analysis of the effect of varying the number of Mel bins 
in spectrogram representations
"""
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json

from utils.general_functions import (
    load_pickle, dump_pickle, convert_numpy_keys
)
from load import main_parallel as main_load
from validation import main as main_validation
from main import main as main_main

FIG_SAVEPATH = Path("figures/analysis/spectrogram_nmels_correlation")
SAVE_PATH = Path("output/mtrf-ridge/analysis/nmels_analysis")
FIG_SAVEPATH.mkdir(parents=True, exist_ok=True)
SAVE_PATH.mkdir(parents=True, exist_ok=True)
N_MELS = np.arange(12, 34) 

correlations = {}
try:
    correlations = load_pickle(path=SAVE_PATH / "checkpoint_correlations.pkl")
except FileNotFoundError:
    pass

for n in tqdm(N_MELS, total=len(N_MELS), desc="N-Mels Correlation Analysis"):
    stimulus = f'Spectrogram-{n}'
    if correlations.get(n) is not None:
        print(f"Skipping already computed {stimulus}")
        continue
    print(
        f'\n\n\n\tProcessing {n}-mel Spectrogram\n',
    )
    
    load_results = main_load(
        situations=['External'],
        stimuli=[stimulus],
        bands=['Broad'],
        save_results=False
    )
    validation_results = main_validation(
        load_results=load_results,
        save_results=False
    )
    main_results = main_main(
        load_results=load_results,
        validation_results=validation_results,
        same_validation_subjects=False,
        save_results=False
    )['External']['Broad'][stimulus]
    
    correlations[n] = main_results['average_correlation_subjects']

    # Save checkpoint
    SAVE_PATH.mkdir(parents=True, exist_ok=True)
    dump_pickle(
        path=SAVE_PATH / "checkpoint_correlations.pkl",
        obj=correlations,
        rewrite=True,
        verbose=True
    )
    # Save json to legible format
    with open(SAVE_PATH / "checkpoint_correlations.json", 'w') as f:
        json_data = {
            "correlations": convert_numpy_keys(correlations),
        }
        json.dump(json_data, f, indent=4)

# Make Spectrogram plot
fig, ax = plt.subplots(
    nrows=1, ncols=1, 
    figsize=(10, 6), 
    # tight_layout=True
)
fig.suptitle("Spectrogram n-mels Correlation Analysis")
mean_correlations = [c.mean() for c in correlations.values()]
mean_correlations_std = [c.mean(axis=1).std(ddof=1)/np.sqrt(c.shape[0]) for c in correlations.values()]
ax.plot(N_MELS, mean_correlations)
ax.fill_between(
    N_MELS, 
    np.array(mean_correlations)-np.array(mean_correlations_std), 
    np.array(mean_correlations)+np.array(mean_correlations_std), 
    alpha=0.2
)
ax.set_xticks(N_MELS)
ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
ax.set_xlabel("N-Mels")
ax.set_ylabel("Inter-Subject Correlation")
ax.legend(title="Number of mel bins", bbox_to_anchor=(1.05, 1), loc='upper left')
fig.tight_layout(rect=[0, 0, 1, 0.97])  

fig.savefig(
    FIG_SAVEPATH / "spectrogram_nmels_correlation.png",
    transparent=True, 
    dpi=500
)

print("Figures saved to:", FIG_SAVEPATH.resolve())