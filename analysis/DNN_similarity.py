"""
This script is designed to find which attributes correlate best with the different DNNs layers
"""
from matplotlib_venn import venn3  
from pathlib import Path

from utils.general_functions import load_pickle
import config

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

layers = [1, 4, 8, 12, 16, 20, 23] # np.arange(1, 24)
backbone = "hubert"

stimuli = ['Spectrogram-21', 'Phonemes-Frequency'] + [f'21DNNs{layer}-{backbone}' for layer in layers]
stimuli = sorted(stimuli)
double_combinations = [
    '_'.join(sorted(['Spectrogram-21', 'Phonemes-Frequency'])),
]
double_combinations += [
    '_'.join(sorted([f'21DNNs{layer}-{backbone}', 'Spectrogram-21'])) 
    for layer in layers
] 
double_combinations += [
    '_'.join(sorted([f'21DNNs{layer}-{backbone}', 'Phonemes-Frequency'])) for layer in layers
] 
triple_combinations = [
    '_'.join(sorted([f'21DNNs{layer}-{backbone}', 'Spectrogram-21', 'Phonemes-Frequency'])) for layer in layers
] 

correlations = {
    stimulus: None for stimulus in stimuli + double_combinations + triple_combinations
} # 95 total entries :S -> TODO We can compute just layers 1, 4, 8, 12, 16, 20, 23 -> 35 entries


# Sole correlations
save_path_dnns_only = Path("output/mtrf-ridge/analysis/DNN_component_analysis/checkpoint_DNN_component_correlations.pkl")
save_path_spectro_only = Path("output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/Spectrogram-21.pkl")
save_path_phon_only = Path("output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/Phonemes-Frequency.pkl")
correlations['Phonemes-Frequency'] = load_pickle(path=save_path_phon_only)['average_correlation_subjects'].mean()
correlations['Spectrogram-21'] = load_pickle(path=save_path_spectro_only)['average_correlation_subjects'].mean()

data_dnns_only = load_pickle(path=save_path_dnns_only)
for layer in layers:
    correlations[f'21DNNs{layer}-{backbone}'] = data_dnns_only["correlations"][backbone][21][layer].mean()


# Save and compute double and triple combinations
save_path = Path("output/mtrf-ridge/analysis/DNN_similarity")
try:
    correlations = load_pickle(save_path / "checkpoint_DNN_similarity_correlations.pkl")
except FileNotFoundError:
    pass

for combination in stimuli + double_combinations + triple_combinations:
    if correlations[combination] is not None:
        print(f"Skipping already computed {combination}")
        continue
    print(
        f'\n\n\n\tProcessing combination {combination}\n',
        f'\n\tStimuli:\t{combination}\n'
    )
    # Run the validation script with arguments for backbone and n_components
    _ = main_load(
        situations=['External'],
        bands=['Broad'],
        stimuli=[combination],
        save_results=True,
        number_of_workers=7
    )
    validation_path = Path(rf'output\mtrf-ridge\External\validation\stims_Standarize_EEG_Standarize\tmin-0.2_tmax0.6\Broad\{stimuli}')
    if validation_path.exists():
        alphas = load_pickle(path=validation_path / 'corr_limit_0.01.pkl')
        print(f"Validation found for {stimuli}, loading from disk.")
    else:
        alphas = main_val(
            situations=['External'],
            stimuli=[stimuli],
            bands=['Broad'],
            save_results=True,
            no_figures=True
        )['External']['Broad'][stimuli]
    
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
    
    # Save checkpoint
    save_path.mkdir(parents=True, exist_ok=True)
    dump_pickle(
        path=save_path / "checkpoint_DNN_component_correlations.pkl",
        obj=correlations,
        rewrite=True,
        verbose=True
    )
    # Save json to legible format
    with open(save_path / "checkpoint_DNN_component_correlations.json", 'w') as f:
        json.dump(convert_numpy_keys(correlations), f, indent=4)
    # Remove saved data to save space
    try:
        if (combination in stimuli) or ('DNNs' not in combination):
            pass
        else:
            dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{combination}')
            shutil.rmtree(dir_to_remove, ignore_errors=True)
    except Exception as e:
        raise(f"Could not remove directory {dir_to_remove}: {e}")

# Get Venn diagrams for triple combinations
savefig_path = Path("figures/analysis/DNN_similarity")
for triple_combination in triple_combinations:
    st1, st2, st3 = triple_combination.split('_')
    double_comb1 = '_'.join(sorted([st1, st2]))
    double_comb2 = '_'.join(sorted([st1, st3]))
    double_comb3 = '_'.join(sorted([st2, st3]))

    # Simple variances
    variance_1 = correlations[st1]**2
    variance_2 = correlations[st2]**2
    variance_3 = correlations[st3]**2
    variance_12 = correlations[double_comb1]**2
    variance_13 = correlations[double_comb2]**2
    variance_23 = correlations[double_comb3]**2
    variance_123 = correlations[triple_combination]**2

    # Shared without each stimulus
    variance_shared_with_1 = variance_123 - variance_23 #100
    variance_shared_with_2 = variance_123 - variance_13 #010
    variance_shared_with_3 = variance_123 - variance_12 #001

    # Explained by subshared, but not by all shared model
    variance_shared_with_12 = variance_13 + variance_23 - variance_3 - variance_123 #110
    variance_shared_with_13 = variance_12 + variance_23 - variance_2 - variance_123 #101
    variance_shared_with_23 = variance_12 + variance_13 - variance_1 - variance_123 #011

    # Explained by one, two, three and full shared model but not by subshared models
    variance_int_complement_submodels = variance_123 + variance_1 + variance_2 + variance_3 - variance_12 - variance_13 - variance_23 #111

    # Get areas 
    areas = [ # the order should be(100, 010, 110, 001, 101, 011, 111)
        variance_shared_with_1, 
        variance_shared_with_2, 
        variance_shared_with_12, 
        variance_shared_with_3,
        variance_shared_with_13, 
        variance_shared_with_23,
        variance_int_complement_submodels
        ] 
    total_area = sum(areas)
    areas = np.array([
        0 if area<0 else area.round(3) 
        for area in areas
    ]) # note that the sum gives shared model variance_123
    
    # Normalize to give percentage of variance explained by full model
    areas = (areas*100/total_area).round(2)

    # Create figure and title
    
    plt.ioff()
    plt.figure(layout='tight')
    layer = int(triple_combination.split('DNNs')[-1].split('-')[0])
    plt.title(f'Spectrogram-Phonemes-DNNs-layer{layer}')

    # Make plot
    venn3(
        subsets=areas, # left area diagran, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
        set_labels=([0], [1], [2]), 
        set_colors=('C0', 'C1', 'purple'), 
        alpha=0.45
        )

    plt.savefig(savefig_path / f'venn3_{layer}.png')
    plt.close()