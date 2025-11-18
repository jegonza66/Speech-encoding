"""
This script is designed to find which attributes correlate best with the different DNNs layers
"""
from matplotlib_venn import venn3, venn2
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import scienceplots
import numpy as np
import shutil
import json
import os

from utils.general_functions import (
    load_pickle, dump_pickle, convert_numpy_keys
)
from load import main_parallel as main_load
from validation import main as main_val
from main import main as main_main
from utils.processing import (
    calculate_partitions_2, calculate_partitions_3, correct_pearson_square
)
plt.style.use(['science'])
rc('text', usetex=True)

SAVE_PATH = Path("output/mtrf-ridge/analysis/DNN_similarity")
SAVE_FIG_PATH = Path("figures/analysis/DNN_similarity")
SAVE_FIG_PATH.mkdir(parents=True, exist_ok=True)
BACKBONES = ["wav2vec2", "wavlm", "hubert"]
LAYERS = list(np.arange(24).astype(int))
NUMBER_OF_DNN_COMPONENTS = 32

total_correlations = {}
for backbone in BACKBONES:
    stimuli = sorted(
        ['Spectrogram-21', 'Phonemes-Discrete'] + [f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}' for layer in LAYERS]
    )
    
    double_combinations = [
        '_'.join(sorted(['Spectrogram-21', 'Phonemes-Discrete'])),
    ]
    double_combinations += [
        '_'.join(sorted([f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}', 'Spectrogram-21'])) 
        for layer in LAYERS
    ] 
    double_combinations += [
        '_'.join(sorted([f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}', 'Phonemes-Discrete'])) 
        for layer in LAYERS
    ] 
    triple_combinations = [
        '_'.join(sorted([f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}', 'Spectrogram-21', 'Phonemes-Discrete'])) for layer in LAYERS
    ] 
    
    # 2+23+1+23+23+23 -> 95 total entries :S 
    correlations = {
        stimulus: None for stimulus in stimuli + double_combinations + triple_combinations
    } 

    # Sole correlations
    save_path_dnns_only = Path("output/mtrf-ridge/analysis/DNN_component_analysis/checkpoint_DNN_component_correlations.pkl")
    save_path_spectro_only = Path("output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/Spectrogram-21.pkl")
    save_path_phon_only = Path("output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/Phonemes-Discrete.pkl")
    correlations['Phonemes-Discrete'] = load_pickle(path=save_path_phon_only)['average_correlation_subjects'].mean()
    correlations['Spectrogram-21'] = load_pickle(path=save_path_spectro_only)['average_correlation_subjects'].mean()
    data_dnns_only = load_pickle(path=save_path_dnns_only)
    for layer in LAYERS:
        correlations[f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}'] = data_dnns_only["correlations"][backbone][NUMBER_OF_DNN_COMPONENTS][layer].mean()
    # 95 - 2 - 23 -> 70 entries :S

    # Save and compute double and triple combinations
    try:
        checkpoint_path = SAVE_PATH / f"checkpoint_{NUMBER_OF_DNN_COMPONENTS}_DNN_{backbone}_similarity_correlations.pkl"
        if not checkpoint_path.is_file():
            raise FileNotFoundError
        correlations = load_pickle(path=checkpoint_path)
    except Exception as e:
        SAVE_PATH.mkdir(parents=True, exist_ok=True)
        print(f"Error loading correlations: {e}\n")

    # Check if there are keys missing, and add them empty
    for stimulus in stimuli + double_combinations + triple_combinations:
        if stimulus not in correlations:
            correlations[stimulus] = None

    number_of_nans, l = sum([1 for v in correlations.values() if v is None]), 0

    for r, combination in enumerate(stimuli + double_combinations + triple_combinations):
        # Skip already computed combinations
        if correlations[combination] is not None:
            print(f"Skipping already computed {combination}")
            continue
        else:
            l += 1
            print(
                f'\n\n\n\tProcessing combination {combination}\n',
                f'\n\tStimuli:\t{combination}\n',
                f'\n\tProgress:\t{l}/{number_of_nans}\n'
            )

        # Run the validation script with arguments for backbone and n_components
        _ = main_load(
            situations=['External'],
            bands=['Broad'],
            stimuli=[combination],
            save_results=True,
            number_of_workers=12
        )

        alphas = main_val(
            situations=['External'],
            stimuli=[combination],
            bands=['Broad'],
            save_results=True,
            no_figures=True,
            n_folds=10,
            recompute=False
        )['External']['Broad'][combination]
        
        # Main results with optimal alpha
        main_results = main_main(
            situations=['External'],
            stimuli=[combination],
            bands=['Broad'],
            save_results=False,
            set_alpha=None,
            same_validation_subjects=False,
            no_figures=True
        )['External']['Broad'][combination]
        
        correlations[combination] = main_results['average_correlation_subjects'].mean()
        
        # Save checkpoint
        dump_pickle(
            path=SAVE_PATH / f"checkpoint_{NUMBER_OF_DNN_COMPONENTS}_DNN_{backbone}_similarity_correlations.pkl",
            obj=correlations,
            rewrite=True,
            verbose=True
        )
        # Save json to legible format
        with open(SAVE_PATH / f"checkpoint_{NUMBER_OF_DNN_COMPONENTS}_DNN_{backbone}_similarity_correlations.json", 'w') as f:
            json.dump(convert_numpy_keys(correlations), f, indent=4)

        # Remove saved data to save space
        try:
            if (combination in stimuli) and ('DNNs' not in combination):
                pass
            else:
                dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{combination}')
                shutil.rmtree(dir_to_remove, ignore_errors=True)
        except Exception as e:
            raise(f"Could not remove directory {dir_to_remove}: {e}")
    total_correlations[backbone] = correlations

    # Get Venn diagrams for triple combinations
    venn3_fig_path = SAVE_FIG_PATH / 'venn3' 
    venn3_fig_path.mkdir(parents=True, exist_ok=True)

    for triple_combination in triple_combinations:
        st1, st2, st3 = triple_combination.split('_')
        double_comb1 = '_'.join(sorted([st1, st2]))
        double_comb2 = '_'.join(sorted([st1, st3]))
        double_comb3 = '_'.join(sorted([st2, st3]))
        corrected_pearson = correct_pearson_square(
            values=[
                correlations[st1]**2, #A 
                correlations[st2]**2, #B
                correlations[st3]**2, #C
                correlations[double_comb1]**2, #AB_Union
                correlations[double_comb2]**2, #AC_Union
                correlations[double_comb3]**2, #BC_Union
                correlations[triple_combination]**2 #ABC_Union
            ]
        )
        areas = calculate_partitions_3(
            **corrected_pearson
        )

        # areas = measured_areas
        total_area = sum(areas)

        # Normalize to give percentage of variance explained by full model
        areas = (np.array(areas)*100/total_area).round(2)

        # Create figure and title
        plt.figure(layout='tight')
        layer = int(triple_combination.split('DNNs')[-1].split('-')[0])
        # plt.title(f'Spectrogram-Phonemes-{backbone}-layer{layer}')

        # Make plot
        layer = st1.split('DNNs')[-1].split('-')[0]
        label1 = st1.split('-')[-1].capitalize() + f' Layer {layer}' if 'DNNs' in st1 else st1
        label2 = st2.split('-')[0].capitalize() if 'Phonemes' in st2 else st2
        label3 = st3.split('-')[0].capitalize() if 'Spectrogram' in st3 else st3
        venn = venn3(
            subsets=areas, # left area diagram, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
            set_labels=(label1, label2, label3), 
            set_colors=('purple', 'orange', '#87CEEB'), 
            alpha=0.45
            )
        for label in venn.subset_labels:
            if label:  
                label.set_text(label.get_text() + r' \%')
                # label.set_fontsize(15)
        # for label in venn.set_labels:
        #     if label:  # Verificar que la etiqueta no sea None
        #         label.set_fontsize(18)        
        plt.savefig(venn3_fig_path / f'venn3_{backbone}_layer{layer}.png')
        plt.close()

    # Same for double combinations
    venn2_fig_path = SAVE_FIG_PATH / 'venn2' 
    venn2_fig_path.mkdir(parents=True, exist_ok=True)
    for double_combination in double_combinations:
        st1, st2 = double_combination.split('_')
        corrected_pearson = correct_pearson_square(
            values=[
                correlations[st1]**2, #A 
                correlations[st2]**2, #B
                correlations[double_combination]**2 #ABC_Union
            ]
        )
        areas = calculate_partitions_2(
            **corrected_pearson
        )
        total_area = sum(areas)
        
        # Normalize to give percentage of variance explained by full model
        areas = (np.array(areas)*100/total_area).round(2)

        # Create figure and title
        
        plt.ioff()
        plt.figure(layout='tight')
        # plt.title(f'{backbone} - {double_combination}')

        # Make plot
        label1 = st1.split('-')[-1].capitalize() if 'DNNs' in st1 else st1
        if 'Phonemes' in st2:
            label2 = st2.split('-')[0].capitalize() if 'Phonemes' in st2 else st2
        elif 'Spectrogram' in st2:
            label2 = st2.split('-')[0].capitalize() if 'Spectrogram' in st2 else st2

        venn = venn2(
            subsets=areas, # left area diagram, right area diagram, shared area <--> (10, 01, 11).
            set_labels=(st1, st2), 
            set_colors=('C0', 'C1'), 
            alpha=0.45
            )
        for label in venn.subset_labels:
            if label:  
                label.set_text(label.get_text() + r' \%')

        plt.savefig(venn2_fig_path / f'venn2_{backbone}_{double_combination}.png')
        plt.close()
# ============================================================================
# Plot shared variance of spectrogram and phonemes as a function of DNN layer 
# separately for each backbone
fig, axes = plt.subplots(
    nrows=2, 
    ncols=3, 
    figsize=(12, 5), 
    layout='tight', 
    sharex=True, 
    sharey='row'
)
for i, backbone in enumerate(BACKBONES):
    stimuli = ['Spectrogram-21', 'Phonemes-Discrete'] + [f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}' for layer in LAYERS]
    stimuli = sorted(stimuli)
    combinations = [
        '_'.join(sorted([f'{NUMBER_OF_DNN_COMPONENTS}DNNs{layer}-{backbone}', 'Phonemes-Discrete','Spectrogram-21'])) 
        for layer in sorted(LAYERS)
    ] 
    shared_variances_ph = []
    shared_variances_sp = []
    total_shared_correlation = []

    for l, combination in enumerate(combinations):
        dnn, ph, sp = combination.split('_')
        dnn_ph = dnn + '_' + ph
        dnn_sp = dnn + '_' + sp
        ph_sp = ph + '_' + sp

        variance_dnn = total_correlations[backbone][dnn]**2
        variance_ph = total_correlations[backbone][ph]**2
        variance_sp = total_correlations[backbone][sp]**2
        variance_shared = total_correlations[backbone][combination]**2
        variance_shared_onlywith_ph = total_correlations[backbone][dnn_sp]**2 
        variance_shared_onlywith_ph += total_correlations[backbone][ph_sp]**2 
        variance_shared_onlywith_ph -= variance_sp + variance_shared
        variance_shared_onlywith_sp = total_correlations[backbone][dnn_ph]**2 
        variance_shared_onlywith_sp += total_correlations[backbone][ph_sp]**2 
        variance_shared_onlywith_sp -= variance_ph + variance_shared

        # Intersection variance = variance_1 + variance_2 - variance_12
        shared_variances_sp.append(100*variance_shared_onlywith_sp/variance_shared)
        shared_variances_ph.append(100*variance_shared_onlywith_ph/variance_shared)
        total_shared_correlation.append(total_correlations[backbone][combination])
    label = r'$(DNNs \cap Ph) \setminus (Sp)$' if backbone=='wav2vec2' else ''
    axes[0,i].plot(
        sorted(LAYERS),
        np.array(shared_variances_ph),
        marker='o',
        label=label,
        color='orange'
    )
    label = r'$(DNNs \cap Sp) \setminus (Ph)$' if backbone=='wav2vec2' else ''
    axes[0,i].plot(
        sorted(LAYERS),
        np.array(shared_variances_sp),
        marker='o',
        label=label,
        color='#87CEEB'  # celeste / sky blue
    )
    axes[0,i].set_title(f'{backbone.capitalize()}')
    axes[0,i].grid()
    axes[0,i].set_xticks(sorted(LAYERS)[::2])

    if i==0:
        axes[0, i].set_ylabel(r'Shared Variance (\%)')
        # axes[0,i].legend(loc='best')

    # Second row: total shared correlation
    label = 'Total Shared Correlation' if backbone=='wav2vec2' else ''
    axes[1,i].plot(
        sorted(LAYERS), np.array(total_shared_correlation), 
        color='black',
        label=label,
        marker='o',
    )
    axes[1,i].grid()
    axes[1,i].set_xlabel('DNN Layer')
    if i==0:
        axes[1, i].set_ylabel(r'Average Correlation')
        # axes[1,i].legend(loc='best')
    # axes[1,i].set_yticks([0.45, 0.47, 0.485])
    # axes[1,i].set_ylim(0.44, 0.49)

legend = fig.legend(
    loc='upper center', 
    bbox_to_anchor=(0.5, 1.05), 
    ncol=3,
    frameon=False,
    fontsize=12
)
fig.savefig(
    SAVE_FIG_PATH / f'shared_over_layers.png',
    dpi=300,
    transparent=True
)

print(f"Figures saved in {SAVE_FIG_PATH.resolve()}")