"""
This script generates a correlation matrix indicating the correlation value of different attributes
"""
from matplotlib.collections import PathCollection
from matplotlib_venn import venn3  
import matplotlib.pyplot as plt
import matplotlib.text as mtext
from pathlib import Path
import seaborn as sns
import pandas as pd
import json, shutil
import numpy as np
import mne
import os

from utils.plot import define_ticks, clustering_by_correlation
from utils.general_functions import dump_pickle, load_pickle
from load import main_parallel as main_load
from validation import main as main_val
from main import main as main_main
import config

stimuli = [
    'Envelope',
    'Pitch-Log-Raw',
    'Spectrogram-21',
    'Phonemes-Discrete',
    'Phonological',
]
stimuli_names = [
    'Envelope',
    'Pitch-Log',
    'Spectrogram',
    'Phonemes',
    'Phonological\nfeatures'
]
bands = [
    'Broad',
    'Delta',
    'Theta',
    'Alpha',
    'Beta'
]

correlation_path = lambda stimulus, band: Path(rf"output\mtrf-ridge\External-External\correlations\same_alpha\tmin-0.2_tmax0.6\{band}\{stimulus}.pkl")
correlations = {}
for band in bands:
    correlations[band] = {}
    for k, stimulus in enumerate(stimuli):
        correlation = load_pickle(
            path=correlation_path(stimulus, band)
        )['average_correlation_subjects'] # shape (n_subjects, n_channels)
        correlations[band][stimulus] = correlation
    
# Visualization across bands and stimuli: take average across channels
data = []
for band_idx, band in enumerate(bands):
    for stim_idx, stimulus in enumerate(stimuli):
        corr = correlations[band][stimulus]  # shape (n_subjects, n_channels)
        subj_means = corr.mean(axis=1) # mean across channels
        subj_sems = corr.std(axis=1, ddof=1) / np.sqrt(corr.shape[1]) # SEM across channels
        for subj, (mean, sem) in enumerate(zip(subj_means, subj_sems)):
            data.append({
                "Band": band,
                "Stimulus": stimuli_names[stim_idx],
                "Subject": subj,
                "MeanCorrelation": mean,
                "SEM": sem
            })

df = pd.DataFrame(data)
fig = plt.figure(
    figsize=(12, 6),
    tight_layout=True
)
ax = plt.gca()
ax.grid(visible=True, which='major', linestyle='--', axis='y', linewidth=0.5)
sns.boxplot(
    data=df, 
    x="Stimulus", 
    y="MeanCorrelation", 
    hue="Band", 
    showfliers=False, 
    whis=[5, 95],
    ax=ax,
    saturation=0.75,
    boxprops=dict(alpha=0.5),
    palette=['C0', 'C1', 'C2', 'C3', 'purple']
)
sns.stripplot(
    data=df, 
    x="Stimulus", 
    y="MeanCorrelation", 
    hue="Band", 
    dodge=True, 
    jitter=True, 
    marker='o',
    alpha=0.5, 
    linewidth=0.5,
    edgecolor='black',
    legend=False,
    ax=ax,
    palette=['C0', 'C1', 'C2', 'C3', 'purple']
)
ax.set_ylabel("Mean Correlation (avg. across channels)", fontsize=15)
ticks = ax.get_xticks()
ax.set_xticks(ticks=ticks, labels=stimuli_names, rotation=30, ha='right', fontsize=15)
ax.set_xlabel(None)
ax.legend(title='Band', loc=(.13, .65), fontsize=12, title_fontsize=12, framealpha=0)
path_fig = Path(config.figures_dir) / 'analysis' / 'model_visualization_matrix_corr' / 'model_visualization_matrix_corr.png'
path_fig.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(
    path_fig,
    transparent=True, 
    dpi=500
)
# fig.show()


## TRF

fig, axes_ = plt.subplots(nrows=2, ncols=2, figsize=(11, 10), layout='tight', sharex=True)
band = 'Broad'
fig.suptitle(f'TRFs ({band} band)', fontsize=20)
for stim in ['Phonemes-Discrete', 'Phonological']:
    mean_average_weights_subjects = load_pickle(
        rf'output\mtrf-ridge\External-External\weights\stims_Standarize_EEG_Standarize\same_alpha\tmin-0.2_tmax0.6\{band}\{stim}\total_weights_per_subject.pkl'
    )['average_weights_subjects'].mean(axis=0)
    axes = axes_[:, 0] if stim == 'Phonemes-Discrete' else axes_[:, 1]

    # Create evoked response as graph of weights averaged across all feats and subjects 
    weights = mean_average_weights_subjects.mean(axis=1)
    evoked = mne.EvokedArray(data=weights, info=config.info_mne)

    # Relabel time 0
    evoked.shift_time(config.times[0], relative=True)

    # Plot
    evoked.plot(
        scalings={'eeg':1}, 
        zorder='std', 
        time_unit='ms',
        show=False, 
        spatial_colors=True, 
        units='mTRF (a.u.)',
        axes=axes[0],
        gfp=False
        # sphere=(0.95, 0.8, 0, 0.5)
    )

    # Add mean of all channels
    axes[0].plot(
        config.times*1e3, #ms
        evoked._data.mean(0), 
        'k', 
        label='Mean', 
        zorder=130, 
        linewidth=2
        )
    if stim=='Phonemes-Discrete':
        axes[0].set_title('Phonemes', fontsize=18)
    else:
        axes[0].set_title('Phonological\nfeatures', fontsize=18)
    # Eliminar la etiqueta "Nave"
    for txt in fig.findobj(mtext.Text):
        if "ave" in txt.get_text():
                txt.remove()

    # Graph properties
    axes[0].grid(visible=True)
    axes[0].set_xlabel(xlabel='')
    axes[0].set_ylabel(ylabel='mTRF (a.u.)', fontsize=18)
    axes[0].legend(framealpha=0)

    for ax in fig.axes:
        # Verificar si el eje contiene un objeto de tipo "PathCollection" (los puntos de los canales)
        for artist in ax.get_children():
            if isinstance(artist, PathCollection):
                ax.remove()  # Eliminar el eje que contiene el esquema de la cabeza original
                break
            
    # Obtener las posiciones de los sensores en 2D
    montage = evoked.info.get_montage()
    pos = montage.get_positions()['ch_pos']  # Diccionario con las posiciones de los canales

    # Crear un eje adicional para la cabecita sin sensores
    ax_head_outline = fig.add_axes([.265, 0.56, 0.15, 0.15])  # [x, y, width, height]

    # Graficar solo el contorno de la cabeza (sin sensores)
    ax_head_outline.patch.set_alpha(0.1) 
    mne.viz.plot_topomap(
        np.zeros(len(evoked.ch_names)),  # Datos ficticios (todos ceros)
        evoked.info,
        axes=ax_head_outline,
        show=False,
        sensors=False,  # No graficar los sensores
        outlines='head',  # Graficar solo el contorno de la cabeza
        cmap='binary',
        contours=1
    )
    ax_head_outline.set_aspect('equal')  # Mantener la proporción de aspecto
    ax_head_outline.axis('off')  # Ocultar los ejes

    # Crear un eje adicional para graficar los sensores
    ax_head = fig.add_axes([.277, 0.562, 0.125, 0.125])  # [x, y, width, height]

    # Convertir las posiciones a un array 2D (x, y)
    colors = [line.get_color() for line in axes[0].get_lines()[:len(evoked.ch_names)]]
    pos_2d = np.array([pos[ch][:2] for ch in evoked.ch_names])  # Solo tomamos las coordenadas x e y
    ax_head.scatter(pos_2d[:, 0], pos_2d[:, 1], c=colors, s=18)  # s es el tamaño de los puntos
    ax_head.set_aspect('equal')  # Mantener la proporción de aspecto
    ax_head.axis('off')  # Ocultar los ejes

    # Now average across channels to make mesh
    feat_weights = mean_average_weights_subjects.mean(axis=0)

    # Perform clustering
    order, null_indexes = clustering_by_correlation(weights=feat_weights) 
    feat_weights = feat_weights[order]

    # Create colormesh figure
    number_of_ticks = feat_weights.shape[0]
    im = axes[1].pcolormesh(
            config.times * 1000, 
            np.arange(number_of_ticks), 
            feat_weights, 
            cmap='RdBu_r', 
            shading='auto',
            vmin=-np.abs(feat_weights).max(),
            vmax=np.abs(feat_weights).max()
            )
    axes[1].set_xlabel('Time (ms)', fontsize=18)
    axes[1].set_ylabel('Phonemes', fontsize=18)
    # axes[1].set_yticklabels(axes[1].get_yticklabels(), fontsize=18)
    # axes[1].tick_params(axis='y', labelsize=18)

    # labels = axes[1].get_yticklabels()
    # for label in labels:
    #     label.get_fontsize()
    #     label.set_fontsize(18)
    # axes[1].tick_params(axis='y', labelsize=18)
    # axes[1].tick_params(axis='x')
    # Set figure configuration
    if stim=='Phonemes-Discrete':
        define_ticks(axes=axes[1], number_of_ticks=number_of_ticks, ylabel='Phonemes', xlabel='Time (ms)', title=None, order=order, zeros_index=null_indexes)
    else:
        define_ticks(axes=axes[1], number_of_ticks=number_of_ticks, ylabel='Phonological', xlabel='Time (ms)', title=None, order=order, zeros_index=null_indexes)
    axes[1].set_ylabel('Phonemes' if stim=='Phonemes-Discrete' else 'Features', fontsize=18)
    axes[1].tick_params(axis='x', labelsize=18)
    axes[0].tick_params(axis='y', labelsize=18)
    axes[1].tick_params(axis='y', labelsize=18)

fig_save_path = Path(config.figures_dir) / 'analysis' / 'model_visualization_matrix_corr' / 'model_visualization_matrix_corr_trf.png'
fig_save_path.parent.mkdir(parents=True, exist_ok=True)

# fig.axes[3].set_position([0.9, 0.015, 0.2, 0.12])  # lower right
fig.savefig(
    fig_save_path,
    transparent=True, 
    dpi=500
)

#TOPO CORR
average_correlation_subjects_phonological = load_pickle(
    rf'output\mtrf-ridge\External-External\correlations\same_alpha\tmin-0.2_tmax0.6\Broad\Phonological.pkl'
)['average_correlation_subjects'] # shape (n_subjects, n_channels)
average_correlation_subjects_phonemes = load_pickle(
    rf'output\mtrf-ridge\External-External\correlations\same_alpha\tmin-0.2_tmax0.6\Broad\Phonemes-Discrete.pkl'
)['average_correlation_subjects'] # shape (n_subjects, n_channels)

# Create figure and title
fig, ax = plt.subplots(nrows=1, ncols=2, layout='tight', figsize=(8, 5))
# plt.suptitle(f'{stim} {coefficient_name} = ({mean_average_coefficient.mean():.3f}'+r'$\pm$'+f'{mean_average_coefficient.std():.3f})')
for average_correlation_subjects, stimulus, axis in zip(
    [average_correlation_subjects_phonemes, average_correlation_subjects_phonological],
    ['Phonemes-Discrete', 'Phonological'],
    ax
):
    vmin = average_correlation_subjects.mean(0).min()
    vmax = average_correlation_subjects.mean(0).max()
    
    # Make topomap
    im = mne.viz.plot_topomap(
        data=average_correlation_subjects.mean(axis=0),  # Mean across subjects
        pos=config.info_mne,
        cmap='OrRd',
        vlim=(vmin, vmax),
        show=False,
        sphere=0.07,
        axes=axis
    )
    if stimulus=='Phonemes-Discrete':
        ticks = [0.293,0.323,0.353,0.383,0.413,0.442]
    else:
        ticks = np.linspace(vmin, vmax, 6).round(3) if vmin != vmax else [vmin]
    plt.colorbar(
        im[0],
        ax=axis,
        shrink=0.85,
        label='Mean Correlation (avg. across subjects)',
        orientation='horizontal',
        boundaries=np.linspace(vmin, vmax, 100) if vmin != vmax else None,
        ticks=ticks
    )
    axis.set_title('Phonemes' if stimulus=='Phonemes-Discrete' else 'Phonological\nfeatures', fontsize=18)

fig.savefig(
    Path(config.figures_dir) / 'analysis' / 'model_visualization_matrix_corr' / 'model_visualization_matrix_corr_topo_corr.png',
    transparent=True,
    dpi=500
)




models = [
    'Spectrogram-21', 
    'Phonemes-Discrete', 
    'Phonological',
    'Spectrogram-21_Phonemes-Discrete',
    'Spectrogram-21_Phonological',
    'Phonemes-Discrete_Phonological',
    'Spectrogram-21_Phonemes-Discrete_Phonological'
]
models = ['_'.join(sorted(model.split('_'))) for model in models]
corr_path = lambda model: Path(f"output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/{model}.pkl")

correlations = {
    model: load_pickle(corr_path(model))['average_correlation_subjects'].mean()
    for model in models
}

# Get Venn diagrams for triple combinations
savefig_path = Path("figures/analysis/model_visualization_matrix_corr/venn3")
savefig_path.mkdir(parents=True, exist_ok=True)

triple_combination = models[-1]
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
# areas = np.array([
#     0 if area<0 else area.round(3) 
#     for area in areas
# ]) # note that the sum gives shared model variance_123

# Normalize to give percentage of variance explained by full model
areas = (np.array(areas)*100/total_area).round(2)

# Create figure and title

plt.ioff()
plt.figure(layout='tight')
# plt.title(f'Spectrogram ∪ Phonemes ∪ Phonological')

# Make plot
venn=venn3(
    subsets=areas, # left area diagram, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
    set_labels=(st1, st2, st3), 
    set_colors=('C0', 'C1', 'purple'), 
    alpha=0.45
    )
for label in venn.subset_labels:
    if label:  # Verificar que la etiqueta no sea None
        label.set_fontsize(18)
for label in venn.set_labels:
    if label:  # Verificar que la etiqueta no sea None
        label.set_fontsize(18)

plt.savefig(savefig_path / f'venn3_{triple_combination}.png', transparent=True, dpi=500)
plt.close()

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

# Save and compute double and triple combinations
save_path = Path("output/mtrf-ridge/analysis/DNN_similarity")
try:
    correlations_old = load_pickle(save_path / "checkpoint_DNN_similarity_correlations.pkl")
    # Update only missing entries
    correlations_old.update(correlations)
    correlations = correlations_old
except FileNotFoundError:
    pass

# Make the same for 3 sheared models (layers 1, 8, 18). Each layer comparing for WavLM, Hubert and wav2vec2
for layer in [1, 8, 18]:
    backbones = ["wavlm", "hubert", "wav2vec2"]

    stimuli = [f'21DNNs{layer}-{backbone}' for backbone in backbones]
    stimuli = sorted(stimuli)
    double_combinations = [
        '_'.join(sorted([f'21DNNs{layer}-wavlm', f'21DNNs{layer}-wav2vec2'])) 
    ] 
    double_combinations += [
        '_'.join(sorted([f'21DNNs{layer}-wavlm', f'21DNNs{layer}-hubert'])) 
    ] 
    double_combinations += [
        '_'.join(sorted([f'21DNNs{layer}-wav2vec2', f'21DNNs{layer}-hubert'])) 
    ] 
    triple_combinations = [
        '_'.join(sorted([f'21DNNs{layer}-wavlm', f'21DNNs{layer}-wav2vec2', f'21DNNs{layer}-hubert'])) 
    ] 
    for stimulus in stimuli + double_combinations + triple_combinations:
        if stimulus not in correlations:
            correlations[stimulus] = None

    # Sole correlations
    save_path_dnns_only = Path("output/mtrf-ridge/analysis/DNN_component_analysis/checkpoint_DNN_component_correlations.pkl")
    data_dnns_only = load_pickle(path=save_path_dnns_only)
    for backbone in backbones:
        correlations[f'21DNNs{layer}-{backbone}'] = data_dnns_only['correlations'][backbone][21][layer].mean()


    for r, combination in enumerate(stimuli + double_combinations + triple_combinations):
        if correlations[combination] is not None:
            print(f"Skipping already computed {combination}")
            continue
        print(
            f'\n\n\n\tProcessing combination {combination}\n',
            f'\n\tStimuli:\t{combination}\n',
            f'\n\tProgress:\t{r+1}/{len(stimuli + double_combinations + triple_combinations)}\n'
        )
        # Run the validation script with arguments for backbone and n_components
        _ = main_load(
            situations=['External'],
            bands=['Broad'],
            stimuli=[combination],
            save_results=True,
            number_of_workers=12
        )
        validation_path = Path(rf'output\mtrf-ridge\External\validation\stims_Standarize_EEG_Standarize\tmin-0.2_tmax0.6\Broad\{combination}')
        if validation_path.exists():
            alphas = load_pickle(path=validation_path / 'corr_limit_0.01.pkl')
            print(f"Validation found for {combination}, loading from disk.")
        else:
            alphas = main_val(
                situations=['External'],
                stimuli=[combination],
                bands=['Broad'],
                save_results=True,
                no_figures=True
            )['External']['Broad'][combination]

        # alphas_total = []
        # for session in config.sessions:
        #     for subject in [1, 2]:
        #         alphas_total.append(alphas[session][subject])
        # alphas_total = np.array(alphas_total)
        # set_alpha = 10**(np.median(np.log10(alphas_total)))
        
        main_results = main_main(
            situations=['External'],
            stimuli=[combination],
            bands=['Broad'],
            save_results=False,
            # set_alpha=set_alpha,
            same_validation_subjects=False,  # Changed to optimal alpha
            no_figures=True
        )['External']['Broad'][combination]
        correlations[combination] = main_results['average_correlation_subjects'].mean()
        
        # Save checkpoint
        save_path.mkdir(parents=True, exist_ok=True)
        dump_pickle(
            path=save_path / "checkpoint_DNN_similarity_correlations.pkl",
            obj=correlations,
            rewrite=True,
            verbose=True
        )
        # Save json to legible format
        with open(save_path / "checkpoint_DNN_similarity_correlations.json", 'w') as f:
            json.dump(convert_numpy_keys(correlations), f, indent=4)
        # Remove saved data to save space
        try:
            if (combination in stimuli) or ('DNNs' not in combination):
                pass
            else:
                # FIXME: remove each stimuli not combination
                dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{combination}')
                shutil.rmtree(dir_to_remove, ignore_errors=True)
        except Exception as e:
            raise(f"Could not remove directory {dir_to_remove}: {e}")

# Get Venn diagrams for triple combinations
savefig_path = Path("figures/analysis/model_visualization_matrix_corr/venn3")
savefig_path.mkdir(parents=True, exist_ok=True)
save_path = Path("output/mtrf-ridge/analysis/DNN_similarity")
correlations = load_pickle(save_path / "checkpoint_DNN_similarity_correlations.pkl")
triple_combinations = [
    '_'.join(sorted([f'21DNNs{layer}-wavlm', f'21DNNs{layer}-wav2vec2', f'21DNNs{layer}-hubert'])) 
    for layer in [1, 8, 18]
]
for triple_combination in triple_combinations:
     # Create figure and title

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
    # areas = np.array([
    #     0 if area<0 else area.round(3) 
    #     for area in areas
    # ]) # note that the sum gives shared model variance_123

    # Normalize to give percentage of variance explained by full model
    areas = (np.array(areas)*100/total_area).round(2)

    # Create figure and title

    plt.ioff()
    plt.figure(layout='tight')
    layer = int(triple_combination.split('DNNs')[-1].split('-')[0])
    plt.title(fr'Layer {layer}', fontsize=20)

    # Make plot
    label_st1 = f'Hubert'
    label_st2 = f'WavLM'
    label_st3 = f'Wav2Vec2'
    venn = venn3(
        subsets=areas, # left area diagram, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
        set_labels=(label_st1, label_st2, label_st3), 
        set_colors=('C0', 'C1', 'purple'), 
        alpha=0.45
    )
    for label in venn.subset_labels:
        if label:  # Verificar que la etiqueta no sea None
            label.set_fontsize(18)
    for label in venn.set_labels:
        if label:  # Verificar que la etiqueta no sea None
            label.set_fontsize(18)
    plt.savefig(savefig_path / f'venn3_{triple_combination}.png', transparent=True, dpi=500)
    plt.close()



plt.figure(layout='tight')
plt.title(fr'Areas', fontsize=20)

# Make plot
label_st1 = f'H'
label_st2 = f'WL'
label_st3 = f'W2'
areas = [ # the order should be(100, 010, 110, 001, 101, 011, 111)
    .25, 
    .25, 
    .05, 
    .25,
    .05, 
    .05,
    .1
    ] 
total_area = sum(areas)
# areas = np.array([
#     0 if area<0 else area.round(3) 
#     for area in areas
# ]) # note that the sum gives shared model variance_123

# Normalize to give percentage of variance explained by full model
areas = (np.array(areas)*1000/total_area).round(2)
venn = venn3(
    subsets=areas, # left area diagram, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
    set_labels=(label_st1, label_st2, label_st3), 
    set_colors=('C0', 'C1', 'purple'), 
    alpha=0.45
)
subset_labels = [
    r'$H \setminus (WL \cup W2)$',
    r'$WL \setminus (H \cup W2)$',
    r'$(H \cap WL) \setminus W2$',
    r'$W2 \setminus (H \cup WL)$',
    r'$(H \cap W2) \setminus WL$',
    r'$(WL \cap W2) \setminus H$',
    r'$(H \cap WL \cap W2)$'
]
lista = [
    variance_shared_with_1, 
    variance_shared_with_2, 
    variance_shared_with_3,
    variance_shared_with_13, 
    variance_shared_with_23,
    variance_shared_with_12, 
    variance_int_complement_submodels
    ]
for el in lista:
    print(el)
for i, label in enumerate(venn.subset_labels):
    label.set_fontsize(18)
    if i in [0,1]:
        x, y = label.get_position()
        if i==0:
            x -= 0.03
        else:
            x += 0.03
        label.set_position((x, y-0.03))
    elif i ==2:
        x, y = label.get_position()
        label.set_position((x, y+0.03))
    elif i==len(venn.subset_labels)-1:
        x, y = label.get_position()
        label.set_position((x, y+0.03))
    elif i in [4,5]:
        x, y = label.get_position()
        if i==4:
            x -= 0.05
        else:
            x += 0.05
        label.set_position((x, y-0.03))
    else:
        x, y = label.get_position()
        label.set_position((x, y+0.07))
    label.set_text(subset_labels[i])

for label in venn.set_labels:
    label.set_text('')
    label.set_fontsize(18)
print(venn.set_labels)
plt.show()
plt.savefig(savefig_path / f'venn3_schematic.png', transparent=True, dpi=500)
plt.close()