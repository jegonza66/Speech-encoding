"""
This script generates a correlation matrix indicating the correlation value of different attributes
"""
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import pandas as pd
import numpy as np

from utils.general_functions import load_pickle
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
    ax=ax
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
    ax=ax
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
from utils.plot import define_ticks, clustering_by_correlation
from matplotlib.collections import PathCollection
import matplotlib.text as mtext
import mne
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

from matplotlib_venn import venn3  
from utils.general_functions import load_pickle
models = [
    'Spectrogram-21', 
    'Phonemes-Discrete', 
    'Phonological',
    'Spectrogram-21_Phonemes-Discrete',
    'Spectrogram-21_Phonological',
    'Phonemes-Discrete_Phonological',
    'Spectrogram-21_Phonemes-Discrete_Phonological'
]

corr_path = lambda model: Path(f"output/mtrf-ridge/External-External/correlations/same_alpha/tmin-0.2_tmax0.6/Broad/{model}.pkl")

correlations = {
    model: load_pickle(corr_path(model))['average_correlation_subjects'].mean()
    for model in models
}

# Get Venn diagrams for triple combinations
savefig_path = Path("figures/analysis/model_visualization_matrix_corr/venn3")
savefig_path.mkdir(parents=True, exist_ok=True)

triple_combination = 'Spectrogram-21_Phonemes-Discrete_Phonological'
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
plt.title(f'Spectrogram ∪ Phonemes ∪ Phonological')

# Make plot
venn3(
    subsets=areas, # left area diagram, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
    set_labels=(st1, st2, st3), 
    set_colors=('C0', 'C1', 'purple'), 
    alpha=0.45
    )

plt.savefig(savefig_path / f'venn3_{triple_combination}.png')
plt.close()