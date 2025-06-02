# Standard libraries
import os, numpy as np, pandas as pd, seaborn as sn
from tqdm import tqdm

# Specific libraries
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde, mode

import librosa
import mne

# Graphics
from matplotlib.patches import PathPatch, Patch
from matplotlib.transforms import Bbox
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import matplotlib.pylab as pylab
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib import rc
import scienceplots

# Modules
from utils.general_functions import load_pickle, dump_pickle, get_maximum_correlation_channels
import config, utils.plot as plot 

rc('text', usetex=True)
plt.style.use(['science'])
pylab.rcParams.update(
    {
        'legend.fontsize': 16,
        'legend.title_fontsize': 16,
        'figure.figsize': (10, 5),
        'figure.titlesize': 20,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize':16,
        'ytick.labelsize':16
        }
    )

# Relevant paths
situation, band = 'External', 'Theta'
figures_path = os.path.normpath(f'figures/{config.model}/{situation}/sensitivity_speech_latency/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
correlation_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/{band}/Phonemes-Discrete-Phonet.pkl')
mtrfs_path = os.path.normpath(f'saves/{config.model}/{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/Phonemes-Discrete-Phonet/total_weights_per_subject.pkl')

# Hyper parameters
NUMBER_OF_CLUSTERS = 2 
KMEANS_NRUNS = 200
ROLLING_WINDOW_SECONDS = .095 # 100 ms it will be a little more due to fixed sample rate .09 da god
SIGNIFICANCE = .05

# Read data n_subj, n_chans, n_feats, n_delays
average_weights_subjects = load_pickle( 
                        path=mtrfs_path
                        )['average_weights_subjects']
average_correlation_across_subject = load_pickle(
                                path=correlation_path
                                )['average_correlation_subjects'].mean(axis=0)

# Take average across all subjects, then select specific channels and apply average across all the selection
filter_best_chans = get_maximum_correlation_channels(average_correlation_across_subject=average_correlation_across_subject, number_of_lat_channels=config.relevant_channels)
average_weights = average_weights_subjects.mean(axis=0)[filter_best_chans].mean(axis=0) # n_feats, n_delays

# Classify labels for categorization
phonemes = config.Exp_info().phonemes_phonet.copy()
phonemes.remove('/sil/')

# group = [
#     '/a/', '/e/', '/i/','/o/', '/u/', \
#     '/g/', '/l/', '/m/', '/b/', '/p/',\
#     '/R/', '/L/', '/n/', '/g/', '/d/'
#     ]
cons_ph = ['/k/', '/f/', '/t/', '/s/', '/x/', '/tS/']
voc_ph = ['/a/', '/e/', '/i/', '/o/', '/u/', '/l/', '/m/', '/b/', '/R/']

consonants = [phonemes.index(ph) for ph in phonemes if ph in cons_ph]
vowels = [phonemes.index(vowel) for vowel in voc_ph]

# Filter just wanted groups and relabel groups
consonants_ordered = []
vowels_ordered = []
for k, i in enumerate(sorted(consonants+vowels)):
    if i in consonants:
      consonants_ordered.append(k)  
    elif i in vowels:
        vowels_ordered.append(k)  

average_weights = average_weights[sorted(consonants+vowels)]
phonemes = [phonemes[i] for i in sorted(consonants+vowels)]

keys_to_phonemes_labels = {}
for i in range(len(phonemes)):
    if i in consonants_ordered:
        keys_to_phonemes_labels[i]='Consonantes'
    elif i in vowels_ordered:
        keys_to_phonemes_labels[i]='Vocales'
    # elif i in dipthongs:
    #     keys_to_phonemes_labels[i]='Diptongos'
    # elif i in semivowels:
    #     keys_to_phonemes_labels[i]='Semi-vocales'
    
        
# color_labels = {'Vocales':'green', 'Semi-vocales':'red', 'Diptongos':'orange', 'Consonantes':'blue'}
color_labels = {'Vocales':'C0', 'Consonantes':'orange'}

manual_labels = []
for j in range(len(phonemes)): 
    if j in vowels_ordered:
        manual_labels.append(0)
    elif j in consonants_ordered:
        manual_labels.append(1)
    # elif j in semivowels:
    #     manual_labels.append(2)
    # elif j in dipthongs:
    #     manual_labels.append(3)   

# Take rolling windows 
sample_window = int((ROLLING_WINDOW_SECONDS*config.sr))
# step = sample_window/config.sr    
# rolling_windows_centers = ( config.times[:-(sample_window-1)] + step / 2 ) * 1e3
rolling_windows  = np.lib.stride_tricks.sliding_window_view(x = config.delays, window_shape=sample_window)
rolling_windows_centers = np.mean(rolling_windows/config.sr, axis=1)

rolling_windows_lalor = [(.05<=config.times)&((config.times)<=.1), (.1<=config.times)&((config.times)<=.15), (.15<=config.times)&((config.times)<=.2)]
rolling_windows_lalor = [np.where(window)[0].tolist() if len(np.where(window)[0].tolist())==6 else np.where(window)[0].tolist()[:-1] for window in rolling_windows_lalor]

# Empty arrays for metrics
f_scores = np.zeros(shape=len(rolling_windows))
f_scores_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))
f_scores_significance = np.zeros(shape=len(rolling_windows))

nmis = np.zeros(shape=len(rolling_windows))
nmis_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))
nmis_significance = np.zeros(shape=len(rolling_windows))

aris = np.zeros(shape=len(rolling_windows))
aris_significance = np.zeros(shape=len(rolling_windows))
aris_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))

dataframes = []

for k, window in tqdm(enumerate(rolling_windows),total=len(rolling_windows)):
    # Use multidimensional scaling (MDS) to convert distances into features 
    mds = MDS(
        n_components=average_weights.shape[0], 
        dissimilarity='precomputed', 
        # dissimilarity='euclidean',
        random_state=i,
        normalized_stress='auto'
        )

    correlation_matrix = np.corrcoef(average_weights[:, window])
    correlation_matrix[np.isnan(correlation_matrix)]=0
    dissimilarity_matrix = 1 - correlation_matrix

    # features = mds.fit_transform(average_weights[:, window])
    features = mds.fit_transform(dissimilarity_matrix)

    # Initialize k-means labels
    kmeans_labels = np.zeros(shape=(KMEANS_NRUNS, average_weights.shape[0]), dtype=int)
    
    # Apply KMeans on the derived feature space
    for i in range(KMEANS_NRUNS):
        kmeans = KMeans(
                n_clusters=NUMBER_OF_CLUSTERS, #consonants and no consonantes
                random_state=i,
                n_init='auto'
                )
        kmeans_labels[i] = kmeans.fit_predict(features)

    # Take mode
    kmeans_labels = mode(kmeans_labels, axis=0, keepdims=True).mode.flatten()
    
    # Compute confusion matrix between k-means clusters and manual labels
    conf_matrix = confusion_matrix(manual_labels, kmeans_labels)

    # Solve the label assignment problem, maximizing agreement
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Relabel k-means using the map to manual labels 
    kmeans_labels = np.array([label_mapping[label] for label in kmeans_labels])

    # Keep track of mappings
    dataframes.append(
                pd.DataFrame(
                        {
                        'Original_index': np.arange(average_weights.shape[0], dtype=int),
                        'Feature_1': features[:, 0],
                        'Feature_2': features[:, 1],
                        'Manual_label': manual_labels,
                        'Cluster_Label': kmeans_labels
                        }   
                    )   
                )   
    # Compute metrics
    f_scores[k] = f1_score(manual_labels, kmeans_labels, average='weighted')
    nmis[k] = normalized_mutual_info_score(manual_labels, kmeans_labels)
    aris[k] = adjusted_rand_score(manual_labels, kmeans_labels)

    # Add random permutation to make benchmark
    random_clusters = np.zeros(shape=(KMEANS_NRUNS*10, average_weights.shape[0]), dtype=int)
    for p in range(KMEANS_NRUNS*10):
        random_cluster = np.random.randint(0, 2, size=len(manual_labels))
        f_scores_random[k, p] = f1_score(manual_labels, random_cluster, average='weighted')
        aris_random[k, p] = adjusted_rand_score(manual_labels, random_cluster)
        nmis_random[k, p] = normalized_mutual_info_score(manual_labels, random_cluster)

    pval_f = (sum(f_scores_random[k]>f_scores[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    pval_a = (sum(aris_random[k]>aris[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    pval_n = (sum(nmis_random[k]>nmis[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    f_scores_significance[k] = pval_f<SIGNIFICANCE 
    aris_significance[k] = pval_a<SIGNIFICANCE
    nmis_significance[k] = pval_n<SIGNIFICANCE

# for metric_label, metric, metric_random, metric_significance, col in zip(['F-score', 'Aris', 'Nmis'], [f_scores, aris, nmis],[f_scores_random, aris_random, nmis_random], [f_scores_significance, aris_significance, nmis_significance], ['C0', 'C1', 'C2']):
metric_label, metric, metric_random, metric_significance, col = 'Ari', aris, aris_random, aris_significance, 'green'
# metric_label, metric, metric_random, metric_significance, col = 'F-scores', f_scores, f_scores_random, f_scores_significance, 'orange'
col='#eb5b34'

selected_window = 37#np.argmax(f_scores)
# selected_window = 37
selected_window_time = rolling_windows_centers[selected_window]

data = {
    'F-scores': f_scores,
    'Aris': aris,
    'Nmis': nmis,
    'F-scores_random': f_scores_random,
    'Aris_random': aris_random,
    'Nmis_random': nmis_random,
    'F-scores_significance': f_scores_significance,
    'Aris_significance': aris_significance,
    'Nmis_significance': nmis_significance,
    'rolling_windows_centers': rolling_windows_centers,
    'dataframes': dataframes,
    'phonemes': phonemes,
    'group_2_labels': cons_ph,
    'group_1_labels': voc_ph,
    'NUMBER_OF_CLUSTERS': NUMBER_OF_CLUSTERS,
    'KMEANS_NRUNS': KMEANS_NRUNS,
    'ROLLING_WINDOW_SECONDS': ROLLING_WINDOW_SECONDS,
    'SIGNIFICANCE': SIGNIFICANCE,
    'selected_window': selected_window,
    'keys_to_phonemes_labels': keys_to_phonemes_labels
}
os.makedirs(os.path.normpath(os.path.join('saves', 'mtrf_ridge_torch', 'External', 'sensitivity_speech_latency')), exist_ok=True)
dump_pickle(path=os.path.normpath(os.path.join('saves', 'mtrf_ridge_torch', 'External', 'sensitivity_speech_latency', 'phonemes_discrete_phonet.pkl')), obj=data, rewrite=True)

# Graficamos
fig, axes = plt.subplots(
    figsize=(14, 7), 
    nrows=1,
    ncols=2,
    # tight_layout=True
    )

# MÉTRICA
axes[0].grid(True)
axes[0].vlines(selected_window_time*1e3, ymin=-0.06, ymax=.9, linestyle='-.', linewidth=1.5, color='black')

# Percentiles y datos
lower_percentile = np.percentile(metric_random, 5, axis=1)
upper_percentile = np.percentile(metric_random, 95, axis=1)

# Aplicamos el degradado basado en la densidad
plot.plot.gradient_fill_density_based(
    rolling_windows_centers*1e3, 
    lower_percentile, 
    upper_percentile, 
    metric_random, 
    fill_color=col, 
    ax=axes[0]
    )

# Puntos significativos
axes[0].plot(
    rolling_windows_centers[metric_significance == 1].flatten()*1e3,
    max(metric) * 1.25 * np.ones(int(np.sum(metric_significance))),
    '*', 
    color='black',
    label='Valores significativos',
    zorder=4
    )

axes[0].plot(rolling_windows_centers*1e3, metric, color=col, zorder=3)
axes[0].scatter(rolling_windows_centers*1e3, metric, color=col, s=6, zorder=3, label='Métrica')

axes[0].set(xlabel='Tiempo (ms)', ylabel=metric_label.upper(), xlim=(5, 550), title=r'\textit{Adjusted Rand Index}')

gradient_patch = Patch(facecolor=col, alpha=0.5, label=r'Distribución nula (5-95\%)')
handles, labels = axes[0].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
handles.append(gradient_patch)
labels.append(r'Distribución nula (5\% - 95\%)')
handles = handles[::-1]
labels = labels[::-1]
axes[0].legend(handles=handles, labels=labels, loc=(.005,.73))

# Visualization
transfo ={'Vocales':'Grupo 1', 'Consonantes':'Grupo 2'}
for i, row in dataframes[selected_window].iterrows():
    axes[1].scatter(row['Feature_1']*1e2, row['Feature_2']*1e1, c='black', s=.5, facecolors='none')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
    axes[1].text(row['Feature_1']*1e2, row['Feature_2']*1e1, r"\textbf{" + f"{phonemes[i]}" + r"}", fontsize=15, ha='right', color=color_labels[keys_to_phonemes_labels[i]])
legend_handles = [Line2D([0], [0], color=color_labels[name], lw=4, label=transfo[name]) for name in color_labels]

axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

axes[1].legend(handles=legend_handles, title="Categorías", loc=(.55,.74))
axes[1].set_title(f'Ventana temporal: {selected_window_time*1e3:.1f} ms - F-score: {f_scores[selected_window]:.2f} - ARI: {aris[selected_window]:.2f}'.replace('.',','), fontsize=16)
axes[1].set_xlabel('MDS 1 (U.A)')
axes[1].set_ylabel('MDS 2 (U.A)')
axes[1].set_xlim(axes[1].get_xlim()[0]-.4, axes[1].get_xlim()[-1])
# axes[1].set_ylim(axes[1].get_ylim()[0]-.005, axes[1].get_ylim()[-1]+.005)
axes[1].grid(True, alpha=.7)

# Remover los patches de degradado
for ax in fig.axes:
    for im in ax.images:
        im.set_clip_path(None)

# Ahora guardar la figura sin que se calcule el bbox de esos patches
fig.savefig(
    os.path.normpath(os.path.join('figures','figuras_tesis', 'resultados', 'mapa_fonemas.png')),
    transparent=False,
    bbox_inches='tight',
    dpi=400
)
fig.show()

# =====================================
# REEPLICA DE ANALISIS PARA FONOLOFICAS
# =====================================

# Relevant paths
situation, band = 'External', 'Theta'
figures_path = os.path.normpath(f'figures/{config.model}/{situation}/sensitivity_speech_latency/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
correlation_path = os.path.normpath(f'saves/{config.model}/{situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/{band}/Phonological.pkl')
mtrfs_path = os.path.normpath(f'saves/{config.model}/{situation}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/Phonological/total_weights_per_subject.pkl')

# Hyper parameters
NUMBER_OF_CLUSTERS = 2 
KMEANS_NRUNS = 200
ROLLING_WINDOW_SECONDS = .095 # 100 ms it will be a little more due to fixed sample rate .09 da god
SIGNIFICANCE = .05

# Read data n_subj, n_chans, n_feats, n_delays
average_weights_subjects = load_pickle( 
                        path=mtrfs_path
                        )['average_weights_subjects']
average_correlation_across_subject = load_pickle(
                                path=correlation_path
                                )['average_correlation_subjects'].mean(axis=0)

# Take average across all subjects, then select specific channels and apply average across all the selection
filter_best_chans = get_maximum_correlation_channels(average_correlation_across_subject=average_correlation_across_subject, number_of_lat_channels=config.relevant_channels)
average_weights = average_weights_subjects.mean(axis=0)[filter_best_chans].mean(axis=0) # n_feats, n_delays

# Classify labels for categorization
phonological = list(config.Exp_info().phonological_labels).copy()

group1_labels = ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal']
group2_labels = ['dental', 'consonantal', 'pause', 'velar', 'flap', 'close', 'strident', 'continuant']

group1 = [phonological.index(ph) for ph in phonological if ph in group1_labels]
group2 = [phonological.index(vowel) for vowel in group2_labels]

# Filter just wanted groups and relabel groups
group1_ordered = []
group2_ordered = []
for k, i in enumerate(sorted(group1+group2)):
    if i in group1:
      group1_ordered.append(k)  
    elif i in group2:
        group2_ordered.append(k)  

average_weights = average_weights[sorted(group1+group2)]
phonological = [phonological[i] for i in sorted(group1+group2)]

keys_to_phonological_labels = {}
for i in range(len(phonological)):
    if i in group1_ordered:
        keys_to_phonological_labels[i]='Consonantes'
    elif i in group2_ordered:
        keys_to_phonological_labels[i]='Vocales'
    # elif i in dipthongs:
    #     keys_to_phonological_labels[i]='Diptongos'
    # elif i in semigroup2:
    #     keys_to_phonological_labels[i]='Semi-vocales'
    
        
# color_labels = {'Vocales':'green', 'Semi-vocales':'red', 'Diptongos':'orange', 'Consonantes':'blue'}
color_labels = {'Vocales':'C0', 'Consonantes':'orange'}

manual_labels = []
for j in range(len(phonological)): 
    if j in group2_ordered:
        manual_labels.append(0)
    elif j in group1_ordered:
        manual_labels.append(1)
    # elif j in semigroup2:
    #     manual_labels.append(2)
    # elif j in dipthongs:
    #     manual_labels.append(3)   

# Take rolling windows 
sample_window = int((ROLLING_WINDOW_SECONDS*config.sr))
# step = (sample_window-1)/config.sr    
# rolling_windows_centers = ( config.times[:-(sample_window-1)] + step / 2) * 1e3
rolling_windows  = np.lib.stride_tricks.sliding_window_view(x = config.delays, window_shape=sample_window)
rolling_windows_centers = np.mean(rolling_windows/config.sr, axis=1)

rolling_windows_lalor = [(.05<=config.times)&((config.times)<=.1), (.1<=config.times)&((config.times)<=.15), (.15<=config.times)&((config.times)<=.2)]
rolling_windows_lalor = [np.where(window)[0].tolist() if len(np.where(window)[0].tolist())==6 else np.where(window)[0].tolist()[:-1] for window in rolling_windows_lalor]

# Empty arrays for metrics
f_scores = np.zeros(shape=len(rolling_windows))
f_scores_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))
f_scores_significance = np.zeros(shape=len(rolling_windows))

nmis = np.zeros(shape=len(rolling_windows))
nmis_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))
nmis_significance = np.zeros(shape=len(rolling_windows))

aris = np.zeros(shape=len(rolling_windows))
aris_significance = np.zeros(shape=len(rolling_windows))
aris_random = np.zeros((len(rolling_windows), KMEANS_NRUNS*10))

dataframes = []

for k, window in tqdm(enumerate(rolling_windows),total=len(rolling_windows)):
    # Use multidimensional scaling (MDS) to convert distances into features 
    mds = MDS(
        n_components=average_weights.shape[0], 
        dissimilarity='precomputed', 
        # dissimilarity='euclidean',
        random_state=i,
        normalized_stress='auto'
        )

    correlation_matrix = np.corrcoef(average_weights[:, window])
    correlation_matrix[np.isnan(correlation_matrix)]=0
    dissimilarity_matrix = 1 - correlation_matrix

    # features = mds.fit_transform(average_weights[:, window])
    features = mds.fit_transform(dissimilarity_matrix)

    # Initialize k-means labels
    kmeans_labels = np.zeros(shape=(KMEANS_NRUNS, average_weights.shape[0]), dtype=int)
    
    # Apply KMeans on the derived feature space
    for i in range(KMEANS_NRUNS):
        kmeans = KMeans(
                n_clusters=NUMBER_OF_CLUSTERS, #consonants and no consonantes
                random_state=i,
                n_init='auto'
                )
        kmeans_labels[i] = kmeans.fit_predict(features)

    # Take mode
    kmeans_labels = mode(kmeans_labels, axis=0, keepdims=True).mode.flatten()
    
    # Compute confusion matrix between k-means clusters and manual labels
    conf_matrix = confusion_matrix(manual_labels, kmeans_labels)

    # Solve the label assignment problem, maximizing agreement
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}

    # Relabel k-means using the map to manual labels 
    kmeans_labels = np.array([label_mapping[label] for label in kmeans_labels])

    # Keep track of mappings
    dataframes.append(
                pd.DataFrame(
                        {
                        'Original_index': np.arange(average_weights.shape[0], dtype=int),
                        'Feature_1': features[:, 0],
                        'Feature_2': features[:, 1],
                        'Manual_label': manual_labels,
                        'Cluster_Label': kmeans_labels
                        }   
                    )   
                )   
    # Compute metrics
    f_scores[k] = f1_score(manual_labels, kmeans_labels, average='weighted')
    nmis[k] = normalized_mutual_info_score(manual_labels, kmeans_labels)
    aris[k] = adjusted_rand_score(manual_labels, kmeans_labels)

    # Add random permutation to make benchmark
    random_clusters = np.zeros(shape=(KMEANS_NRUNS*10, average_weights.shape[0]), dtype=int)
    for p in range(KMEANS_NRUNS*10):
        random_cluster = np.random.randint(0, 2, size=len(manual_labels))
        f_scores_random[k, p] = f1_score(manual_labels, random_cluster, average='weighted')
        aris_random[k, p] = adjusted_rand_score(manual_labels, random_cluster)
        nmis_random[k, p] = normalized_mutual_info_score(manual_labels, random_cluster)

    pval_f = (sum(f_scores_random[k]>f_scores[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    pval_a = (sum(aris_random[k]>aris[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    pval_n = (sum(nmis_random[k]>nmis[k]) + 1)/(KMEANS_NRUNS*10 + 1)
    f_scores_significance[k] = pval_f<SIGNIFICANCE 
    aris_significance[k] = pval_a<SIGNIFICANCE
    nmis_significance[k] = pval_n<SIGNIFICANCE

# for metric_label, metric, metric_random, metric_significance, col in zip(['F-score', 'Aris', 'Nmis'], [f_scores, aris, nmis],[f_scores_random, aris_random, nmis_random], [f_scores_significance, aris_significance, nmis_significance], ['C0', 'C1', 'C2']):
metric_label, metric, metric_random, metric_significance, col = 'Ari', aris, aris_random, aris_significance, 'green'
# metric_label, metric, metric_random, metric_significance, col = 'F-scores', f_scores, f_scores_random, f_scores_significance, 'orange'
col='#eb5b34'

selected_window = 41#np.argmax(f_scores)
aris[selected_window]
selected_window_time = rolling_windows_centers[selected_window]*1e3

data = {
    'F-scores': f_scores,
    'Aris': aris,
    'Nmis': nmis,
    'F-scores_random': f_scores_random,
    'Aris_random': aris_random,
    'Nmis_random': nmis_random,
    'F-scores_significance': f_scores_significance,
    'Aris_significance': aris_significance,
    'Nmis_significance': nmis_significance,
    'rolling_windows_centers': rolling_windows_centers,
    'dataframes': dataframes,
    'phonological': phonological,
    'group_1_labels': group1_labels,
    'group_2_labels': group2_labels,
    'NUMBER_OF_CLUSTERS': NUMBER_OF_CLUSTERS,
    'KMEANS_NRUNS': KMEANS_NRUNS,
    'ROLLING_WINDOW_SECONDS': ROLLING_WINDOW_SECONDS,
    'SIGNIFICANCE': SIGNIFICANCE,
    'selected_window': selected_window,
    'keys_to_phonological_labels': keys_to_phonological_labels
}
os.makedirs(os.path.normpath(os.path.join('saves', 'mtrf_ridge_torch', 'External', 'sensitivity_speech_latency')), exist_ok=True)
dump_pickle(path=os.path.normpath(os.path.join('saves', 'mtrf_ridge_torch', 'External', 'sensitivity_speech_latency', 'phonological.pkl')), obj=data)
# Graficamos
fig, axes = plt.subplots(
    figsize=(14, 7), 
    nrows=1,
    ncols=2,
    # tight_layout=True
    )

# MÉTRICA
axes[0].grid(True)
axes[0].vlines(selected_window_time, ymin=-0.06, ymax=.9, linestyle='-.', linewidth=1.5, color='black')

max(metric) * 1.25
# Percentiles y datos
lower_percentile = np.percentile(metric_random, 5, axis=1)
upper_percentile = np.percentile(metric_random, 95, axis=1)

# Aplicamos el degradado basado en la densidad
plot.gradient_fill_density_based(
    rolling_windows_centers*1e3, 
    lower_percentile, 
    upper_percentile, 
    metric_random, 
    fill_color=col, 
    ax=axes[0]
    )

# Puntos significativos
axes[0].plot(
    rolling_windows_centers[metric_significance == 1].flatten()*1e3,
    max(metric) * 1.25 * np.ones(int(np.sum(metric_significance))),
    '*', 
    color='black',
    label='Valores significativos',
    zorder=4
    )

axes[0].plot(rolling_windows_centers*1e3, metric, color=col, zorder=3)
axes[0].scatter(rolling_windows_centers*1e3, metric, color=col, s=6, zorder=3, label='Métrica')



axes[0].set(xlabel='Tiempo (ms)', ylabel=metric_label.upper(), xlim=(5, 550), title=r'\textit{Adjusted Rand Index}')

gradient_patch = Patch(facecolor=col, alpha=0.5, label=r'Distribución nula (5-95\%)')
handles, labels = axes[0].get_legend_handles_labels()
handles = handles[::-1]
labels = labels[::-1]
handles.append(gradient_patch)
labels.append(r'Distribución nula (5\% - 95\%)')
handles = handles[::-1]
labels = labels[::-1]
axes[0].legend(handles=handles, labels=labels, loc=(.005,.73))

# Visualization
transfo ={'Vocales':'Grupo 1', 'Consonantes':'Grupo 2'}
for i, row in dataframes[selected_window].iterrows():
    axes[1].scatter(row['Feature_1']*1e1, row['Feature_2']*1e1, c='black', s=.5, facecolors='none')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
    axes[1].text(row['Feature_1']*1e1, row['Feature_2']*1e1, r"\textbf{" + f"{phonological[i]}" + r"}", fontsize=15, ha='right', color=color_labels[keys_to_phonological_labels[i]])
legend_handles = [Line2D([0], [0], color=color_labels[name], lw=4, label=transfo[name]) for name in color_labels]

axes[1].ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

axes[1].legend(handles=legend_handles, title="Categorías", loc=(.57,.74))
axes[1].set_title(f'Ventana temporal: {selected_window_time:.1f} ms - F-score: {f_scores[selected_window]:.2f} - ARI: {aris[selected_window]:.2f}'.replace('.',','), fontsize=16)
axes[1].set_xlabel('MDS 1 (U.A)')
axes[1].set_ylabel('MDS 2 (U.A)')
axes[1].set_xlim(axes[1].get_xlim()[0]-.9, axes[1].get_xlim()[-1]+.001)
# axes[1].set_ylim(axes[1].get_ylim()[0]-.005, axes[1].get_ylim()[-1]+.005)
axes[1].grid(True, alpha=.7)

# Remover los patches de degradado
for ax in fig.axes:
    for im in ax.images:
        im.set_clip_path(None)

# Ahora guardar la figura sin que se calcule el bbox de esos patches
# fig.savefig(
#     'C:/Users/jocta/Documents/tesis_escrita/imagenes/resultados/mapa_fonologicas.svg',
#     transparent=True,
#     bbox_inches='tight'
# )
fig.show()

