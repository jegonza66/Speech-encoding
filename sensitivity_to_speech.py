# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn

# Specific libraries
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode
from sklearn.manifold import MDS
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

import matplotlib.pylab as pylab# Default size is 10 pts, the scalings (10pts*scale) are: #'xx-small':0.579,'x-small':0.694,'s # mall':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
from matplotlib.lines import Line2D
pylab.rcParams.update(
                    {
                    'legend.fontsize': 'xx-large',
                    'legend.title_fontsize': 'xx-large',
                    'figure.figsize': (8, 6),
                    'figure.titlesize': 'xx-large',
                    'axes.labelsize': 'xx-large',
                    'axes.titlesize':'xx-large',
                    'xtick.labelsize':'x-large',
                    'ytick.labelsize':'x-large'
                    }
                    )

# Modules
from funciones import load_pickle
import config, plot 

# Whether to use or not just relevant channels
relevant_channels = 12 # None
number_of_clusters = 2 #consonants and vocals

# true_labels = dict()

# Relevant paths
figures_path = os.path.normpath(f'figures/sensitivity_speech_latency/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
correlation_path = os.path.normpath(f'saves/{config.model}/{config.situation}/correlations/tmin{config.tmin}_tmax{config.tmax}/{config.bands[0]}/Phonemes_Discrete_Phonet.pkl')
mtrfs_path = os.path.normpath(f'saves/{config.situation}/{config.model}/weights/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{config.bands[0]}/Phonemes-Discrete-Phonet/total_weights_per_subject.pkl')

# Read data
average_weights_subjects = load_pickle( # n_subj, n_chans, n_feats, n_delays
                                    path=os.path.join(
                                                    mtrfs_path, 
                                                    config.bands[0], 
                                                    'phonemes',
                                                    'total_weights_per_subject.pkl'
                                                    )
                                    )['average_weights_subjects']

# Take average across all subjects, then select specific channels and apply average across all selected channels
average_weights = average_weights_subjects.mean(axis=0)[config.channel_selection].mean(axis=0) # n_feats, n_delays
average_weights = average_weights[[i for i in np.arange(39) if i not in [4,37]]] #remove bad phones
plot.average_regression_weights(
                            average_weights_subjects=average_weights_subjects,
                            info=config.info,
                            save=config.save_figures,
                            save_path='figures/',
                            times=config.times,
                            n_feats=[37],
                            stim='phonemes',
                            selection=config.channel_selection,
                            hierarchical_clustering=False
                            )

# Create true labels for categorization
phonemes = config.ordered_phonemes.copy()
vowels = [phonemes.index(vowel) for vowel in list(np.unique(list(config.vowels.values())))]
semivowels = [phonemes.index(semivowel) for semivowel in list(np.unique(list(config.semivowels.values())))]
dipthongs = [phonemes.index(dip) for dip in list(np.unique(list(config.dipthongs.values())))]
consonants = [phonemes.index(conso) for conso in list(np.unique(list(config.consonants.values())))]
keys_to_phonemes_labels = {}
for i in range(len(phonemes)):
    if i in consonants:
        keys_to_phonemes_labels[i]='Consonantes'
    elif i in dipthongs:
        keys_to_phonemes_labels[i]='Diptongos'
    elif i in semivowels:
        keys_to_phonemes_labels[i]='Semi-vocales'
    elif i in vowels:
        keys_to_phonemes_labels[i]='Vocales'
        
color_labels = {'Vocales':'green', 'Semi-vocales':'red', 'Diptongos':'orange', 'Consonantes':'blue'}

original_manual_labels = []
for j in np.arange(len(phonemes)): 
    if j in vowels:
        original_manual_labels.append(0)
    elif j in semivowels:
        original_manual_labels.append(1)
    elif j in dipthongs:
        original_manual_labels.append(2)
    elif j in consonants:
        original_manual_labels.append(3)
        
# To split between consonants and non-consonants
manual_labels = np.array(original_manual_labels)
manual_labels[manual_labels!=3] = 0
manual_labels[manual_labels==3] = 1

# Take rolling windows 
rolling_window = .05 # 50 ms it will be a little more due to fixed sample rate
sample_window = int(rolling_window*config.eeg_sample_rate)
rolling_windows  = np.lib.stride_tricks.sliding_window_view(x = config.delays, window_shape=sample_window)
rolling_windows_lalor = [(.05<=config.times)&((config.times)<=.1), (.1<=config.times)&((config.times)<=.15), (.15<=config.times)&((config.times)<=.2)]
rolling_windows_lalor = [np.where(window)[0].tolist() if len(np.where(window)[0].tolist())==6 else np.where(window)[0].tolist()[:-1] for window in rolling_windows_lalor]

f_scores, nmis, aris = [], [], [] 
f_scores_random, nmis_random, aris_random = [], [], [] 
dataframes = []
# Number of k-means runs
nruns = 200
for k, window in enumerate(rolling_windows):
    print(f'{round(100*k/len(rolling_windows),2)}%', end='\n')
    
    # Use multidimensional scaling (MDS) to convert distances into features 
    mds = MDS(
            n_components=5,#average_weights.shape[0], 
            dissimilarity='euclidean', 
            random_state=i,
            normalized_stress='auto'
            )
    
    features = mds.fit_transform(average_weights[:, window])

    # Initialize k-means labels
    kmeans_labels = np.zeros(shape=(nruns, average_weights.shape[0]), dtype=int)
    
    # Apply KMeans on the derived feature space
    for i in range(nruns):
        kmeans = KMeans(
                        n_clusters=2, #consonants and no consonantes
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
    dataframes.append(pd.DataFrame({
    'Original_index': np.arange(average_weights.shape[0], dtype=int),
    'Feature_1': features[:, 0],
    'Feature_2': features[:, 1],
    'Manual_label': original_manual_labels,
    'Cluster_Label': kmeans_labels
    })
    )
    # Compute metrics
    f_scores.append(f1_score(manual_labels, kmeans_labels, average='weighted'))
    nmis.append(normalized_mutual_info_score(manual_labels, kmeans_labels))
    aris.append(adjusted_rand_score(manual_labels, kmeans_labels))

    # Add random permutation to make benchmark
    random_clusters = np.zeros(shape=(nruns*10, average_weights.shape[0]), dtype=int)
    for p in range(nruns*10):
        random_clusters[p] = np.random.randint(0, 2, size=len(manual_labels))
    random_clusters = mode(random_clusters, axis=0, keepdims=True).mode.flatten()

    # Compute metrics for random clustering
    f_scores_random.append(f1_score(manual_labels, random_clusters, average='weighted'))
    aris_random.append(adjusted_rand_score(manual_labels, random_clusters))
    nmis_random.append(normalized_mutual_info_score(manual_labels, random_clusters))

step = sample_window/config.eeg_sample_rate    
average_windows = (config.times[:-5]+step/2)*1e3
# average_windows = (config.times[:-5])*1e3
fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True, tight_layout=True, figsize=(7,9))

ax[0].scatter(average_windows, f_scores, label='F-scores', zorder=1, s=6, color='C0')
ax[0].plot(average_windows, f_scores, zorder=1, color='C0')
ax[0].scatter(average_windows, f_scores_random, label='Moda del control', s=6, zorder=1, color='C0', alpha=.2)
ax[0].plot(average_windows, f_scores_random, zorder=1, color='C0', alpha=.4)
ax[0].axvline(x=average_windows[nmis.index(max(nmis))], color='black', linestyle='--', linewidth=2, label=f'{1e3*round(np.mean(rolling_windows[nmis.index(max(nmis))]/config.eeg_sample_rate),2)} ms')
ax[0].set_ylabel('Puntuación')
ax[0].legend(title='Métrica')
ax[0].grid(visible=True, zorder=0)
ax[1].scatter(average_windows, aris, label='Adj. Rand Index', zorder=1, s=6, color='C1')
ax[1].plot(average_windows, aris, zorder=1, color='C1')
ax[1].scatter(average_windows, aris_random, label='Moda del control', s=6, zorder=1, color='C1', alpha=.2)
ax[1].plot(average_windows, aris_random, zorder=1, color='C1', alpha=.4)
ax[1].axvline(x=average_windows[nmis.index(max(nmis))], color='black', linestyle='--', linewidth=2, label=f'{1e3*round(np.mean(rolling_windows[nmis.index(max(nmis))]/config.eeg_sample_rate),2)} ms')
ax[1].set_ylabel('Puntuación')
ax[1].legend(title='Métrica')
ax[1].grid(visible=True, zorder=0)

ax[2].scatter(average_windows, nmis, label='Normalized Mutual Inf.', zorder=1, s=6, color='C2')
ax[2].plot(average_windows, nmis, zorder=1, color='C2')
ax[2].scatter(average_windows, nmis_random, label='Moda del control', s=6, zorder=1, color='C2', alpha=.2)
ax[2].plot(average_windows, nmis_random, zorder=1, color='C2', alpha=.4)
ax[2].axvline(x=average_windows[nmis.index(max(nmis))], color='black', linestyle='--', linewidth=2, label=f'{1e3*round(np.mean(rolling_windows[nmis.index(max(nmis))]/config.eeg_sample_rate),2)} ms')
ax[2].set_ylabel('Puntuación')
ax[2].legend(title='Métrica')
ax[2].grid(visible=True, zorder=0)
# plt.scatter(average_windows, aris, label='Adj. Rand Index', zorder=2, color='C1')
# plt.plot(average_windows, aris, zorder=2, color='C1')
# plt.scatter(average_windows, nmis, label='Normalized Mutual Inf.', zorder=3, color='C2')
# plt.plot(average_windows, nmis, zorder=3, color='C2')
ax[2].set_xlabel('Valor medio de la ventana (ms)')

fig.savefig('figures/metricas.png', dpi=600)

# ventanas_en_tiempo =  [window/config.eeg_sample_rate for window in rolling_windows]
# ventanas_en_tiempo[nmis.index(max(nmis))]

# Visualization
# if window.tolist() in rolling_windows_lalor:
# if window in rolling_windows_lalor:    
selected_window = nmis.index(max(nmis))

plt.figure(figsize=(8, 6))
for i, row in dataframes[selected_window].iterrows():
    plt.scatter(row['Feature_1'], row['Feature_2'], c='black', s=.5, facecolors='none')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
    plt.text(row['Feature_1'], row['Feature_2'], f"{config.ordered_phonemes[i]}", fontsize=15, ha='right', color=color_labels[keys_to_phonemes_labels[i]])
legend_handles = [Line2D([0], [0], color=color_labels[name], lw=4, label=name) for name in color_labels]
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
plt.legend(handles=legend_handles, title="Categorías")
plt.title(f'Consonantes vs. No consonantes - F-score: {round(f_scores[selected_window],2)}', fontsize=16)
plt.xlabel('MDS 1 (u.a)')
plt.ylabel('MDS 2 (u.a)')
plt.grid(True)
plt.savefig(f'figures/clusterizacion.png', dpi=600)
