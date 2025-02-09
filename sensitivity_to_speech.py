# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn
from matplotlib.lines import Line2D
from tqdm import tqdm

# Specific libraries
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import MDS

from scipy.spatial.distance import pdist, squareform
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode

# Modules
from funciones import load_pickle, get_maximum_correlation_channels
import config, plot 

# Relevant paths
situation, band = config.situations[0], config.bands[0]
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

group = ['/a/', '/e/', '/i/','/o/', '/u/',\
        '/g/', '/m/', '/l/','/R/', '/p/', '/b/']

consonants = [phonemes.index(ph) for ph in phonemes if ph not in group]
vowels = [phonemes.index(vowel) for vowel in group]

keys_to_phonemes_labels = {}
for i in range(len(phonemes)):
    if i in consonants:
        keys_to_phonemes_labels[i]='Consonantes'
    elif i in vowels:
        keys_to_phonemes_labels[i]='Vocales'
    # elif i in dipthongs:
    #     keys_to_phonemes_labels[i]='Diptongos'
    # elif i in semivowels:
    #     keys_to_phonemes_labels[i]='Semi-vocales'
    
        
# color_labels = {'Vocales':'green', 'Semi-vocales':'red', 'Diptongos':'orange', 'Consonantes':'blue'}
color_labels = {'Vocales':'C0', 'Consonantes':'C1'}

manual_labels = []
for j in np.arange(len(phonemes)): 
    if j in vowels:
        manual_labels.append(0)
    elif j in consonants:
        manual_labels.append(1)
    # elif j in semivowels:
    #     manual_labels.append(2)
    # elif j in dipthongs:
    #     manual_labels.append(3)   

# Take rolling windows 
sample_window = int((ROLLING_WINDOW_SECONDS*config.sr))
step = sample_window/config.sr    
rolling_windows_centers = ( config.times[:-(sample_window-1)] + step / 2 ) * 1e3
rolling_windows  = np.lib.stride_tricks.sliding_window_view(x = config.delays, window_shape=sample_window)

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

for metric_label, metric, metric_random, metric_significance, col in zip(['F-score', 'Aris', 'Nmis'], [f_scores, aris, nmis],[f_scores_random, aris_random, nmis_random], [f_scores_significance, aris_significance, nmis_significance], ['C0', 'C1', 'C2']):
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1, 
        sharex=True, 
        tight_layout=True, 
        figsize=(8,6)
        )

    ax.scatter(rolling_windows_centers, metric, zorder=1, s=6, color=col)
    ax.plot(rolling_windows_centers, metric, zorder=1, color=col)
    ax.fill_between(
        x=rolling_windows_centers, 
        y1=metric_random.min(axis=1), # min across all permutations
        y2=metric_random.max(axis=1), 
        alpha=0.1,
        label='Distribución del control', 
        color=col,
        zorder=1
        )
    ax.plot(
        rolling_windows_centers, 
        metric_random.mean(axis=1),
        color=col,
        alpha=.4,
        zorder=1,
        label='Valor medio de la distribución del control'
    )
    
    ax.fill_between(
        x=rolling_windows_centers, 
        y1=np.percentile(metric_random, 50-25, axis=1), # min across all permutations
        y2=np.percentile(metric_random, 50+25, axis=1),
        alpha=0.2,
        label='50 % de la distribución del control', 
        color=col,
        zorder=1
        )
    ax.plot(
        rolling_windows_centers[metric_significance==1].flatten(),
        max(metric)*1.25*np.ones(shape=int(np.sum(metric_significance))),
        '*',
        color='black',
        label='Valores significativos'
        )
    
    ax.axvline(
        x=rolling_windows_centers[metric==max(metric)][np.argmin(np.abs(rolling_windows_centers[metric==max(metric)] - np.mean(rolling_windows_centers[metric==max(metric)])))], 
        color='black', linestyle='--', linewidth=2, 
        label=f'{1e3*round(np.mean(rolling_windows[metric==max(metric)]/config.sr),2)} ms'
        )
    ax.set(
        ylabel=metric_label,
        xlabel='Tiempo (ms)',
        )
    ax.legend()
    ax.grid(visible=True, zorder=0)
    fig.show()

# Visualization
# if window.tolist() in rolling_windows_lalor:
# if window in rolling_windows_lalor:    
selected_window = np.argmax(f_scores)
selected_window_time = rolling_windows_centers[selected_window]+sample_window/2

# for p in range(41):
#     selected_window=p
plt.figure(figsize=(8, 6))
for i, row in dataframes[selected_window].iterrows():
    plt.scatter(row['Feature_1'], row['Feature_2'], c='black', s=.5, facecolors='none')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
    plt.text(row['Feature_1'], row['Feature_2'], f"{phonemes[i]}", fontsize=15, ha='right', color=color_labels[keys_to_phonemes_labels[i]])
legend_handles = [Line2D([0], [0], color=color_labels[name], lw=4, label=name) for name in color_labels]
plt.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
plt.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
plt.legend(handles=legend_handles, title="Categorías")
plt.title(f'Center time: {selected_window_time:.1f} ms- F-score: {f_scores[selected_window]:.2f}', fontsize=16)
plt.xlabel('MDS 1 (U.A)')
plt.ylabel('MDS 2 (U.A)')
plt.grid(True)
plt.show(block=False)