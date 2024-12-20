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
                    'legend.fontsize': 'x-large',
                    'legend.title_fontsize': 'x-large',
                    'figure.figsize': (8, 6),
                    'figure.titlesize': 'xx-large',
                    'axes.labelsize': 'x-large',
                    'axes.titlesize':'x-large',
                    'xtick.labelsize':'large',
                    'ytick.labelsize':'large'
                    }
                    )

# Modules
from auxiliary import load_pickle
import config, plot 

# Whether to use or not just relevant channels
relevant_channels = 12 # None
number_of_clusters = 2 #consonants and vocals
rolling_window = .05 # 50 ms it will be a little more due to fixed sample rate
sample_window = int(.05*config.eeg_sample_rate)
# true_labels = dict()

# Relevant paths
figures_path = os.path.normpath(f'figures/sensitivity_speech_latency/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
correlation_path = os.path.normpath(f'output/correlations/tmin{config.tmin}_tmax{config.tmax}/')
mtrfs_path = os.path.normpath(f'output/mtrfs//stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/')
preprocesed_data_path = os.path.normpath(f'preprocessed_data/tmin{config.tmin}_tmax{config.tmax}/') 

# Read data
average_weights_subjects = load_pickle( # n_subj, n_chans, n_feats, n_delays
                                    path=os.path.join(
                                                    mtrfs_path, 
                                                    config.bands[0], 
                                                    'phonemes',
                                                    'total_weights_per_subject.pkl'
                                                    )
                                    )['average_weights_subjects']

plot.average_regression_weights(
                            average_weights_subjects=average_weights_subjects,
                            info=config.info,
                            save=config.save_figures,
                            save_path='figures/',
                            times=config.times,
                            n_feats=[39],
                            stim=config.stimuli[0],
                            selection=config.channel_selection,
                            hierarchical_clustering=False
                            )

# Take average across all subjects, then select specific channels and apply average across all selected channels
average_weights = average_weights_subjects.mean(axis=0)[config.channel_selection].mean(axis=0) # n_feats, n_delays
average_weights = average_weights[[i for i in np.arange(39) if i not in [4,37]]] #remove bad phones

# Create true labels for categorization
phonemes = config.ordered_phonemes.copy()
vowels = [phonemes.index(vowel) for vowel in list(np.unique(list(config.vowels.values())))]
semivowels = [phonemes.index(semivowel) for semivowel in list(np.unique(list(config.semivowels.values())))]
dipthongs = [phonemes.index(dip) for dip in list(np.unique(list(config.dipthongs.values())))]
consonants = [phonemes.index(conso) for conso in list(np.unique(list(config.consonants.values())))]
color_labels = {'Vocales':'green', 'Semi-vocales':'pink', 'Diptongos':'orange', 'Consonantes':'blue'}

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
# rolling_windows  = np.lib.stride_tricks.sliding_window_view(x = config.delays, window_shape=sample_window)
rolling_windows = [(.05<=config.times)&((config.times)<=.1), (.1<=config.times)&((config.times)<=.15), (.15<=config.times)&((config.times)<=.2)]
f_scores, nmis, aris = [], [], [] 

# Comput F-score for each window
for k, window in enumerate(rolling_windows):
    # Compute pairwise correlation matrix
    correlation_matrix = np.corrcoef(average_weights[:, window]) # n_feats X n_feats

    # Step 2: Convert correlation matrix to a dissimilarity matrix
    dissimilarity_matrix = 1 - correlation_matrix

    # Ensure the diagonal of the distance matrix is zero and that the matrix is symmetric
    np.fill_diagonal(dissimilarity_matrix, 0)
    dissimilarity_matrix = (dissimilarity_matrix + dissimilarity_matrix.T) / 2
    dissimilarity_matrix = np.nan_to_num(dissimilarity_matrix, nan=1)

    # # Flatten the dissimilarity matrix to compute distances for KMeans
    # distance_matrix = squareform(dissimilarity_matrix)

    # features = []
    nruns=100
    kmeans_labels = np.zeros(shape=(nruns, average_weights.shape[0]), dtype=int)
    # Use multidimensional scaling (MDS) to convert distances into features # TODO SALTAR
    mds = MDS(
            n_components=average_weights.shape[0], 
            dissimilarity='euclidean', 
            random_state=42,
            normalized_stress='auto'
            )
    # pca = PCA(n_components=2, random_state=42)
    # features.append(pca.fit_transform(mds.fit_transform(dissimilarity_matrix)))
    # features = mds.fit_transform(dissimilarity_matrix)
    features = mds.fit_transform(average_weights[:, window])
    mds_2 = MDS(
            n_components=5,#average_weights.shape[0], 
            dissimilarity='precomputed', 
            random_state=42,
            normalized_stress='auto',
            )
    features_2 = mds_2.fit_transform(dissimilarity_matrix)
    for i in range(nruns):
        # # Use multidimensional scaling (MDS) to convert distances into features # TODO SALTAR
        # mds = MDS(
        #         n_components=5,#average_weights.shape[0], 
        #         dissimilarity='precomputed', 
        #         random_state=42,
        #         normalized_stress='auto'
        #         )
        # # pca = PCA(n_components=2, random_state=42)
        # # features.append(pca.fit_transform(mds.fit_transform(dissimilarity_matrix)))
        # features.append(mds.fit_transform(dissimilarity_matrix))
        
        # # Perform PCA on the features (from MDS) to reduce to 2 components
        
        # Apply KMeans on the derived feature space
        kmeans = KMeans(
                        n_clusters=2, #consonants and vocals
                        random_state=i,
                        n_init='auto'
                        )
        kmeans_labels[i] = kmeans.fit_predict(features)
    kmeans_labels = mode(kmeans_labels, axis=0, keepdims=True).mode.flatten()
    # features = np.mean(features, axis=0)
    # kmeans_labels = [round(el) for el in np.mean(kmeans_labels, axis=0)]
    
    # Compute confusion matrix between k-means clusters and manual labels
    conf_matrix = confusion_matrix(manual_labels, kmeans_labels)

    # Solve the label assignment problem
    row_ind, col_ind = linear_sum_assignment(-conf_matrix)  # Maximize agreement

    # Map k-means labels to manual labels
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    kmeans_labels = np.array([label_mapping[label] for label in kmeans_labels])

    
    # Keep track of mappings
    df = pd.DataFrame({
    'Original_index': np.arange(average_weights.shape[0], dtype=int),
    'Feature_1': features[:, 0],
    'Feature_2': features[:, 1],
    'Manual_label': original_manual_labels,
    'Cluster_Label': kmeans_labels
    })
    
    # Visualization
    plt.figure(figsize=(8, 6))
    # scatter = plt.scatter(df['Feature_1'], df['Feature_2'], c=df['Cluster_Label'], cmap='viridis', s=500, edgecolor='k')
    # plt.colorbar(scatter, label='Cluster')
    for i, row in df.iterrows():
        plt.scatter(row['Feature_1'], row['Feature_2'], c='black', s=1, facecolors='none')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
        color = 'C0' if i in consonants else 'C1'
        plt.text(row['Feature_1'], row['Feature_2'], f"{config.ordered_phonemes[i]}", fontsize=15, ha='right', color=color)
    # for i, row in df.iterrows():
        # plt.scatter(row['Feature_1'], row['Feature_2'], c=f'C{int(row["Cluster_Label"])}')#, label = f"Row {row['Original_Row_Index']}")#, fontsize=9, ha='right')
    legend_handles = [Line2D([0], [0], color=color_labels[name], lw=4, label=name) for name in color_labels]
    plt.legend(handles=legend_handles, title="Phoneme Categories")
    plt.title('KMeans Clustering Results with Row Mapping', fontsize=16)
    plt.xlabel('Feature 1 (MDS)', fontsize=12)
    plt.ylabel('Feature 2 (MDS)', fontsize=12)
    plt.grid(True)
    plt.savefig('figures/prueba.png')

    # Create a 3D scatter plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the first three features (e.g., MDS components)
    scatter = ax.scatter(
        features[:, 0], features[:, 1], features[:, 2], 
        c=kmeans_labels, cmap='viridis', s=100, edgecolor='k'
    )

    # Add annotations for each point (e.g., phoneme labels)
    for i, (x, y, z) in enumerate(zip(features[:, 0], features[:, 1], features[:, 2])):
        ax.text(x, y, z, config.ordered_phonemes[i], fontsize=8, color='black')

    # Set titles and labels
    ax.set_title('3D Scatter Plot of Clustering Results', fontsize=16)
    ax.set_xlabel('Feature 1', fontsize=12)
    ax.set_ylabel('Feature 2', fontsize=12)
    ax.set_zlabel('Feature 3', fontsize=12)

    # Add a colorbar to indicate cluster membership
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label('Cluster Label', fontsize=12)

    # Show the plot
    plt.tight_layout()
    for elev in range(0, 90, 45):
        for azim in range(0, 360, 45):
            ax.view_init(elev=elev, azim=azim)
            plt.savefig(f'figures/3d_scatter_elev{elev}_azim{azim}.png', dpi=300)
        
    # Assume `manual_labels` and `kmeans_labels` are already defined
    # Compute confusion matrix
    conf_matrix = confusion_matrix(manual_labels, kmeans_labels)

    # Compute Precision, Recall, and F-score
    precision = precision_score(manual_labels, kmeans_labels, average='weighted')
    recall = recall_score(manual_labels, kmeans_labels, average='weighted')
    f_score = f1_score(manual_labels, kmeans_labels, average='weighted')
    nmi = normalized_mutual_info_score(manual_labels, kmeans_labels)
    ari = adjusted_rand_score(manual_labels, kmeans_labels)

    # Display results
    f_scores.append(f_score)
    precisions.append(precision)
    nmis.append(nmi)
    aris.append(ari)
    recalls.append(recall)
    
    precision = precision_score(manual_labels_inverted, kmeans_labels, average='weighted')
    recall = recall_score(manual_labels_inverted, kmeans_labels, average='weighted')
    f_score = f1_score(manual_labels_inverted, kmeans_labels, average='weighted')
    nmi = normalized_mutual_info_score(manual_labels_inverted, kmeans_labels)
    ari = adjusted_rand_score(manual_labels_inverted, kmeans_labels)

    # Display results
    f_scores_i.append(f_score)
    precisions_i.append(precision)
    nmis_i.append(nmi)
    aris_i.append(ari)
    recalls_i.append(recall)

step = sample_window/config.eeg_sample_rate

plt.figure(figsize=(8, 6))
plt.scatter(np.arange(3),f_scores, label='F-scores', zorder=1)
plt.plot(np.arange(3),f_scores, zorder=1)
plt.scatter(np.arange(3),f_scores_i, label='F-scores_i', zorder=1)
plt.plot(np.arange(3),f_scores_i, zorder=1)
# plt.scatter((config.times[:-5]+step/2)*1e3, f_scores, label='F-scores', zorder=1)
# plt.plot((config.times[:-5]+step/2)*1e3, f_scores, zorder=1)
# plt.scatter((config.times[:-5]+step/2)*1e3, f_scores_i, label='F-scores_i', zorder=1)
# plt.plot((config.times[:-5]+step/2)*1e3, f_scores_i, zorder=1)
# plt.scatter((config.times[:-5]+step/2)*1e3, aris, label='Adj. Rand Index', zorder=2)
# plt.plot((config.times[:-5]+step/2)*1e3, aris, zorder=2)
# plt.scatter((config.times[:-5]+step/2)*1e3, nmis, label='Normalized Mutual Inf.', zorder=3)
# plt.plot((config.times[:-5]+step/2)*1e3, nmis, zorder=3)
# plt.scatter(config.times[:-5]*1e3, precisions, label='Precision', zorder=2)
# plt.plot(config.times[:-5]*1e3, precisions, zorder=2)
# plt.scatter(config.times[:-5]*1e3, recalls, label='Recall', zorder=3)
# plt.plot(config.times[:-5]*1e3, recalls, zorder=3)
plt.xlabel('Time (ms)')
plt.legend()
plt.grid(visible=True, zorder=0)
plt.savefig('figures/prueba.png')


f_scores_i