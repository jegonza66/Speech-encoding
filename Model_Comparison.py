# Standard libraries
import matplotlib.pyplot as plt, numpy as np, os, mne, pandas as pd, seaborn as sn, copy
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from matplotlib import colormaps

# Specific libraries
# from statsmodels.stats.multitest import fdrcorrection
from matplotlib_venn import venn3, venn2  # venn3_circles
from scipy.spatial import ConvexHull
from scipy.stats import wilcoxon

# Modules
from funciones import load_pickle, cohen_d, all_possible_combinations, get_maximum_correlation_channels
import setup
exp_info = setup.exp_info()

# Default size is 10 pts, the scalings (10pts*scale) are:
#'xx-small':0.579,'x-small':0.694,'s
# mall':0.833,'medium':1.0,'large':1.200,'x-large':1.440,'xx-large':1.728,None:1.0}
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'legend.title_fontsize': 'x-large',
          'figure.figsize': (8, 6),
          'figure.titlesize': 'xx-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'large',
          'ytick.labelsize':'large'}
pylab.rcParams.update(params)

# Model parametrs
model ='mtrf'
situation = 'External'
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
tmin, tmax = -.2, .6
sr = 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)
montage = mne.channels.make_standard_montage('biosemi128')
info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)

# Whether to use or not just relevant channels
relevant_channels = 12 # None

# Relevant paths
path_figures = os.path.normpath(f'figures/{model}/model_comparison/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
final_corr_path = os.path.normpath(f'saves/{model}/{situation}/correlations/tmin{tmin}_tmax{tmax}/')
weights_path = os.path.normpath(f'saves/{model}/{situation}/weights//stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/')
preprocesed_data_path = os.path.normpath(f'saves/preprocessed_data/{situation}/tmin-0.2_tmax0.6/') 

# Code parameters
save_figures = True
# ================================================================================
# CONVEX HULL:  it contrasts the correlation of each part against the joint model.
# ================================================================================
# path_convex_hull = os.path.join(path_figures,'convex_hull')

# # Relevant parameters
# bands = ['Theta']
# stims = 'Deltas_Spectrogram_Phonological'
# stims = '_'.join(sorted(stims.split('_'))) 
# substims = stims.split('_')
# avg_corr, good_ch = {}, {}

# # Iterate over bands
# for band in bands:
#     # Fill dictionaries with data
#     for stim in substims + [stims]:
#         data = load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))
#         if relevant_channels:
#             filter_relevant_channels_filter = get_maximum_correlation_channels(data['average_correlation_subjects'].mean(axis=0), number_of_lat_channels=relevant_channels)
#             good_ch[stim] = data['repeated_good_correlation_channels_subjects'][:,filter_relevant_channels_filter].ravel()
#             avg_corr[stim] = data['average_correlation_subjects'][:,filter_relevant_channels_filter].ravel()
#         else:
#             good_ch[stim] = data['repeated_good_correlation_channels_subjects'].ravel()
#             avg_corr[stim] = data['average_correlation_subjects'].ravel()

#     # Make plot
#     plt.ioff()
#     plt.figure(layout='tight')
#     plt.title(band)

#     for i, stim in enumerate(substims):
#         # Identify Hull (la cáscara de los datos, i.e, el borde)
#         stim_points = np.array([avg_corr[stims], avg_corr[stim]]).transpose()
#         stim_hull = ConvexHull(stim_points)

#         # Plot good channels for each stim
#         plt.plot(avg_corr[stims][good_ch[stim] != 0], 
#                  avg_corr[stim][good_ch[stim] != 0], 
#                  '.', 
#                  color=f'C{i}',
#                  label=stim, 
#                  ms=2.5)

#         # Plot bad channels for each stim
#         plt.plot(avg_corr[stims][good_ch[stim] == 0], 
#                  avg_corr[stim][good_ch[stim] == 0], 
#                  '.', 
#                  color='grey',
#                 alpha=0.5, 
#                 label='Failed permutation test', 
#                 markersize=2)
#         plt.fill(stim_points[stim_hull.vertices, 0], 
#                  stim_points[stim_hull.vertices, 1], 
#                  color=f'C{i}',
#                  alpha=0.3, 
#                  linewidth=0)

#     # Get limits
#     xlimit, ylimit = plt.xlim(), plt.ylim()
#     plt.plot([xlimit[0], 0.7], [xlimit[0], 0.7], 'k--', zorder=0)
#     plt.hlines(0, xlimit[0], xlimit[1], color='grey', linestyle='dashed')
#     plt.vlines(0, ylimit[0], ylimit[1], color='grey', linestyle='dashed')
    
#     plt.ylabel('Individual model (r)')
#     plt.xlabel('Full model (r)')

#     # Legend
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = dict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys(), markerscale=3)

#     # Save
#     if save_figures:
#         temp_path = os.path.join(path_convex_hull, f'{band}')
#         os.makedirs(temp_path, exist_ok=True)
#         if relevant_channels:
#             plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{stims}.png'))
#             plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{stims}.svg'))
#         else:
#             plt.savefig(os.path.join(temp_path, f'{stims}.png'))
#             plt.savefig(os.path.join(temp_path, f'{stims}.svg'))
#     plt.close()

# # =========================================================================================================
# # VENN DIAGRAMS: it makes diagrams explaining correlation of each part of a shared model (upto 3 features).
# # =========================================================================================================
# path_venn_diagrams = os.path.join(path_figures,'venn_diagrams')

# # Relevant parameters
# bands = ['Theta']
# stimuli = ['Phonological', 'Spectrogram', 'Deltas']

# # Arrange shared stimuli
# stimuli = sorted(stimuli)
# double_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==2]
# triple_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==3]
# all_stimuli = stimuli + double_combinations + triple_combinations
# mean_correlations = {}

# # Iterate over bands
# for band in bands:
#     for stim in all_stimuli:
#         # Get average correlation of each stimulus
#         data = load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))['average_correlation_subjects']
#         if relevant_channels:
#             filter_relevant_channels_filter = get_maximum_correlation_channels(data.mean(axis=0), number_of_lat_channels=relevant_channels)
#             mean_correlations[stim] = data[:,filter_relevant_channels_filter].mean()
#         else:
#             mean_correlations[stim] = data.mean()

#     for stim12 in double_combinations:
#         stim1, stim2 = stim12.split('_')

#         # Get squared correlation of stimuli
#         variance_1 = mean_correlations[stim1] ** 2
#         variance_2 = mean_correlations[stim2] ** 2
#         variance_12 = mean_correlations[stim12] ** 2 # this represent the union of the two

#         # This represent the shared variances explained by the intersections of sets 1 and 2 (intersection between 1 and 2)
#         variance_intersection_12 = variance_1 + variance_2 - variance_12 #11

#         # This is the realtive complemente of 1 and 2: portion of the variance solely explained by 1 and 2, respectively
#         variance_explained_by_1 = variance_12 - variance_2 #10
#         variance_explained_by_2 = variance_12 - variance_1 #01
        
#         # Get list with areas
#         areas = [variance_explained_by_1, variance_explained_by_2, variance_intersection_12] # note that the sum gives shared model
#         areas = [0 if area<0 else area.round(3) for area in areas]
        
#         # Create figure and title
#         title = stim12.replace('_', ' and ')
#         plt.ioff()
#         plt.figure(layout='tight')
#         plt.title(f'Diagram {band} band - {title}')

#         # Make plot
#         venn2(subsets=areas, # left area diagran, right area diagram, shared area <--> (10, 01, 11)
#               set_labels=(stim1, stim2),
#               set_colors=('C0', 'C1'), 
#               alpha=0.45)

#         # Save figure
#         if save_figures:
#             temp_path = os.path.join(path_venn_diagrams, f'{band}')
#             os.makedirs(temp_path, exist_ok=True)
#             if relevant_channels:
#                 plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{stim12}.png'))
#                 plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{stim12}.svg'))
#             else:
#                 plt.savefig(os.path.join(temp_path, f'{stim12}.png'))
#                 plt.savefig(os.path.join(temp_path, f'{stim12}.svg'))
#         plt.close()
        
#         # Make a print with information
#         stim1_percent = (variance_explained_by_1 * 100 /np.sum(areas)).round(2)
#         stim2_percent = (variance_explained_by_2 * 100 /np.sum(areas)).round(2)
#         shared_percent = (variance_intersection_12 * 100 /np.sum(areas)).round(2)
#         print(f'\nPercentage explained by {stim1} is {stim1_percent} %',
#               f'\nPercentage explained by {stim2} is {stim2_percent} %',
#               f'\nShared percentage explained is {shared_percent} %')
#         #TODO Y la variancia no explicada?

#     if triple_combinations:
#         # Get squared correlation of stimuli
#         variance_1 = mean_correlations[all_stimuli[0]]**2
#         variance_2 = mean_correlations[all_stimuli[1]]**2
#         variance_3 = mean_correlations[all_stimuli[2]]**2
#         variance_12 = mean_correlations[all_stimuli[3]]**2
#         variance_13 = mean_correlations[all_stimuli[4]]**2
#         variance_23 = mean_correlations[all_stimuli[5]]**2
#         variance_123 = mean_correlations[all_stimuli[6]]**2

#         # Shared without each stimulus
#         variance_shared_with_1 = variance_123 - variance_23 #100
#         variance_shared_with_2 = variance_123 - variance_13 #010
#         variance_shared_with_3 = variance_123 - variance_12 #001
        
#         # Explained by subshared, but not by all shared model
#         variance_shared_with_12 = variance_13 + variance_23 - variance_3 - variance_123 #110
#         variance_shared_with_13 = variance_12 + variance_23 - variance_2 - variance_123 #101
#         variance_shared_with_23 = variance_12 + variance_13 - variance_1 - variance_123 #011

#         # Explained by one, two, three and full shared model but not by subshared models
#         variance_int_complement_submodels = variance_123 + variance_1 + variance_2 + variance_3 - variance_12 - variance_13 - variance_23 #111

#         areas = [variance_shared_with_1, variance_shared_with_2, variance_shared_with_3, \
#                  variance_shared_with_12, variance_shared_with_13, variance_shared_with_23,\
#                  variance_int_complement_submodels] 
#         areas = [0 if area<0 else area.round(3) for area in areas] # note that the sum gives shared model variance_123

#         # Create figure and title
#         title = all_stimuli[-1].replace('_', ', ')
#         plt.ioff()
#         plt.figure(layout='tight')
#         plt.title(f'Diagram {band} band - {title}')

#         # Make plot
#         venn3(subsets=areas, # left area diagran, right area diagram, shared area <--> (100, 010, 110, 001, 101, 011, 111).
#               set_labels=(all_stimuli[0], all_stimuli[1], all_stimuli[2]), 
#               set_colors=('C0', 'C1', 'purple'), 
#               alpha=0.45)

#         if save_figures:
#             temp_path = os.path.join(path_venn_diagrams, f'{band}')
#             os.makedirs(temp_path, exist_ok=True)
#             if relevant_channels:
#                 plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{all_stimuli[-1]}.png'))
#                 plt.savefig(os.path.join(temp_path, f'relevant_channels_{relevant_channels}_{all_stimuli[-1]}.svg'))
#             else:
#                 plt.savefig(os.path.join(temp_path, f'{all_stimuli[-1]}.png'))
#                 plt.savefig(os.path.join(temp_path, f'{all_stimuli[-1]}.svg'))
#         plt.close()

#         # # Make a print with information
#         # stim1_percent = (variance_shared_with_1 * 100 /np.sum(areas)).round(2)
#         # stim2_percent = (variance_shared_with_2 * 100 /np.sum(areas)).round(2)
#         # stim3_percent = (variance_shared_with_2 * 100 /np.sum(areas)).round(2)
#         # shared_percent = (variance_complement_12 * 100 /np.sum(areas)).round(2)
#         # print(f'\nExclusive Percentage explained by {stim1} is {stim1_percent} %',
#         #       f'\nExclusive Percentage explained by {stim2} is {stim2_percent} %',
#         #       f'\nshared percentage explained is {shared_percent} %')


# # Relevant parameters
# bands = ['Theta']
# stimuli = ['Phonological', 'Spectrogram', 'Deltas']

# # Arrange shared stimuli
# stimuli = sorted(stimuli)
# double_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==2]
# triple_combinations = ['_'.join(sorted(combination)) for combination in all_possible_combinations(stimuli) if len(combination)==3]
# all_stimuli = stimuli + double_combinations + triple_combinations
# mean_correlations = {}

# # Iterate over bands
# for band in bands:
#     for stim in all_stimuli:
#         # Get average correlation of each stimulus
#         data = load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))['average_correlation_subjects']
#         if relevant_channels:
#             filter_relevant_channels_filter = get_maximum_correlation_channels(data.mean(axis=0), number_of_lat_channels=relevant_channels)
#             mean_correlations[stim] = data[:,filter_relevant_channels_filter].mean()
#         else:
#             mean_correlations[stim] = data.mean()

# ================================================================================================================
# CORRELATION MATRIX TOPOMAP: make matrix with correlation topomap ordering across features, situations and bands.
# ================================================================================================================
path_correlation_matrix_topo = os.path.join(path_figures,'correlation_matrix_topo')

# Relevant parameters
bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']
stimuli = ['Pitch-Log-Raw', 'Envelope', 'Spectrogram', 'Mfccs', 'Phonemes-Discrete-Phonet', 'Phonological'] # 'Envelope_Phonemes-Discrete-Manual'
n_stims, n_bands = len(stimuli), len(bands)

# Get mean correlations across subjects and total max and min
correlations = {(stim,band):load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))['average_correlation_subjects'].mean(axis=0) for stim in stimuli for band in bands}
minimum_cor, maximum_cor = min([correlation.min() for correlation in correlations.values()]), max([correlation.max() for correlation in correlations.values()])

# Create figure and title
fig, axes = plt.subplots(
                        # figsize=(3*n_stims,1.5*n_bands), 
                        nrows=n_bands, 
                        ncols=n_stims, 
                        layout="constrained"
                        )
fig.suptitle('Correlation topomaps')
if n_stims==1:
    axes = axes.reshape(n_bands, 1)
elif n_bands==1:
    axes = axes.reshape(1, n_stims)

# Configure axis
for ax, col in zip(axes[0], stimuli):
    if col=='Phonemes-Discrete-Phonet':
        col = 'Phonemes'
    elif col=='Pitch-Log-Raw':
        col = 'Pitch-Log'
    ax.set_title(col, fontsize=12)
for ax, row in zip(axes[:,0], bands):
    ax.set_ylabel(row, rotation=90, fontsize=12)

# Build scale
# normalizer = LogNorm(vmin=np.round(minimum_cor,2), vmax=np.round(maximum_cor,2))
normalizer = Normalize(vmin=np.round(minimum_cor,2), vmax=np.round(maximum_cor,2))
im = cm.ScalarMappable(norm=normalizer, cmap='Reds')

# Iterate over bands
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        # Get average correlation of each stimulus across subjects
        average_correlation = correlations[(stim,band)]

        # Plot topomap        
        mne.viz.plot_topomap(
            data=average_correlation, 
            pos=info_mne, 
            axes=axes[i, j], 
            show=False, 
            sphere=0.07, 
            cmap='Reds', 
            # vlim=(minimum_cor, maximum_cor),
            cnorm=normalizer
            )

# Make colorbar
fig.colorbar(im, ax=axes.ravel().tolist())
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_correlation_matrix_topo,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'correlation_matrix_topo.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'correlation_matrix_topo.svg'))
plt.close()

# ==============================================================================================================
# SIMILARITY MATRIX TOPOMAP: make matrix with similarity topomap ordering across features, situations and bands.
# ==============================================================================================================
path_similarities_matrix_topo = os.path.join(path_figures,'similarities_matrix_topo')

# Relevant parameters
# Relevant parameters
bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']
stimuli = ['Pitch-Log-Raw', 'Envelope', 'Spectrogram', 'Mfccs', 'Phonemes-Discrete-Phonet', 'Phonological'] # 'Envelope_Phonemes-Discrete-Manual'
n_stims, n_bands = len(stimuli), len(bands)

# Get similarities across subjects and total max and min
similarities = {}
for band in bands:
    for stim in stimuli:
        average_weights_subjects = load_pickle(path=os.path.join(weights_path, band, stim, 'total_weights_per_subject.pkl'))['average_weights_subjects']
        n_subjects, n_chan, _, _ = average_weights_subjects.shape
        average_weights = average_weights_subjects.mean(axis=2)
        correlation_matrices = np.zeros(shape=(n_chan, n_subjects, n_subjects))

        # Calculate correlation betweem subjects
        for channel in range(n_chan):
            matrix = average_weights[:,channel,:] # TODO HAVE ONE MORE DIMENSION
            correlation_matrices[channel] = np.corrcoef(matrix)

        # Correlacion por canal
        absolute_correlation_per_channel = np.zeros(n_chan)
        for channel in range(n_chan):
            channel_corr_values = correlation_matrices[channel][np.tril_indices(n_subjects, k=-1)]
            absolute_correlation_per_channel[channel] = np.mean(np.abs(channel_corr_values))
        
        # Append to similarities
        similarities[(stim, band)] = absolute_correlation_per_channel
minimum_sim, maximum_sim = min([similarity.min() for similarity in similarities.values()]), max([similarity.max() for similarity in similarities.values()])

# Create figure and title
fig, axes = plt.subplots(
                        nrows=n_bands, 
                        ncols=n_stims, 
                        layout="constrained"
                        )   
fig.suptitle('Similarity topomaps')
if n_stims==1:
    axes = axes.reshape(n_bands, 1)
elif n_bands==1:
    axes = axes.reshape(1, n_stims)

# Set axis labels
for ax, col in zip(axes[0], stimuli):
    if col=='Phonemes-Discrete-Phonet':
        col = 'Phonemes'
    elif col=='Pitch-Log-Raw':
        col = 'Pitch-Log'
    ax.set_title(col, fontsize=12)
for ax, row in zip(axes[:,0], bands):
    ax.set_ylabel(row, rotation=90, fontsize=12)

# Build scale
normalizer = Normalize(vmin=np.round(minimum_sim,2), vmax=np.round(maximum_sim,2))
im = cm.ScalarMappable(norm=normalizer, cmap='Greens')

# Iterate over bands
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        # Get average correlation of each stimulus across subjects
        similarity = similarities[(stim,band)]

        # Plot topomap        
        mne.viz.plot_topomap(
            data=similarity, 
            pos=info_mne, 
            axes=axes[i, j], 
            show=False, 
            sphere=0.07, 
            cmap='Greens', 
            # vlim=(similarity.min(), similarity.max()),
            cnorm=normalizer
            )

# Add colorbar
fig.colorbar(im, ax=axes.ravel().tolist())
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_similarities_matrix_topo,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'similarities_matrix_topo.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'similarities_matrix_topo.svg'))
plt.close()

# ===================================================================================================================
# TOPOGRAPHIC DISTRIBUTION HEATMAPS: make heatmaps with topographic information across features, situations and bands
# ===================================================================================================================
path_distribution = os.path.join(path_figures,'distribution__heatmaps')

# Relevant parameters
bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']
stimuli = ['Pitch-Log-Raw', 'Envelope', 'Spectrogram', 'Mfccs', 'Phonemes-Discrete-Phonet', 'Phonological'] # 'Envelope_Phonemes-Discrete-Manual'
n_stims, n_bands = len(stimuli), len(bands)

# Get mean correlations across subjects and total max and min
correlations = {(stim,band):load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))['average_correlation_subjects'].mean(axis=0) for stim in stimuli for band in bands}
minimum_cor, maximum_cor = min([correlation.min() for correlation in correlations.values()]), max([correlation.max() for correlation in correlations.values()])

# Get groups to make average correlations
n_groups_x, n_groups_y = 12, 12
posicion_x = np.array([montage._get_ch_pos()[ch][0] for ch in montage._get_ch_pos()])
posicion_y = np.array([montage._get_ch_pos()[ch][1] for ch in montage._get_ch_pos()])
bins_x = np.linspace(posicion_x.min(), posicion_x.max(), n_groups_x)
bins_y = np.linspace(posicion_y.min(), posicion_y.max(), n_groups_y)
groups_x, groups_y = [], []
for i in range(1, n_groups_y):
    group_y = []
    for ch in montage._get_ch_pos():
        y_value = montage._get_ch_pos()[ch][1]
        if bins_y[i-1]<=y_value<=bins_y[i]:
            group_y.append(info_mne.ch_names.index(ch))
    groups_y.append(group_y)
for i in range(1, n_groups_x):
    group_x = []
    for ch in montage._get_ch_pos():
        x_value = montage._get_ch_pos()[ch][0]
        y_value = montage._get_ch_pos()[ch][1]
        if (bins_x[i-1]<=x_value<=bins_x[i]) and (y_value>=0) :
            group_x.append(info_mne.ch_names.index(ch))
    groups_x.append(group_x)
# Para ver si está balanceado
plt.figure()
plt.title('bins_x')
plt.hist(posicion_x, bins_x)
plt.grid(visible=True)
plt.yticks(ticks=np.arange(0,20))
plt.show(block=False)
plt.figure()
plt.title('bins_y')
plt.hist(posicion_y, bins_y)
plt.grid(visible=True)
plt.yticks(ticks=np.arange(0,20))
plt.show(block=False)

# =======================
# HEATMAP LATERALIZATION
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout='True')
axes[0].set_title('Correlation: Phonological-Alpha',
                   y=1.05, 
                   verticalalignment="top")
axes[0].grid(visible=True)
axes[0].set_axisbelow(True)
# axes[0].scatter(np.arange(n_groups_y-1), [correlations[('Phonological','Theta')][group].mean() for group in groups_y], color='black')
cmap = colormaps['plasma']
colors = cmap(np.linspace(0, 1, len(groups_x)))
for i, corr, c in zip(np.arange(n_groups_x-1), [correlations[('Phonological','Alpha')][group].mean() for group in groups_x], colors):
    axes[0].scatter(i, corr, color=c)
axes[0].set_xticks(np.arange(n_groups_x-1))
axes[0].set_xticklabels([f'G{i}'for i in np.arange(n_groups_x-1)])
axes[0].set_xlabel('Channel group selection')
axes[0].set_ylabel('Average correlation')
subax = axes[0].inset_axes([.05,.07,.6,.6])
mne.viz.plot_sensors(
                    info=info_mne,
                    show_names=False,
                    block=False,
                    pointsize=35,
                    cmap='plasma',
                    axes=subax,
                    ch_groups=groups_x
                    )
# # Create a legend
# cmap = colormaps['plasma']
# colors = cmap(np.linspace(0, 1, len(groups_x)))
# handles = [mpatches.Patch(color=colour, label=f'G{i}') for i, colour in enumerate(colors)]
# axes[0].legend(handles=handles, loc='lower right', frameon=True, fontsize=10)
z = np.zeros(shape=(len(bands),len(stimuli)))
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        average_correlation = correlations[(stim,band)]
        left_group, right_group = [], []
        for l in range(n_groups_x-1):
            if l<4:
                for ch in groups_x[l]:
                    left_group.append(ch)
            elif l>6:
                for ch in groups_x[l]:
                    right_group.append(ch)
        z[i,j] = average_correlation[right_group].mean()-average_correlation[left_group].mean()
axes[1].set_title('Lateralization: right(G[0-3])-left(G[7-10])')
im = axes[1].imshow(
                z, 
                vmin=z.min(), 
                vmax=z.max(),
                cmap='plasma'
                )
axes[1].set_xticks(np.arange(len(stimuli)))
axes[1].set_xticklabels([stim.split('-')[0] if stim!='Pitch-Log-Raw' else 'Pitch-Log' for stim in stimuli], minor=False, fontsize=12, rotation=35)
axes[1].set_yticks(np.arange(len(bands)))
axes[1].set_yticklabels(bands, minor=False, fontsize=12)
# axes[1].xaxis.tick_top()
fig.colorbar(im, ax=axes[1], label='Correlation difference')
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_distribution,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'lateralization.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'lateralization.svg'))
plt.close()

# =======================
# HEATMAP LATERALIZATION
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout='True')
axes[0].set_title('Correlation: Phonemes-Theta',
                   y=1.05, 
                   verticalalignment="top")
axes[0].grid(visible=True)
axes[0].set_axisbelow(True)
# axes[0].scatter(np.arange(n_groups_y-1), [correlations[('Phonemes-Discrete-Phonet','Theta')][group].mean() for group in groups_y], color='black')
cmap = colormaps['hsv']
colors = cmap(np.linspace(0, 1, len(groups_x)))
for i, corr, c in zip(np.arange(n_groups_x-1), [correlations[('Phonemes-Discrete-Phonet','Theta')][group].mean() for group in groups_x], colors):
    axes[0].scatter(i, corr, color=c)
axes[0].set_xticks(np.arange(n_groups_x-1))
axes[0].set_xticklabels([f'G{i}'for i in np.arange(n_groups_x-1)])
axes[0].set_xlabel('Channel group selection')
axes[0].set_ylabel('Average correlation')
subax = axes[0].inset_axes([.02,.07,.45,.45])
mne.viz.plot_sensors(
                    info=info_mne,
                    show_names=False,
                    block=False,
                    pointsize=20,
                    cmap='hsv',
                    axes=subax,
                    ch_groups=groups_x
                    )
# # Create a legend
# cmap = colormaps['plasma']
# colors = cmap(np.linspace(0, 1, len(groups_x)))
# handles = [mpatches.Patch(color=colour, label=f'G{i}') for i, colour in enumerate(colors)]
# axes[0].legend(handles=handles, loc='lower right', frameon=True, fontsize=10)
z = np.zeros(shape=(len(bands),len(stimuli)))
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        average_correlation = correlations[(stim,band)]
        sides_group, center_group = [], []
        for l in range(n_groups_x-1):
            if 4<=l<=6:
                for ch in groups_x[l]:
                    center_group.append(ch)
            else:
                for ch in groups_x[l]:
                    sides_group.append(ch)
        z[i,j] = average_correlation[sides_group].mean()-average_correlation[center_group].mean()
axes[1].set_title('Centralization: sides(G[0-3&7-10])-center(G[4-6])')
im = axes[1].imshow(
                z, 
                vmin=z.min(), 
                vmax=z.max(),
                cmap='plasma'
                )
axes[1].set_xticks(np.arange(len(stimuli)))
axes[1].set_xticklabels([stim.split('-')[0] if stim!='Pitch-Log-Raw' else 'Pitch-Log' for stim in stimuli], minor=False, fontsize=12, rotation=35)
axes[1].set_yticks(np.arange(len(bands)))
axes[1].set_yticklabels(bands, minor=False, fontsize=12)
# axes[1].xaxis.tick_top()
fig.colorbar(im, ax=axes[1], label='Correlation difference')
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_distribution,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'centralization.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'centralization.svg'))
plt.close()

# ==========================
# HEATMAP ANTERIOR-POSTERIOR
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,4), constrained_layout='True')
axes[0].set_title('Correlation: Phonological-Theta',
                   y=1.05, 
                   verticalalignment="top")
axes[0].grid(visible=True)
axes[0].set_axisbelow(True)
# axes[0].scatter(np.arange(n_groups_y-1), [correlations[('Phonological','Theta')][group].mean() for group in groups_y], color='black')
cmap = colormaps['cividis']
colors = cmap(np.linspace(0, 1, len(groups_y)))
for i, corr, c in zip(np.arange(n_groups_y-1), [correlations[('Phonological','Theta')][group].mean() for group in groups_y], colors):
    axes[0].scatter(i, corr, color=c)
axes[0].set_xticks(np.arange(n_groups_y-1))
axes[0].set_xticklabels([f'G{i}'for i in np.arange(n_groups_y-1)])
axes[0].set_xlabel('Channel group selection')
axes[0].set_ylabel('Average correlation')
subax = axes[0].inset_axes([.35,.1,.7,.7])
mne.viz.plot_sensors(
                    info=info_mne,
                    show_names=False,
                    block=False,
                    pointsize=35,
                    cmap='cividis',
                    axes=subax,
                    ch_groups=groups_y
                    )
# # Create a legend
# cmap = colormaps['cividis']
# colors = cmap(np.linspace(0, 1, len(groups_y)))
# handles = [mpatches.Patch(color=colour, label=f'G{i}') for i, colour in enumerate(colors)]
# axes[0].legend(handles=handles, loc='lower right', frameon=True, fontsize=10)
z = np.zeros(shape=(len(bands),len(stimuli)))
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        average_correlation = correlations[(stim,band)]
        z[i,j] = np.corrcoef([average_correlation[group].mean() for group in groups_y], np.arange(n_groups_y-1))[1,0]

axes[1].set_title('Anterior-Posterior correlation')
im = axes[1].imshow(
                z, 
                vmin=z.min(), 
                vmax=z.max(),
                cmap='plasma'
                )
axes[1].set_xticks(np.arange(len(stimuli)))
axes[1].set_xticklabels([stim.split('-')[0] if stim!='Pitch-Log-Raw' else 'Pitch-Log' for stim in stimuli], minor=False, fontsize=12, rotation=35)
axes[1].set_yticks(np.arange(len(bands)))
axes[1].set_yticklabels(bands, minor=False, fontsize=12)
# axes[1].xaxis.tick_top()
fig.colorbar(im, ax=axes[1], label=r'Linear correlation')
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_distribution,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'anterior_posterior.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'anterior_posterior.svg'))
plt.close()

# =================================================================================================
# CORRELATION HEATMAPS: make heatmaps with correlation values across features, situations and bands
# =================================================================================================
path_correlation_heatmaps = os.path.join(path_figures,'correlation_heatmaps')

# Relevant parameters
bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']
stimuli = ['Pitch-Log-Raw', 'Envelope', 'Spectrogram', 'Mfccs', 'Phonemes-Discrete-Phonet', 'Phonological'] # 'Envelope_Phonemes-Discrete-Manual'
n_stims, n_bands = len(stimuli), len(bands)

# Get mean correlations across subjects and total max and min
correlations = {(stim,band):load_pickle(path=os.path.join(final_corr_path, band, stim +'.pkl'))['average_correlation_subjects'].mean(axis=0) for stim in stimuli for band in bands}
minimum_cor, maximum_cor = min([correlation.min() for correlation in correlations.values()]), max([correlation.max() for correlation in correlations.values()])

z = np.zeros(shape=(len(bands),len(stimuli)))
for i, band in enumerate(bands):
    for j, stim in enumerate(stimuli):
        z[i,j] = correlations[(stim,band)].mean()

fig = plt.figure(constrained_layout=True)
ax = fig.gca()
# plt.title('External')
im = plt.imshow(
                z, 
                vmin=z.min(), 
                vmax=z.max(),
                cmap='plasma'
                )
ax.set_xticks(np.arange(len(stimuli)))
ax.set_xticklabels([stim.split('-')[0] if stim!='Pitch-Log-Raw' else 'Pitch-Log' for stim in stimuli], minor=False, fontsize=12, rotation=35)
ax.set_yticks(np.arange(len(bands)))
ax.set_yticklabels(bands, minor=False, fontsize=12)
ax.xaxis.tick_top()
fig.colorbar(im, ax=ax, label=r'Average correlation')
fig.show()

# Save figure
if save_figures:
    short_stimuli = [stimulus.split('-')[0] if stimulus!='Pitch-Log-Raw' else 'Pitch-Log' for stimulus in stimuli]
    temp_path = os.path.join(path_correlation_heatmaps,'_'.join(sorted(short_stimuli)))
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'heatmap_corr.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'heatmap_corr.svg'))
plt.close()

# # ===================
# # VIOLIN/TOPO (BANDS)
# # ===================
# path_violin = os.path.join(path_figures,'violin_topo')

# # Relevant parameters
# bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']
# stimulus = 'Spectrogram'
# avg_corr = {} 

# # Make figure
# fig, axs = plt.subplots(figsize=(14, 5), ncols=5, nrows=2, layout='tight')
# plt.suptitle(situation) #TODO PONER CON FIG ARRIBA CREO

# # Iterate over bands
# for i, band in enumerate(bands):
#     info = load_pickle(path=os.path.join(preprocesed_data_path, band, 'EEG/info.pkl'))  #TODO check path
#     data = load_pickle(path=os.path.join(final_corr_path, band, stimulus +'.pkl'))
#     avg_corr[band] = data['average_correlation_subjects'].mean(axis=0) # TODO CHECK mean if correct axis
    
#     # Make topomap
#     im = mne.viz.plot_topomap(avg_corr[band].ravel(), 
#                               info, 
#                               axes=axs[0, i], 
#                               show=False, 
#                               sphere=0.07, 
#                               cmap='Reds',
#                               vmin=avg_corr[band].min(), 
#                               vmax=avg_corr[band].max())
#     # Add colorbar
#     cbar = plt.colorbar(im[0], ax=axs[0, i], orientation='vertical', shrink=0.5)

# # Remove axis that aren't used
# for ax_row in axs[1:]:
#     for ax in ax_row:
#         ax.remove()

# # Make violin plots
# ax = fig.add_subplot(2, 1, (2, 3))
# sn.violinplot(data=pd.DataFrame(avg_corr), 
#               palette={band: 'darkgrey' for band in bands}, 
#               ax=ax)

# # Set configuration of labels
# ax.set(ylabel='Correlation', 
#        xticklabels=['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'],
#        ylim=[0, 0.5])
# ax.grid(visible=True)

# # Save figure
# if save_figures:
#     os.makedirs(path_violin, exist_ok=True)
#     plt.savefig(os.path.join(path_violin, f'{stimulus}.png'))
#     plt.savefig(os.path.join(path_violin, f'{stimulus}.svg'))
# plt.close()

# # # ===================
# # # VIOLIN/MTRF (BANDS)

# # info_path = 'saves/preprocessed_data/tmin-0.6_tmax-0.003/EEG/info.pkl'
# # f = open(info_path, 'rb')
# # info = pickle.load(f)
# # f.close()

# # tmin_corr, tmax_corr = -0.6, -0.003
# # tmin_w, tmax_w = -0.6, 0.2
# # delays = - np.arange(np.floor(tmin_w * info['sfreq']), np.ceil(tmax_w * info['sfreq']), dtype=int)
# # times = np.linspace(delays[0] * np.sign(tmin_w) * 1 / info['sfreq'], np.abs(delays[-1]) * np.sign(tmax_w) * 1 / info['sfreq'], len(delays))
# # times = np.flip(-times)

# # model = 'Ridge'
# # situacion = 'Internal_BS'

# # Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin mTRF/'.format(model, situacion, tmin_corr, tmax_corr)
# # Save_fig = True
# # Correlaciones = {}
# # mTRFs = {}

# # stim = 'Spectrogram'
# # Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']

# # for Band in Bands:
# #     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin_corr, tmax_corr, stim, Band), 'rb')
# #     Corr, Pass = pickle.load(f)
# #     f.close()
# #     Correlaciones[Band] = Corr.mean(0)

# # f = open('saves/{}/{}/Original/Stims_Normalize_EEG_Standarize/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.format(model, situacion, tmin_w, tmax_w, stim, 'Theta', stim, 'Theta'), 'rb')
# # mTRFs_Theta = pickle.load(f)
# # f.close()

# # my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta1': 'darkgrey'}

# # fontsize = 17
# # plt.rcParams.update({'font.size': fontsize})
# # fig, axs = plt.subplots(figsize=(8, 5), nrows=2, gridspec_kw={'height_ratios': [1, 1]})

# # plt.suptitle(situacion)
# # sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=axs[0])
# # axs[0].set_ylabel('Correlation')
# # axs[0].set_ylim([-0.1, 0.5])
# # axs[0].grid()
# # axs[0].set_xticklabels(['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'])


# # spectrogram_weights_chanels = mTRFs_Theta.reshape(info['nchan'], 16, len(times)).mean(1)

# # # Adapt for ERP
# # spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

# # evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
# # evoked.times = times
# # evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
# #             show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[1])
# # # axs[1].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
# # axs[1].set_ylim([-0.016, 0.013])
# # if times[0] < 0:
# #     # ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
# #     axs[1].axvline(x=0, ymin=0, ymax=1, color='grey')

# # # axs[1].grid()

# # fig.tight_layout()
# # plt.show()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + '{}.png'.format(stim))
# #     plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

# # # =======================
# # # WILCOXON TEST TOPOPLOTS
# # import numpy as np
# # import pickle
# # import matplotlib.pyplot as plt
# # import os
# # from scipy.stats import wilcoxon
# # import mne
# # # from statsmodels.stats.multitest import fdrcorrection

# # model = 'Ridge'
# # Band = 'Theta'
# # stim = 'Spectrogram'
# # situaciones = ['External', 'Internal', 'Ambos', 'Internal_BS', 'Silencio']
# # tmin, tmax = -0.6, -0.003
# # stat_test = 'log'  # fdr/log/cohen
# # mask = False
# # Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/{}'.format(Band, stim, tmin, tmax, stat_test)
# # if mask and stat_test != 'cohen':
# #     Run_graficos_path += '_mask/'
# # else:
# #     Run_graficos_path += '/'

# # Save_fig = True
# # Display_fig = False
# # if Display_fig:
# #     plt.ion()
# # else:
# #     plt.ioff()

# # montage = mne.channels.make_standard_montage('biosemi128')
# # channel_names = montage.ch_names
# # info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# # Correlaciones = {}

# # for situacion in situaciones:
# #     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
# #     Corr, Pass = pickle.load(f)
# #     f.close()

# #     Correlaciones[situacion] = Corr.transpose()

# # # Calculate wilcoxon test over rows of the dictionary
# # stats = {}
# # pvals = {}

# # for sit1, sit2 in zip(('External', 'External', 'External', 'External', 'Internal', 'Internal', 'Ambos', 'Ambos', 'Ambos', 'Internal_BS'),
# #                       ('Internal', 'Ambos', 'Silencio', 'Internal_BS', 'Silencio', 'Internal_BS', 'Internal_BS', 'Silencio', 'Internal', 'Silencio')):
# #     print(sit1, sit2)

# #     dist1 = Correlaciones[sit1]
# #     dist2 = Correlaciones[sit2]

# #     stat = []
# #     pval = []
# #     for i in range(len(dist1)):
# #         res = wilcoxon(dist1[i], dist2[i])
# #         stat.append(res[0])
# #         pval.append(res[1])

# #     stats[f'{sit1}-{sit2}'] = stat
# #     pvals[f'{sit1}-{sit2}'] = pval

# #     if stat_test == 'fdr':
# #         # passed, corrected_pval = fdrcorrection(pvals=pval, alpha=0.05, method='p')

# #         if mask:
# #             corrected_pval[~passed] = 1

# #         # Plot pvalue
# #         fig, ax = plt.subplots()
# #         fig.suptitle(f'FDR p-value: {sit1}-{sit2}\n'
# #                      f'mean: {round(np.mean(corrected_pval), 6)} - '
# #                      f'min: {round(np.min(corrected_pval), 6)} - '
# #                      f'max: {round(np.max(corrected_pval), 6)}\n'
# #                      f'passed: {sum(passed)}', fontsize=17)
# #         im = mne.viz.plot_topomap(corrected_pval, vmin=0, vmax=1, pos=info, axes=ax, show=Display_fig, sphere=0.07,
# #                                   cmap='Reds_r')
# #         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
# #         cbar.ax.yaxis.set_tick_params(labelsize=17)
# #         cbar.ax.set_ylabel(ylabel='FDR corrected p-value', fontsize=17)

# #         fig.tight_layout()

# #         if Save_fig:
# #             os.makedirs(Run_graficos_path, exist_ok=True)
# #             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
# #             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

# #     elif stat_test == 'log':
# #         log_pval = np.log10(pval)
# #         if mask:
# #             log_pval[np.array(pval) > 0.05/128] = 0

# #         # Plot pvalue
# #         fig, ax = plt.subplots()
# #         fig.suptitle(f'p-value: {sit1}-{sit2}\n'
# #                      f'mean: {round(np.mean(pvals[f"{sit1}-{sit2}"]), 6)} - '
# #                      f'min: {round(np.min(pvals[f"{sit1}-{sit2}"]), 6)} - '
# #                      f'max: {round(np.max(pvals[f"{sit1}-{sit2}"]), 6)}\n'
# #                      f'passed: {sum(np.array(pval)< 0.05/128)}', fontsize=17)
# #         im = mne.viz.plot_topomap(log_pval, vmin=-6, vmax=-2, pos=info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds_r')
# #         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85, ticks=[-6, -5, -4, -3, -2])
# #         cbar.ax.yaxis.set_tick_params(labelsize=17)
# #         cbar.ax.set_yticklabels(['<10-6', '10-5', '10-4', '10-3', '10-2'])
# #         cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)


# #         fig.tight_layout()

# #         if Save_fig:
# #             os.makedirs(Run_graficos_path, exist_ok=True)
# #             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
# #             plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

# #         # Plot statistic
# #         fig, ax = plt.subplots()
# #         fig.suptitle(f'stat: {sit1}-{sit2}\n'
# #                      f'Mean: {round(np.mean(stats[f"{sit1}-{sit2}"]), 3)}', fontsize=19)
# #         im = mne.viz.plot_topomap(stats[f'{sit1}-{sit2}'], info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds')
# #         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
# #         cbar.ax.yaxis.set_tick_params(labelsize=17)
# #         cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)
# #         fig.tight_layout()

# #         if Save_fig:
# #             os.makedirs(Run_graficos_path, exist_ok=True)
# #             plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.png'.format(Band))
# #             plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.svg'.format(Band))

# #     elif stat_test == 'cohen':
# #         cohen_ds = []
# #         for i in range(len(dist1)):
# #             cohen_ds.append(cohen_d(dist1[i], dist2[i]))

# #         # Plot pvalue
# #         fig, ax = plt.subplots()
# #         fig.suptitle(f'Cohen d: {sit1}-{sit2}\n'
# #                      f'mean: {round(np.mean(cohen_ds), 2)} +- {round(np.std(cohen_ds), 2)} -\n'
# #                      f'min: {round(np.min(cohen_ds), 2)} - '
# #                      f'max: {round(np.max(cohen_ds), 2)}', fontsize=17)
# #         im = mne.viz.plot_topomap(cohen_ds, vmin=0, vmax=4, pos=info, axes=ax, show=Display_fig, sphere=0.07,
# #                                   cmap='Reds')
# #         cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
# #         cbar.ax.yaxis.set_tick_params(labelsize=17)
# #         cbar.ax.set_ylabel(ylabel='Cohens d', fontsize=17)

# #         fig.tight_layout()

# #         if Save_fig:
# #             os.makedirs(Run_graficos_path, exist_ok=True)
# #             plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.png'.format(Band))
# #             plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.svg'.format(Band))


# # # ===================
# # ## Violin Plot Stims

# # Save_fig = True
# # model = 'Ridge'
# # situacion = 'External'
# # tmin, tmax = -0.6, 0
# # Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)

# # Band = 'Theta'
# # Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

# # Correlaciones = {}
# # for stim in Stims:
# #     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
# #     Corr, Pass = pickle.load(f)
# #     f.close()

# #     Correlaciones[stim] = Corr.mean(0)

# # # Violin plot
# # plt.ion()
# # plt.figure(figsize=(19, 5))
# # sn.violinplot(data=pd.DataFrame(Correlaciones))
# # plt.ylabel('Correlation', fontsize=24)
# # plt.yticks(fontsize=20)
# # plt.xticks(fontsize=24)
# # plt.grid()
# # plt.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + '{}.png'.format(Band))
# #     plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

# # # Box plot
# # # my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta1': 'C0'}

# # fig, ax = plt.subplots()
# # ax = sn.boxplot(data=pd.DataFrame(Correlaciones), width=0.35)
# # ax.set_ylabel('Correlation')
# # # ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
# # for patch in ax.artists:
# #     r, g, b, a = patch.get_facecolor()
# #     patch.set_facecolor((r, g, b, .4))

# # # Create an array with the colors you want to use
# # # colors = ["C0", "grey"]
# # # # Set your custom color palette
# # # palette = sn.color_palette(colors)
# # sn.swarmplot(data=pd.DataFrame(Correlaciones), size=2, alpha=0.4)
# # plt.tick_params(labelsize=13)
# # ax.xaxis.label.set_size(15)
# # ax.yaxis.label.set_size(15)
# # plt.grid()
# # fig.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + '{}.png'.format(Band))
# #     plt.savefig(Run_graficos_path + '{}.svg'.format(Band))
# # # ==========================
# # ## Violin Plot Situation


# # model = 'Ridge'
# # Band = 'Theta'
# # stim = 'Spectrogram'
# # situaciones = ['External', 'Internal', 'Ambos', 'Internal_BS', 'Silencio']
# # tmin, tmax = -0.6, -0.003
# # fdr = True
# # if fdr:
# #     Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/fdr/'.format(Band, stim, tmin, tmax)
# # else:
# #     Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/log/'.format(Band, stim, tmin, tmax)
# # Save_fig = True
# # Display_fig = False
# # if Display_fig:
# #     plt.ion()
# # else:
# #     plt.ioff()

# # montage = mne.channels.make_standard_montage('biosemi128')
# # channel_names = montage.ch_names
# # info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# # Correlaciones = {}

# # for situacion in situaciones:
# #     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
# #     Corr, Pass = pickle.load(f)
# #     f.close()

# #     Correlaciones[situacion] = Corr.transpose().ravel()

# # my_pal = {'External': 'darkgrey', 'Internal': 'darkgrey', 'Ambos': 'darkgrey', 'Internal_BS': 'darkgrey', 'Silencio': 'darkgrey'}

# # plt.figure(figsize=(19, 5))
# # sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal)
# # plt.ylabel('Correlation', fontsize=24)
# # plt.yticks(fontsize=20)
# # plt.xticks(fontsize=24)
# # plt.grid()
# # ax = plt.gca()
# # ax.set_xticklabels(['Listening', 'Speech\nproduction', 'Listening \n (speaking)', 'Speaking \n (listening)',
# #                     'Silence'], fontsize=24)
# # plt.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + 'Violin_plot.png'.format(Band))
# #     plt.savefig(Run_graficos_path + 'Violin_plot.svg'.format(Band))

# # ## TRF amplitude

# # model = 'Ridge'
# # Band = 'Theta'
# # stim = 'Spectrogram'
# # sr = 128
# # tmin, tmax = -0.6, -0.003
# # delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
# # times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
# # times = np.flip(-times)
# # Stims_preprocess = 'Normalize'
# # EEG_preprocess = 'Standarize'
# # Save_fig = True

# # Listening_25_folds_path = 'saves/25_folds/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
# #     format(model, 'External', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
# # Listening_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
# #     format(model, 'External', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
# # Ambos_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
# #     format(model, 'Ambos', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)

# # Run_graficos_path = 'gráficos/Amplitude_Comparison/{}/{}/tmin{}_tmax{}/'.format(Band, stim, tmin, tmax)

# # # Load TRFs
# # f = open(Listening_25_folds_path, 'rb')
# # TRF_25 = pickle.load(f)
# # f.close()
# # TRF_25 = np.flip(TRF_25.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# # f = open(Listening_path, 'rb')
# # TRF_escucha = pickle.load(f)
# # f.close()
# # TRF_escucha = np.flip(TRF_escucha.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# # f = open(Ambos_path, 'rb')
# # TRF_ambos = pickle.load(f)
# # f.close()
# # TRF_ambos = np.flip(TRF_ambos.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

# # montage = mne.channels.make_standard_montage('biosemi128')
# # channel_names = montage.ch_names
# # info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# # mean_25 = TRF_25.mean(0)
# # mean_escucha = TRF_escucha.mean(0)
# # mean_ambos = TRF_ambos.mean(0)

# # plt.ion()

# # plt.figure(figsize=(15, 5))
# # plt.plot(times*1000, mean_escucha, label='L|O')
# # plt.plot(times*1000, mean_25, label='L|O downsampled')
# # plt.plot(times*1000, mean_ambos, label='L|B')
# # plt.xlim([(times*1000).min(), (times*1000).max()])
# # plt.xlabel('Time (ms)')
# # plt.ylabel('TRF (a.u.)')
# # plt.grid()
# # plt.legend()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + 'Plot.png'.format(Band))
# #     plt.savefig(Run_graficos_path + 'Plot.svg'.format(Band))

# # fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)
# # for i, plot_data, title in zip(range(3), [TRF_escucha, TRF_25, TRF_ambos], ['L|O', 'L|O downsampled', 'L | B']):
# #     evoked = mne.EvokedArray(plot_data, info)
# #     evoked.times = times
# #     evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
# #                 show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[i])
# #     axs[i].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
# #     axs[i].xaxis.label.set_size(14)
# #     axs[i].yaxis.label.set_size(14)
# #     axs[i].set_ylim([-0.016, 0.015])
# #     axs[i].tick_params(axis='both', labelsize=14)
# #     axs[i].grid()
# #     axs[i].legend(fontsize=12)
# #     axs[i].set_title(f'{title}', fontsize=15)
# #     if i != 2:
# #         axs[i].set_xlabel('', fontsize=14)
# #     else:
# #         axs[i].set_xlabel('Time (ms)', fontsize=14)

# # fig.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + 'Subplots.png'.format(Band))
# #     plt.savefig(Run_graficos_path + 'Subplots.svg'.format(Band))



# # ## Box Plot Bandas Decoding

# # tmin, tmax = -0.4, 0.2
# # model = 'Decoding'
# # situacion = 'External'

# # Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Box Plots/'.format(model, situacion, tmin, tmax)
# # Save_fig = False

# # stim = 'Envelope'
# # Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']

# # Correlaciones = pd.DataFrame(columns=['Corr', 'Sig'])

# # for Band in Bands:
# #     f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
# #     Corr_Pass = pickle.load(f)
# #     f.close()
# #     Corr = Corr_Pass[0]
# #     Pass = Corr_Pass[1]
# #     temp_df = pd.DataFrame({'Corr': Corr.ravel(), 'Sig': Pass.ravel(), 'Band': [Band]*len(Corr.ravel())})
# #     Correlaciones = pd.concat((Correlaciones, temp_df))

# # Correlaciones['Permutations test'] = np.where(Correlaciones['Sig'] == 1, 'NonSignificant', 'Significant')

# # my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta1': 'C0'}

# # fig, ax = plt.subplots()
# # ax = sn.boxplot(x='Band', y='Corr', data=Correlaciones, width=0.35, palette=my_pal)
# # # ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
# # for patch in ax.artists:
# #     r, g, b, a = patch.get_facecolor()
# #     patch.set_facecolor((r, g, b, .4))

# # # Create an array with the colors you want to use
# # colors = ["C0", "grey"]
# # # Set your custom color palette
# # palette = sn.color_palette(colors)
# # sn.swarmplot(x='Band', y='Corr', data=Correlaciones, hue='Permutations test', size=3, palette=palette)
# # plt.tick_params(labelsize=13)
# # ax.xaxis.label.set_size(15)
# # ax.yaxis.label.set_size(15)
# # fig.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + '{}.png'.format(stim))
# #     plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

# # for Band in Bands:
# #     print(f'\n{Band}')
# #     Passed_folds = (Correlaciones.loc[Correlaciones['Band'] == Band]['Permutations test'] == 'Significant').sum()
# #     Passed_subj = 0
# #     for subj in range(len(Pass)):
# #         Passed_subj += all(Pass[subj] < 1)
# #     print(f'Passed folds: {Passed_folds}/{len(Correlaciones.loc[Correlaciones["Band"] == Band])}')
# #     print(f'Passed subjects: {Passed_subj}/{len(Pass)}')

# # ## Heatmaps
# # model = 'Ridge'
# # situacion = 'External'
# # tmin, tmax = -0.6, -0.003
# # Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Heatmaps/'.format(model, situacion, tmin, tmax)
# # Save_fig = True

# # Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta1']
# # Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

# # Corrs_map = np.zeros((len(Stims), len(Bands)))
# # Sig_map = np.zeros((len(Stims), len(Bands)))

# # for i, stim in enumerate(Stims):
# #     Corr_stim = []
# #     Sig_stim = []
# #     for Band in Bands:
# #         f = open('saves/{}/{}/correlations/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
# #         Corr_Band, Sig_Band = pickle.load(f)
# #         f.close()
# #         Corr_stim.append(Corr_Band.mean())
# #         Sig_stim.append(Sig_Band.mean(1).sum(0))
# #     Corrs_map[i] = Corr_stim
# #     Sig_map[i] = Sig_stim

# # fig, ax = plt.subplots()
# # plt.imshow(Corrs_map)
# # plt.title('Correlation', fontsize=15)
# # ax.set_yticks(np.arange(len(Stims)))
# # ax.set_yticklabels(Stims, fontsize=13)
# # ax.set_xticks(np.arange(len(Bands)))
# # ax.set_xticklabels(Bands, fontsize=13)
# # # ax.xaxis.tick_top()
# # cbar = plt.colorbar(shrink=0.7, aspect=15)
# # cbar.ax.tick_params(labelsize=13)
# # fig.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + 'Corr.png'.format(Band))
# #     plt.savefig(Run_graficos_path + 'Corr.svg'.format(Band))

# # fig, ax = plt.subplots()
# # plt.imshow(Sig_map)
# # plt.title('Significance', fontsize=15)
# # ax.set_yticks(np.arange(len(Stims)))
# # ax.set_yticklabels(Stims, fontsize=13)
# # ax.set_xticks(np.arange(len(Bands)))
# # ax.set_xticklabels(Bands, fontsize=13)
# # # ax.xaxis.tick_top()
# # cbar = plt.colorbar(shrink=0.7, aspect=15)
# # cbar.ax.tick_params(labelsize=13)
# # fig.tight_layout()

# # if Save_fig:
# #     os.makedirs(Run_graficos_path, exist_ok=True)
# #     plt.savefig(Run_graficos_path + 'Stat.png'.format(Band))
# #     plt.savefig(Run_graficos_path + 'Stat.svg'.format(Band))


# # ## Plot por subjects
# # import mne

# # montage = mne.channels.make_standard_montage('biosemi128')
# # channel_names = montage.ch_names
# # info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

# # model = 'Ridge'
# # situacion = 'External'
# # alpha = 100
# # tmin, tmax = -0.6, -0.003

# # Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
# # Save_fig = False

# # Bands = ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']
# # Bands = ['Theta']
# # for Band in Bands:

# #     f = open('saves/Ridge/correlations/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
# #     Corr_Envelope, Pass_Envelope = pickle.load(f)
# #     f.close()

# #     f = open('saves/Ridge/correlations/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
# #     Corr_Pitch, Pass_Pitch = pickle.load(f)
# #     f.close()

# #     f = open(
# #         'saves/Ridge/correlations/tmin{}_tmax{}/Envelope_Pitch_Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band),
# #         'rb')
# #     Corr_Envelope_Pitch_Spectrogram, Pass_Envelope_Pitch_Spectrogram = pickle.load(f)
# #     f.close()

# #     f = open(
# #         'saves/Ridge/correlations/tmin{}_tmax{}/Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
# #     Corr_Spectrogram, Pass_Spectrogram = pickle.load(f)
# #     f.close()

# #     plt.ion()
# #     plt.figure()
# #     plt.title(Band)
# #     plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
# #     for i in range(len(Corr_Envelope)):
# #         plt.plot(Corr_Envelope[i], Corr_Pitch[i], '.', label='Subject {}'.format(i + 1))
# #     plt.legend()
# #     plt.grid()


# ================================================================================================================
# CORRELATION MATRIX TOPOMAP: make matrix with correlation topomap ordering across features, situations and bands.
# ================================================================================================================
path_specific_weights = os.path.join(path_figures,'specific_weights')

#===============
# ENVELOPE THETA

# Relevant parameters
band = 'Theta'
stimulus = 'Envelope'

# Get average weights across subjects
weights = load_pickle(path=os.path.join(weights_path, band, stimulus, 'total_weights_per_subject.pkl'))['average_weights_subjects'].mean(axis=0).mean(axis=1)
# weights = mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=1) #-mean_average_weights_subjects[:, index_slice[0]:index_slice[1], :].mean(axis=1).mean(axis=0)

# Create figure and title
fig, axes = plt.subplots(
                        # figsize=(3*n_stims,1.5*n_bands), 
                        nrows=1, 
                        ncols=1, 
                        layout="constrained"
                        )
fig.suptitle('Envelope-Theta weights')
evoked = mne.EvokedArray(data=weights, info=info_mne)
evoked.shift_time(times[0], relative=True)
evoked.plot(
            scalings={'eeg':1}, 
            zorder='std', 
            # gfp=True,
            time_unit='ms',
            show=False, 
            spatial_colors=True, 
            # unit=False, 
            units='mTRF (a.u.)',
            axes=axes
            )
axes.plot(
        times*1000, #ms
        evoked._data.mean(0), 
        'k--', 
        label='Mean', 
        zorder=130, 
        linewidth=2
        )
# Graph properties
axes.legend()
axes.grid(visible=True)
axes.set(xlabel='Time (ms)')
fig.show()

# Save figure
if save_figures:
    temp_path = os.path.join(path_specific_weights, stimulus)
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'{stimulus}_{band}.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'{stimulus}_{band}.svg'))
plt.close()

#========================
# PHONEMES-DISCRETE THETA

# Relevant parameters
band = 'Theta'
stimulus = 'Phonemes-Discrete-Phonet'

# Get average weights across subjects
weights = load_pickle(path=os.path.join(weights_path, band, stimulus, 'total_weights_per_subject.pkl'))['average_weights_subjects'].mean(axis=0).mean(axis=0)

# Create figure and title
fig, axes = plt.subplots(
                        # figsize=(3*n_stims,1.5*n_bands), 
                        nrows=1, 
                        ncols=1, 
                        layout="constrained"
                        )
fig.suptitle('Envelope-Theta weights')


# Create colormesh figure
im = axes.pcolormesh(
                    times*1000,
                    np.arange(weights.shape[0]), 
                    weights, 
                    cmap='RdBu', 
                    shading='auto',
                    vmin=-weights.max(), 
                    vmax=weights.max()
                    )
axes.set(ylabel='Phonemes', yticks=np.arange(weights.shape[0]), yticklabels=exp_info.ph_labels_phonet[:-1])
axes.tick_params(axis='both', labelsize='medium') # Change labelsize because there are too many phonemes

# Make color bar
fig.colorbar(
            im,
            ax=axes,
            orientation='horizontal',
            label='Amplitude (a.u.)',
            shrink=1,
            aspect=20
            )
# Graph properties
axes.set(xlabel='Time (ms)')
fig.show()

# Save figure
if save_figures:
    temp_path = os.path.join(path_specific_weights, stimulus)
    os.makedirs(temp_path, exist_ok=True)
    fig.savefig(os.path.join(temp_path, f'{stimulus}_{band}.png'), dpi=600)
    fig.savefig(os.path.join(temp_path, f'{stimulus}_{band}.svg'))
plt.close()

