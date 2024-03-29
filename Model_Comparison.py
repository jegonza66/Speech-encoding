import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import mne
from scipy.spatial import ConvexHull

model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, 0

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Convex_Hull/'.format(model, situacion, tmin, tmax)
Save_fig = True

Bands = ['Theta']
Stims = 'Envelope_Pitch_Phonemes'
sub_stims = Stims.split('_')
Corr = {}
Pass = {}

for Band in Bands:
    for stim in sub_stims + [Stims]:
        f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax,
                                                                                    stim, Band), 'rb')

        corr_data, pass_data = pickle.load(f)
        Corr[stim] = corr_data.ravel()
        Pass[stim] = pass_data.ravel()
        f.close()

    # PLOT
    plt.ion()
    plt.figure()
    plt.title(Band)

    for i, stim in enumerate(sub_stims):
        stim_points = np.array([Corr[Stims], Corr[stim]]).transpose()
        stim_hull = ConvexHull(stim_points)

        plt.plot(Corr[Stims][Pass[stim] != 0], Corr[stim][Pass[stim] != 0], '.', color=f'C{i}',
                 label=stim, ms=2.5)

        plt.plot(Corr[Stims][Pass[stim] == 0], Corr[stim][Pass[stim] == 0], '.', color='grey',
                 alpha=0.5, label='Perm. test failed', ms=2)
        plt.fill(stim_points[stim_hull.vertices, 0], stim_points[stim_hull.vertices, 1], color=f'C{i}',
                 alpha=0.3, lw=0)

    # Lines
    plt.plot([plt.xlim()[0], 0.7], [plt.xlim()[0], 0.7], 'k--', zorder=0)
    xlimit, ylimit = plt.xlim(), plt.ylim()
    plt.hlines(0, xlimit[0], xlimit[1], color='grey', linestyle='dashed')
    plt.vlines(0, ylimit[0], ylimit[1], color='grey', linestyle='dashed')
    plt.set_xlim = xlimit
    plt.set_ylim = ylimit
    plt.ylabel('Individual model (r)')
    plt.xlabel('Full model (r)')

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), markerscale=3)

    # Save
    plt.tight_layout()
    if Save_fig:
        fig_savepath = Run_graficos_path + f'{Band}/'
        os.makedirs(fig_savepath, exist_ok=True)
        plt.savefig(fig_savepath + f'{Stims}.png')
        plt.savefig(fig_savepath + f'{Stims}.svg')


## Violin / Topo (Bands)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import mne

tmin, tmax = -0.6, -0.003
model = 'Ridge'
situacion = 'Escucha'

info_path = 'saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
f = open(info_path, 'rb')
info = pickle.load(f)
f.close()

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Topo/'.format(model, situacion, tmin, tmax)
Save_fig = True
Correlaciones = {}

stim = 'Spectrogram'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[Band] = Corr.mean(0)

my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta_1': 'darkgrey'}

fontsize = 19
plt.rcParams.update({'font.size': fontsize})
# fig, axs = plt.subplots(figsize=(10, 4), ncols=5, nrows=2, gridspec_kw={'wspace': 0.25})
fig, axs = plt.subplots(figsize=(14, 5), ncols=5, nrows=2)

for i, Band in enumerate(Bands):
    ax = axs[0, i]
    fig.tight_layout()
    im = mne.viz.plot_topomap(Correlaciones[Band].ravel(), info, axes=ax, show=False, sphere=0.07, cmap='Reds',
                              vmin=Correlaciones[Band].min(), vmax=Correlaciones[Band].max())
    cbar = plt.colorbar(im[0], ax=ax, orientation='vertical', shrink=0.5)
    cbar.ax.tick_params(labelsize=fontsize)

for ax_row in axs[1:]:
    for ax in ax_row:
        ax.remove()

ax = fig.add_subplot(2, 1, (2, 3))

plt.suptitle(situacion)
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=ax)
ax.set_ylabel('Correlation')
ax.set_ylim([0, 0.5])
ax.grid()
ax.set_xticklabels(['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'])
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

## Violin / mTRF (Bands)
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import mne
import numpy as np

info_path = 'saves/Preprocesed_Data/tmin-0.6_tmax-0.003/EEG/info.pkl'
f = open(info_path, 'rb')
info = pickle.load(f)
f.close()

tmin_corr, tmax_corr = -0.6, -0.003
tmin_w, tmax_w = -0.6, 0.2
delays = - np.arange(np.floor(tmin_w * info['sfreq']), np.ceil(tmax_w * info['sfreq']), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin_w) * 1 / info['sfreq'], np.abs(delays[-1]) * np.sign(tmax_w) * 1 / info['sfreq'], len(delays))
times = np.flip(-times)

model = 'Ridge'
situacion = 'Ambos_Habla'

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin mTRF/'.format(model, situacion, tmin_corr, tmax_corr)
Save_fig = True
Correlaciones = {}
mTRFs = {}

stim = 'Spectrogram'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin_corr, tmax_corr, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()
    Correlaciones[Band] = Corr.mean(0)

f = open('saves/{}/{}/Original/Stims_Normalize_EEG_Standarize/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.format(model, situacion, tmin_w, tmax_w, stim, 'Theta', stim, 'Theta'), 'rb')
mTRFs_Theta = pickle.load(f)
f.close()

my_pal = {'All': 'darkgrey', 'Delta': 'darkgrey', 'Theta': 'C1', 'Alpha': 'darkgrey', 'Beta_1': 'darkgrey'}

fontsize = 17
plt.rcParams.update({'font.size': fontsize})
fig, axs = plt.subplots(figsize=(8, 5), nrows=2, gridspec_kw={'height_ratios': [1, 1]})

plt.suptitle(situacion)
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal, ax=axs[0])
axs[0].set_ylabel('Correlation')
axs[0].set_ylim([-0.1, 0.5])
axs[0].grid()
axs[0].set_xticklabels(['Broad band', 'Delta', 'Theta', 'Alpha', 'Low Beta'])


spectrogram_weights_chanels = mTRFs_Theta.reshape(info['nchan'], 16, len(times)).mean(1)

# Adapt for ERP
spectrogram_weights_chanels = np.flip(spectrogram_weights_chanels, axis=1)

evoked = mne.EvokedArray(spectrogram_weights_chanels, info)
evoked.times = times
evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
            show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[1])
# axs[1].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
axs[1].set_ylim([-0.016, 0.013])
if times[0] < 0:
    # ax.axvspan(ax.get_xlim()[0], 0, alpha=0.4, color='grey', label='Pre-Stimuli')
    axs[1].axvline(x=0, ymin=0, ymax=1, color='grey')

# axs[1].grid()

fig.tight_layout()
plt.show()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))


## Wilcoxon test topoplots
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from scipy.stats import wilcoxon
import mne
from statsmodels.stats.multitest import fdrcorrection
from Statistics import cohen_d

model = 'Ridge'
Band = 'Theta'
stim = 'Spectrogram'
situaciones = ['Escucha', 'Habla_Propia', 'Ambos', 'Ambos_Habla', 'Silencio']
tmin, tmax = -0.6, -0.003
stat_test = 'log'  # fdr/log/cohen
mask = False
Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/{}'.format(Band, stim, tmin, tmax, stat_test)
if mask and stat_test != 'cohen':
    Run_graficos_path += '_mask/'
else:
    Run_graficos_path += '/'

Save_fig = True
Display_fig = False
if Display_fig:
    plt.ion()
else:
    plt.ioff()

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

Correlaciones = {}

for situacion in situaciones:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[situacion] = Corr.transpose()

# Calculate wilcoxon test over rows of the dictionary
stats = {}
pvals = {}

for sit1, sit2 in zip(('Escucha', 'Escucha', 'Escucha', 'Escucha', 'Habla_Propia', 'Habla_Propia', 'Ambos', 'Ambos', 'Ambos', 'Ambos_Habla'),
                      ('Habla_Propia', 'Ambos', 'Silencio', 'Ambos_Habla', 'Silencio', 'Ambos_Habla', 'Ambos_Habla', 'Silencio', 'Habla_Propia', 'Silencio')):
    print(sit1, sit2)

    dist1 = Correlaciones[sit1]
    dist2 = Correlaciones[sit2]

    stat = []
    pval = []
    for i in range(len(dist1)):
        res = wilcoxon(dist1[i], dist2[i])
        stat.append(res[0])
        pval.append(res[1])

    stats[f'{sit1}-{sit2}'] = stat
    pvals[f'{sit1}-{sit2}'] = pval

    if stat_test == 'fdr':
        passed, corrected_pval = fdrcorrection(pvals=pval, alpha=0.05, method='p')

        if mask:
            corrected_pval[~passed] = 1

        # Plot pvalue
        fig, ax = plt.subplots()
        fig.suptitle(f'FDR p-value: {sit1}-{sit2}\n'
                     f'mean: {round(np.mean(corrected_pval), 6)} - '
                     f'min: {round(np.min(corrected_pval), 6)} - '
                     f'max: {round(np.max(corrected_pval), 6)}\n'
                     f'passed: {sum(passed)}', fontsize=17)
        im = mne.viz.plot_topomap(corrected_pval, vmin=0, vmax=1, pos=info, axes=ax, show=Display_fig, sphere=0.07,
                                  cmap='Reds_r')
        cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
        cbar.ax.yaxis.set_tick_params(labelsize=17)
        cbar.ax.set_ylabel(ylabel='FDR corrected p-value', fontsize=17)

        fig.tight_layout()

        if Save_fig:
            os.makedirs(Run_graficos_path, exist_ok=True)
            plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
            plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

    elif stat_test == 'log':
        log_pval = np.log10(pval)
        if mask:
            log_pval[np.array(pval) > 0.05/128] = 0

        # Plot pvalue
        fig, ax = plt.subplots()
        fig.suptitle(f'p-value: {sit1}-{sit2}\n'
                     f'mean: {round(np.mean(pvals[f"{sit1}-{sit2}"]), 6)} - '
                     f'min: {round(np.min(pvals[f"{sit1}-{sit2}"]), 6)} - '
                     f'max: {round(np.max(pvals[f"{sit1}-{sit2}"]), 6)}\n'
                     f'passed: {sum(np.array(pval)< 0.05/128)}', fontsize=17)
        im = mne.viz.plot_topomap(log_pval, vmin=-6, vmax=-2, pos=info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds_r')
        cbar = plt.colorbar(im[0], ax=ax, shrink=0.85, ticks=[-6, -5, -4, -3, -2])
        cbar.ax.yaxis.set_tick_params(labelsize=17)
        cbar.ax.set_yticklabels(['<10-6', '10-5', '10-4', '10-3', '10-2'])
        cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)


        fig.tight_layout()

        if Save_fig:
            os.makedirs(Run_graficos_path, exist_ok=True)
            plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.png'.format(Band))
            plt.savefig(Run_graficos_path + f'pval_{sit1}-{sit2}.svg'.format(Band))

        # Plot statistic
        fig, ax = plt.subplots()
        fig.suptitle(f'stat: {sit1}-{sit2}\n'
                     f'Mean: {round(np.mean(stats[f"{sit1}-{sit2}"]), 3)}', fontsize=19)
        im = mne.viz.plot_topomap(stats[f'{sit1}-{sit2}'], info, axes=ax, show=Display_fig, sphere=0.07, cmap='Reds')
        cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
        cbar.ax.yaxis.set_tick_params(labelsize=17)
        cbar.ax.set_ylabel(ylabel='p-value', fontsize=17)
        fig.tight_layout()

        if Save_fig:
            os.makedirs(Run_graficos_path, exist_ok=True)
            plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.png'.format(Band))
            plt.savefig(Run_graficos_path + f'stat_{sit1}-{sit2}.svg'.format(Band))

    elif stat_test == 'cohen':
        cohen_ds = []
        for i in range(len(dist1)):
            cohen_ds.append(cohen_d(dist1[i], dist2[i]))

        # Plot pvalue
        fig, ax = plt.subplots()
        fig.suptitle(f'Cohen d: {sit1}-{sit2}\n'
                     f'mean: {round(np.mean(cohen_ds), 2)} +- {round(np.std(cohen_ds), 2)} -\n'
                     f'min: {round(np.min(cohen_ds), 2)} - '
                     f'max: {round(np.max(cohen_ds), 2)}', fontsize=17)
        im = mne.viz.plot_topomap(cohen_ds, vmin=0, vmax=4, pos=info, axes=ax, show=Display_fig, sphere=0.07,
                                  cmap='Reds')
        cbar = plt.colorbar(im[0], ax=ax, shrink=0.85)
        cbar.ax.yaxis.set_tick_params(labelsize=17)
        cbar.ax.set_ylabel(ylabel='Cohens d', fontsize=17)

        fig.tight_layout()

        if Save_fig:
            os.makedirs(Run_graficos_path, exist_ok=True)
            plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.png'.format(Band))
            plt.savefig(Run_graficos_path + f'cohen_{sit1}-{sit2}.svg'.format(Band))



## Violin Plot Stims
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn

Save_fig = True
model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, 0
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Violin Plots/'.format(model, situacion, tmin, tmax)

Band = 'Theta'
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

Correlaciones = {}
for stim in Stims:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[stim] = Corr.mean(0)

# Violin plot
plt.ion()
plt.figure(figsize=(19, 5))
sn.violinplot(data=pd.DataFrame(Correlaciones))
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=24)
plt.grid()
plt.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(Band))
    plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

# Box plot
# my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta_1': 'C0'}

fig, ax = plt.subplots()
ax = sn.boxplot(data=pd.DataFrame(Correlaciones), width=0.35)
ax.set_ylabel('Correlation')
# ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

# Create an array with the colors you want to use
# colors = ["C0", "grey"]
# # Set your custom color palette
# palette = sn.color_palette(colors)
sn.swarmplot(data=pd.DataFrame(Correlaciones), size=2, alpha=0.4)
plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
plt.grid()
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(Band))
    plt.savefig(Run_graficos_path + '{}.svg'.format(Band))

## Violin Plot Situation
import pandas as pd
import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sn
from scipy.stats import wilcoxon
import mne
from statsmodels.stats.multitest import fdrcorrection

model = 'Ridge'
Band = 'Theta'
stim = 'Spectrogram'
situaciones = ['Escucha', 'Habla_Propia', 'Ambos', 'Ambos_Habla', 'Silencio']
tmin, tmax = -0.6, -0.003
fdr = True
if fdr:
    Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/fdr/'.format(Band, stim, tmin, tmax)
else:
    Run_graficos_path = 'gráficos/SIS_statistics/{}/{}/tmin{}_tmax{}/log/'.format(Band, stim, tmin, tmax)
Save_fig = True
Display_fig = False
if Display_fig:
    plt.ion()
else:
    plt.ioff()

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

Correlaciones = {}

for situacion in situaciones:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr, Pass = pickle.load(f)
    f.close()

    Correlaciones[situacion] = Corr.transpose().ravel()

my_pal = {'Escucha': 'darkgrey', 'Habla_Propia': 'darkgrey', 'Ambos': 'darkgrey', 'Ambos_Habla': 'darkgrey', 'Silencio': 'darkgrey'}

plt.figure(figsize=(19, 5))
sn.violinplot(data=pd.DataFrame(Correlaciones), palette=my_pal)
plt.ylabel('Correlation', fontsize=24)
plt.yticks(fontsize=20)
plt.xticks(fontsize=24)
plt.grid()
ax = plt.gca()
ax.set_xticklabels(['Listening', 'Speech\nproduction', 'Listening \n (speaking)', 'Speaking \n (listening)',
                    'Silence'], fontsize=24)
plt.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Violin_plot.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Violin_plot.svg'.format(Band))

## TRF amplitude

import numpy as np
import pickle
import matplotlib.pyplot as plt
import mne

model = 'Ridge'
Band = 'Theta'
stim = 'Spectrogram'
sr = 128
tmin, tmax = -0.6, -0.003
delays = - np.arange(np.floor(tmin * sr), np.ceil(tmax * sr), dtype=int)
times = np.linspace(delays[0] * np.sign(tmin) * 1 / sr, np.abs(delays[-1]) * np.sign(tmax) * 1 / sr, len(delays))
times = np.flip(-times)
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'
Save_fig = True

Listening_25_folds_path = 'saves/25_folds/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
Listening_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Escucha', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)
Ambos_path = 'saves/{}/{}/Original/Stims_{}_EEG_{}/tmin{}_tmax{}/Stim_{}_EEG_Band_{}/Pesos_Totales_{}_{}.pkl'.\
    format(model, 'Ambos', Stims_preprocess, EEG_preprocess, tmin, tmax, stim, Band, stim, Band)

Run_graficos_path = 'gráficos/Amplitude_Comparison/{}/{}/tmin{}_tmax{}/'.format(Band, stim, tmin, tmax)

# Load TRFs
f = open(Listening_25_folds_path, 'rb')
TRF_25 = pickle.load(f)
f.close()
TRF_25 = np.flip(TRF_25.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

f = open(Listening_path, 'rb')
TRF_escucha = pickle.load(f)
f.close()
TRF_escucha = np.flip(TRF_escucha.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

f = open(Ambos_path, 'rb')
TRF_ambos = pickle.load(f)
f.close()
TRF_ambos = np.flip(TRF_ambos.reshape(info['nchan'], 16, len(delays)), axis=2).mean(1)

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

mean_25 = TRF_25.mean(0)
mean_escucha = TRF_escucha.mean(0)
mean_ambos = TRF_ambos.mean(0)

plt.ion()

plt.figure(figsize=(15, 5))
plt.plot(times*1000, mean_escucha, label='L|O')
plt.plot(times*1000, mean_25, label='L|O downsampled')
plt.plot(times*1000, mean_ambos, label='L|B')
plt.xlim([(times*1000).min(), (times*1000).max()])
plt.xlabel('Time (ms)')
plt.ylabel('TRF (a.u.)')
plt.grid()
plt.legend()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Plot.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Plot.svg'.format(Band))

fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 9), sharex=True)
for i, plot_data, title in zip(range(3), [TRF_escucha, TRF_25, TRF_ambos], ['L|O', 'L|O downsampled', 'L | B']):
    evoked = mne.EvokedArray(plot_data, info)
    evoked.times = times
    evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', titles=dict(eeg=''),
                show=False, spatial_colors=True, unit=True, units='TRF (a.u.)', axes=axs[i])
    axs[i].plot(times * 1000, evoked._data.mean(0), "k--", label="Mean", zorder=130, linewidth=2)
    axs[i].xaxis.label.set_size(14)
    axs[i].yaxis.label.set_size(14)
    axs[i].set_ylim([-0.016, 0.015])
    axs[i].tick_params(axis='both', labelsize=14)
    axs[i].grid()
    axs[i].legend(fontsize=12)
    axs[i].set_title(f'{title}', fontsize=15)
    if i != 2:
        axs[i].set_xlabel('', fontsize=14)
    else:
        axs[i].set_xlabel('Time (ms)', fontsize=14)

fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Subplots.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Subplots.svg'.format(Band))


## Venn Diagrams
from matplotlib_venn import venn3, venn2  # venn3_circles
from matplotlib import pyplot as plt
import pickle
import os
import numpy as np
import copy

tmin, tmax = -0.6, 0
model = 'Ridge'
situacion = 'Escucha'
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Venn_Diagrams/'.format(model, situacion, tmin, tmax)
save_path = 'saves/{}/{}/Final_Correlation/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
Save_fig = True


# f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/Mean_Correlations.pkl'.format(model, situacion, tmin, tmax), 'rb')
# Mean_Correlations = pickle.load(f)
# f.close()

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']
Bands = ['Theta']


for Band in Bands:
    Mean_Correlations = {}

    stims_base = ['Spectrogram', 'Phonemes']
    stims = copy.copy(stims_base)

    for i in range(len(stims_base)-1):
        for j in range(i+1, len(stims_base)):
            stims.append('{}_{}'.format(stims_base[i], stims_base[j]))
    stims.append('_'.join(stims_base))

    for stim in stims:
        # open FInal correlation file
        f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'rb')
        Final_correlation = pickle.load(f)
        f.close()

        Mean_Correlations[stim] = Final_correlation[0].mean()

    if len(stims_base) == 2:
        r2_1 = Mean_Correlations[stims[0]] ** 2
        r2_2 = Mean_Correlations[stims[1]] ** 2
        r2_12 = Mean_Correlations[stims[2]] ** 2
        r2_int_12 = r2_1 + r2_2 - r2_12

        r2u_1 = r2_12 - r2_2
        # r2u_2 = Mean_Correlations[Band][stims[1]][0]**2 + r2_int_123 - r2_int_12 - r2_int_23
        r2u_2 = r2_12 - r2_1

        sets = [r2u_1, r2u_2, r2_int_12]
        sets_0 = []
        for set in sets:
            if set < 0:
                set = 0
            sets_0.append(set)

        plt.figure()
        plt.title('{}'.format(Band))
        venn2(subsets=np.array(sets_0).round(5), set_labels=(stims[0], stims[1]),
              set_colors=('C0', 'C1'), alpha=0.45)
        plt.tight_layout()

        if Save_fig:
            fig_savepath = Run_graficos_path + f'{Band}/'
            os.makedirs(fig_savepath, exist_ok=True)
            plt.savefig(fig_savepath + f'{stims[-1]}.png')
            plt.savefig(fig_savepath + f'{stims[-1]}.svg')

        stim1_percent = r2_1 * 100 /np.sum(sets_0)
        stim2_percent = r2_2 * 100 /np.sum(sets_0)

        Envelope_u_percent = r2u_1 * 100 /np.sum(sets_0)
        Pitch_u_percent = r2u_2 * 100 /np.sum(sets_0)


    if len(stims_base) == 3:
        r2_1 = Mean_Correlations[stims[0]]**2
        r2_2 = Mean_Correlations[stims[1]]**2
        r2_3 = Mean_Correlations[stims[2]]**2
        r2_12 = Mean_Correlations[stims[3]]**2
        r2_13 = Mean_Correlations[stims[4]]**2
        r2_23 = Mean_Correlations[stims[5]]**2
        r2_123 = Mean_Correlations[stims[6]]**2

        r2_int_12 = r2_1 + r2_2 - r2_12
        r2_int_13 = r2_1 + r2_3 - r2_13
        r2_int_23 = r2_2 + r2_3 - r2_23
        r2_int_123 = r2_123 + r2_1 + r2_2 + r2_3 - r2_12 - r2_13 - r2_23

        r2u_int_12 = r2_1 + r2_2 - r2_12 - r2_int_123
        r2u_int_13 = r2_1 + r2_3 - r2_13 - r2_int_123
        r2u_int_23 = r2_2 + r2_3 - r2_23 - r2_int_123

        # r2u_1 = Mean_Correlations[Band][stims[0]][0]**2 + r2_int_123 - r2_int_12 - r2_int_13
        r2u_1 = r2_123 - r2_23
        # r2u_2 = Mean_Correlations[Band][stims[1]][0]**2 + r2_int_123 - r2_int_12 - r2_int_23
        r2u_2 = r2_123 - r2_13
        # r2u_3 = Mean_Correlations[Band][stims[2]][0]**2 + r2_int_123 - r2_int_13 - r2_int_23
        r2u_3 = r2_123 - r2_12

        sets = [r2u_1, r2u_2, r2u_int_12, r2u_3, r2u_int_13, r2u_int_23, r2_int_123]
        sets_0 = []
        for set in sets:
            if set < 0:
                set = 0
            sets_0.append(set)

        plt.figure()
        plt.title('{}'.format(Band))
        venn3(subsets=np.array(sets_0).round(5), set_labels=(stims[0], stims[1], stims[2]), set_colors=('C0', 'C1', 'purple'), alpha=0.45)

        plt.tight_layout()
        if Save_fig:
            fig_savepath = Run_graficos_path + f'{Band}/'
            os.makedirs(fig_savepath, exist_ok=True)
            plt.savefig(fig_savepath + f'{stims[-1]}.png')
            plt.savefig(fig_savepath + f'{stims[-1]}.svg')

# Envelope_percent = r2_1 * 100 /np.sum(sets_0)
# Pitch_percent = r2_2 * 100 /np.sum(sets_0)
# Spectrogram_percent = r2_3 * 100 /np.sum(sets_0)
#
# Envelope_u_percent = r2u_1 * 100 /np.sum(sets_0)
# Pitch_u_percent = r2u_2 * 100 /np.sum(sets_0)
# Spectrogram_u_percent = r2u_3 * 100 /np.sum(sets_0)

## Box Plot Bandas Decoding
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os
import seaborn as sn
import numpy as np

tmin, tmax = -0.4, 0.2
model = 'Decoding'
situacion = 'Escucha'

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Box Plots/'.format(model, situacion, tmin, tmax)
Save_fig = False

stim = 'Envelope'
Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']

Correlaciones = pd.DataFrame(columns=['Corr', 'Sig'])

for Band in Bands:
    f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
    Corr_Pass = pickle.load(f)
    f.close()
    Corr = Corr_Pass[0]
    Pass = Corr_Pass[1]
    temp_df = pd.DataFrame({'Corr': Corr.ravel(), 'Sig': Pass.ravel(), 'Band': [Band]*len(Corr.ravel())})
    Correlaciones = pd.concat((Correlaciones, temp_df))

Correlaciones['Permutations test'] = np.where(Correlaciones['Sig'] == 1, 'NonSignificant', 'Significant')

my_pal = {'All': 'C0', 'Delta': 'C0', 'Theta': 'C0', 'Alpha': 'C0', 'Beta_1': 'C0'}

fig, ax = plt.subplots()
ax = sn.boxplot(x='Band', y='Corr', data=Correlaciones, width=0.35, palette=my_pal)
# ax = sn.violinplot(x='Band', y='Corr', data=Correlaciones, width=0.35)
for patch in ax.artists:
    r, g, b, a = patch.get_facecolor()
    patch.set_facecolor((r, g, b, .4))

# Create an array with the colors you want to use
colors = ["C0", "grey"]
# Set your custom color palette
palette = sn.color_palette(colors)
sn.swarmplot(x='Band', y='Corr', data=Correlaciones, hue='Permutations test', size=3, palette=palette)
plt.tick_params(labelsize=13)
ax.xaxis.label.set_size(15)
ax.yaxis.label.set_size(15)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + '{}.png'.format(stim))
    plt.savefig(Run_graficos_path + '{}.svg'.format(stim))

for Band in Bands:
    print(f'\n{Band}')
    Passed_folds = (Correlaciones.loc[Correlaciones['Band'] == Band]['Permutations test'] == 'Significant').sum()
    Passed_subj = 0
    for subj in range(len(Pass)):
        Passed_subj += all(Pass[subj] < 1)
    print(f'Passed folds: {Passed_folds}/{len(Correlaciones.loc[Correlaciones["Band"] == Band])}')
    print(f'Passed subjects: {Passed_subj}/{len(Pass)}')

## Heatmaps
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

model = 'Ridge'
situacion = 'Escucha'
tmin, tmax = -0.6, -0.003
Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/Heatmaps/'.format(model, situacion, tmin, tmax)
Save_fig = True

Bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta_1']
Stims = ['Spectrogram', 'Envelope', 'Pitch', 'Shimmer']

Corrs_map = np.zeros((len(Stims), len(Bands)))
Sig_map = np.zeros((len(Stims), len(Bands)))

for i, stim in enumerate(Stims):
    Corr_stim = []
    Sig_stim = []
    for Band in Bands:
        f = open('saves/{}/{}/Final_Correlation/tmin{}_tmax{}/{}_EEG_{}.pkl'.format(model, situacion, tmin, tmax, stim, Band), 'rb')
        Corr_Band, Sig_Band = pickle.load(f)
        f.close()
        Corr_stim.append(Corr_Band.mean())
        Sig_stim.append(Sig_Band.mean(1).sum(0))
    Corrs_map[i] = Corr_stim
    Sig_map[i] = Sig_stim

fig, ax = plt.subplots()
plt.imshow(Corrs_map)
plt.title('Correlation', fontsize=15)
ax.set_yticks(np.arange(len(Stims)))
ax.set_yticklabels(Stims, fontsize=13)
ax.set_xticks(np.arange(len(Bands)))
ax.set_xticklabels(Bands, fontsize=13)
# ax.xaxis.tick_top()
cbar = plt.colorbar(shrink=0.7, aspect=15)
cbar.ax.tick_params(labelsize=13)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Corr.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Corr.svg'.format(Band))

fig, ax = plt.subplots()
plt.imshow(Sig_map)
plt.title('Significance', fontsize=15)
ax.set_yticks(np.arange(len(Stims)))
ax.set_yticklabels(Stims, fontsize=13)
ax.set_xticks(np.arange(len(Bands)))
ax.set_xticklabels(Bands, fontsize=13)
# ax.xaxis.tick_top()
cbar = plt.colorbar(shrink=0.7, aspect=15)
cbar.ax.tick_params(labelsize=13)
fig.tight_layout()

if Save_fig:
    os.makedirs(Run_graficos_path, exist_ok=True)
    plt.savefig(Run_graficos_path + 'Stat.png'.format(Band))
    plt.savefig(Run_graficos_path + 'Stat.svg'.format(Band))


## Plot por subjects
import mne

montage = mne.channels.make_standard_montage('biosemi128')
channel_names = montage.ch_names
info = mne.create_info(ch_names=channel_names[:], sfreq=128, ch_types='eeg').set_montage(montage)

model = 'Ridge'
situacion = 'Escucha'
alpha = 100
tmin, tmax = -0.6, -0.003

Run_graficos_path = 'gráficos/Model_Comparison/{}/{}/tmin{}_tmax{}/'.format(model, situacion, tmin, tmax)
Save_fig = False

Bands = ['Delta', 'Theta', 'Alpha', 'Beta_1', 'Beta_2', 'All']
Bands = ['Theta']
for Band in Bands:

    f = open('saves/Ridge/Final_Correlation/tmin{}_tmax{}/Envelope_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Envelope, Pass_Envelope = pickle.load(f)
    f.close()

    f = open('saves/Ridge/Final_Correlation/tmin{}_tmax{}/Pitch_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Pitch, Pass_Pitch = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/tmin{}_tmax{}/Envelope_Pitch_Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band),
        'rb')
    Corr_Envelope_Pitch_Spectrogram, Pass_Envelope_Pitch_Spectrogram = pickle.load(f)
    f.close()

    f = open(
        'saves/Ridge/Final_Correlation/tmin{}_tmax{}/Spectrogram_EEG_{}.pkl'.format(tmin, tmax, Band), 'rb')
    Corr_Spectrogram, Pass_Spectrogram = pickle.load(f)
    f.close()

    plt.ion()
    plt.figure()
    plt.title(Band)
    plt.plot([plt.xlim()[0], 0.55], [plt.xlim()[0], 0.55], 'k--')
    for i in range(len(Corr_Envelope)):
        plt.plot(Corr_Envelope[i], Corr_Pitch[i], '.', label='Subject {}'.format(i + 1))
    plt.legend()
    plt.grid()
