from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')  # Use the Qt5Agg backend for interactive plotting
matplotlib.use('Agg')  # Use Agg backend for saving figures without display
import numpy as np
import mne 

from utils.general_functions import load_pickle
import config


path_trfs = Path(r'output\mtrf_ridge_torch\External\weights\stims_Normalize_EEG_Standarize\tmin-0.2_tmax0.6\Theta\Phonological')
trfs = load_pickle(
    path=path_trfs/'total_weights_per_subject.pkl'
)['average_weights_subjects']

labels = list(
    ph_feat for ph_feat in config.exp_info.phonological_labels.keys()\
    if ph_feat not in ['pause', 'trill'] 
    )
index1 = [
    labels.index(label) for label in labels\
    if label in config.exp_info.phonological_labels1
    ]
index2 = [
    labels.index(label) for label in labels\
    if label in config.exp_info.phonological_labels2
    ]

trfs1 = trfs[:, :, index1, :].mean(axis=0).mean(axis=1)
trfs2 = trfs[:, :, index2, :].mean(axis=0).mean(axis=1)


fig = plt.figure(
    tight_layout=True, 
    figsize=(10, 6), 
    dpi=500
)

ax1 = fig.add_subplot(311)

# Plot diff trfs over time 
evoked = mne.EvokedArray(
    data=trfs1-trfs2, 
    info=config.info_mne
)
evoked.shift_time(
    config.times[0], 
    relative=True
)
evoked_plot = evoked.plot(
    scalings={'eeg':1},
    zorder='std',
    time_unit='ms',
    show=False,
    spatial_colors=True,
    # unit=False,
    units='mTRFs (U.A)',
    axes=ax1,
    gfp=False
    )
# Eliminar la etiqueta "Nave"
for text in evoked_plot.axes[0].texts:
    if "ave" in text.get_text():
        text.set_visible(False)  # Ocultar el texto
ax1.plot(
    config.times*1e3, #ms
    evoked._data.mean(axis=0),
    'black',
    label='Valor medio',
    zorder=130,
    linewidth=2
    )


ax2 = fig.add_subplot(312, sharex=ax1)

# Plot diff trfs over time 
evoked = mne.EvokedArray(
    data=trfs1, 
    info=config.info_mne
)
evoked.shift_time(
    config.times[0], 
    relative=True
)
evoked_plot = evoked.plot(
    scalings={'eeg':1},
    zorder='std',
    time_unit='ms',
    show=False,
    spatial_colors=True,
    # unit=False,
    units='mTRFs (U.A)',
    axes=ax2,
    gfp=False
    )
# Eliminar la etiqueta "Nave"
for text in evoked_plot.axes[0].texts:
    if "ave" in text.get_text():
        text.set_visible(False)  # Ocultar el texto
ax2.plot(
    config.times*1e3, #ms
    evoked._data.mean(axis=0),
    'black',
    label='Valor medio',
    zorder=130,
    linewidth=2
    )


ax3 = fig.add_subplot(313, sharex=ax1)

# Plot diff trfs over time 
evoked = mne.EvokedArray(
    data=trfs2, 
    info=config.info_mne
)
evoked.shift_time(
    config.times[0], 
    relative=True
)
evoked_plot = evoked.plot(
    scalings={'eeg':1},
    zorder='std',
    time_unit='ms',
    show=False,
    spatial_colors=True,
    # unit=False,
    units='mTRFs (U.A)',
    axes=ax3,
    gfp=False
    )
# Eliminar la etiqueta "Nave"
for text in evoked_plot.axes[0].texts:
    if "ave" in text.get_text():
        text.set_visible(False)  # Ocultar el texto
ax3.plot(
    config.times*1e3, #ms
    evoked._data.mean(axis=0),
    'black',
    label='Valor medio',
    zorder=130,
    linewidth=2
    )
# fig.savefig(
#     Path(r'figures\analysis\topo_phonological_difference') / 'phonological1_topo.png', 
#     bbox_inches='tight', 
#     dpi=500
# )


old_eeg = load_pickle(path=Path(r'saves_old\preprocessed_data\External\tmin-0.2_tmax0.6\EEG\Theta\Causal\Sesion21.pkl'))[0].mean(axis=1)
new_eeg = load_pickle(path=Path(r'saves\preprocessed_data\External\tmin-0.2_tmax0.6\EEG\Theta\Causal\Sesion21.pkl'))[0].mean(axis=1)
oldd_eeg = load_pickle(path=Path(r'savesoldd\preprocessed_data\External\tmin-0.2_tmax0.6\EEG\Theta\Causal\Sesion21.pkl'))[0].mean(axis=1)

samples_info_old = load_pickle(path=Path(r'saves_old\preprocessed_data\External\tmin-0.2_tmax0.6\samples_info\samples_info_21.pkl'))
samples_info_new = load_pickle(path=Path(r'saves\preprocessed_data\External\tmin-0.2_tmax0.6\samples_info\samples_info_21.pkl'))
samples_info_old['trial_lengths1']==samples_info_new['trial_lengths1']  # Check if trial lengths are the same
samples_info_old['trial_lengths2']==samples_info_new['trial_lengths2']  # Check if trial lengths are the same
samples_info_old['keep_indexes1']==samples_info_new['keep_indexes1']  # Check if trial lengths are the same
samples_info_old['keep_indexes2']==samples_info_new['keep_indexes2']  # Check if trial lengths are the same

fig, ax = plt.subplots(
    figsize=(10, 6), 
    dpi=500,
    tight_layout=True
)
ax.plot(
    old_eeg[:500],
    label='Antiguo',
    color='blue'
)
ax.plot(
    oldd_eeg[:500],
    label='No tan nuevo',
    color='orange'
)
ax.plot(
    new_eeg[:500],
    label='Nuevo',
    color='green'
)
ax.legend(  )
fig.savefig(
    Path(r'figures\analysis\topo_phonological_difference') / 'eegelope_comparison.png', 
    bbox_inches='tight', 
    dpi=500
)