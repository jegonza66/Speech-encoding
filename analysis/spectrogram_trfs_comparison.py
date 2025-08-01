from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib
import mne 

from utils.general_functions import load_pickle
import config

matplotlib.use('Agg')  # Use Agg backend for saving figures without display

situations = [
    'External',
    'Internal',
    'External_BS',
    'Internal_BS'
]
stimulus = 'Spectrogram'
bands = [
    'Delta',
    'Theta',
    'Alpha',
    'Beta',
    'Broad'
]


for band in bands:
    fig, axes = plt.subplots(
            nrows=2, 
            ncols=2, 
            figsize=(8, 3), 
            dpi=600, 
            sharex=True, 
            sharey=True,
            tight_layout=True
        )
    axes = axes.flatten()
    for s, situation in enumerate(situations):
        
        path_trfs = Path(rf'output\mtrf-ridge\{situation}\weights\stims_Standarize_EEG_Standarize\distinct_alpha\tmin-0.2_tmax0.6\{band}\{stimulus}')
        trfs = load_pickle(
            path=path_trfs/'total_weights_per_subject.pkl'
        )['average_weights_subjects'].mean(axis=0).mean(axis=1) # Shape n_chan, n_times
        
        evoked = mne.EvokedArray(
            data=trfs, 
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
            axes=axes[s],
        )
        
        # Eliminar la etiqueta "Nave"
        for txt in fig.findobj(mtext.Text):
            if "ave" in txt.get_text():
                 txt.remove()
                
        # axes[s].plot(
        #     config.times*1e3, #ms
        #     evoked._data.mean(axis=0),
        #     'black',
        #     label='Valor medio',
        #     zorder=130,
        #     linewidth=1.2
        # )
        axes[s].set_xlabel('Time (ms)') if s >= 2 else axes[s].set_xlabel('')
        axes[s].set_ylim(-0.015, 0.015)
        axes[s].set_title(situation, fontsize=10)
        fig.suptitle(f'{stimulus} - {band}', fontsize=12)
        fig.savefig(
            rf'figures\analysis\spectrogram_trfs_comparison\{band}.png', 
        )   