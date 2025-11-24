import matplotlib.pyplot as plt
import matplotlib.text as mtext
from matplotlib import rc
import scienceplots
import matplotlib

plt.style.use(['science'])
rc('text', usetex=True)

from pathlib import Path
import mne

from utils.general_functions import load_pickle
import config

SAVE_FIG_DIR = Path("figures/analysis/situation_trf_comparison/")
SAVE_FIG_DIR.mkdir(parents=True, exist_ok=True)
SHAREY = False
STIMULUS = 'Spectrogram-21'
SITUATIONS = [
    'External-External',
    'External-Internal',
    'External-External_BS',
    'External-Internal_BS'
]
SITUATIONS_STR_MAP = {
    'External-External': 'Ex-External',
    'External-Internal': 'Ex-Internal',
    'External-External_BS': 'Ex-External (BS)',
    'External-Internal_BS': 'Ex-Internal (BS)'
}
BANDS = [
    'Delta',
    'Theta',
    'Alpha',
    'Beta',
    'Broad'
]
CORRELATIONS_DIR = lambda situation, band: Path(
    rf'output\mtrf-ridge\{situation}\correlations\same_alpha\tmin-0.2_tmax0.6\{band}\{STIMULUS}.pkl'
)
TRFS_PATH = lambda situation, band: Path(
    rf'output\mtrf-ridge\{situation}\weights\stims_Standarize_EEG_Standarize\same_alpha\tmin-0.2_tmax0.6\{band}\{STIMULUS}\total_weights_per_subject.pkl'
)

for band in BANDS:
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(8, 3),
        # dpi=600,
        sharex=True,
        sharey=SHAREY,
        constrained_layout=True
    )
    fig.suptitle(f'{STIMULUS} - {band}', fontsize=12)
    axes = axes.flatten()
    for s, situation in enumerate(SITUATIONS):
        trfs = load_pickle(
            path=TRFS_PATH(situation, band)
        )['average_weights_subjects'].mean(axis=0).mean(axis=1)
        correlations = load_pickle(
            path=CORRELATIONS_DIR(situation, band)
        )['average_correlation_subjects']
        corr_mean = correlations.mean()
        corr_std = correlations.mean(axis=1).std(ddof=1)/correlations.shape[1]**0.5

        situation = SITUATIONS_STR_MAP[situation] + fr' - $\rho = ({corr_mean:.2f} \pm {corr_std:.2f})$'


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
            units='mTRFs (U.A)' if s in [0,2] else None,
            axes=axes[s],
        )

        # Eliminar la etiqueta "Nave"
        for txt in fig.findobj(mtext.Text):
            if "ave" in txt.get_text():
                 txt.remove()

        # axes[s].plot(
        #     config.times*1e3, #ms
        #     evoked._data.mean(axis=0),
        #     color='black',
        #     label='Mean value across channels',
        #     zorder=130,
        #     linewidth=1.2
        # )
        axes[s].set_xlabel('Time (ms)') if s >= 2 else axes[s].set_xlabel('')
        if SHAREY==True:
            if situation=='External-External':
                maximum = abs(trfs).max()
        else:
            maximum = abs(trfs).max()
        axes[s].set_ylim(-maximum, maximum)
        axes[s].set_title(situation, fontsize=10)
        axes[s].legend(loc='upper right', fontsize=8)
    fig.savefig(
        SAVE_FIG_DIR / f'situation_trf_comparison_{STIMULUS}_{band}_sharey_{SHAREY}.png',
        dpi=600
    )
    # plt.show(block=True)
print('Figures saved in ', SAVE_FIG_DIR.resolve())