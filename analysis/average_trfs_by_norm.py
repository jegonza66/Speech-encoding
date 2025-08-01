from pathlib import Path
import numpy as np
import mne

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import matplotlib
matplotlib.use('Agg')  

from utils.general_functions import load_pickle
from utils.processing import clustering_by_correlation
import config

stimuli = [
    "Envelope",
    "Pitch-Log-Raw",
    "Spectrogram",
    "Phonological",
    "Phonemes",
    "Phonemes-Discrete"    
]
bands = [
    "Delta",
    "Theta",
    "Alpha",
    "Beta",
    "Broad"
]
Path(rf'figures\analysis\average_trfs_by_norm').mkdir(
    parents=True, 
    exist_ok=True
)
files_not_found = []
for band in bands:
    for stimulus in stimuli:
        
        # Load the data
        # data_path = Path(
        #     rf"output\mtrf-ridge\External-External\weights\stims_Standarize_EEG_Standarize\same_alpha\tmin-0.2_tmax0.6\{band}\{stimulus}\total_weights_per_subject.pkl"
        # )# TODO va con distinct_alpha, pero no se corrió todavía
        data_path = Path(
            rf"output\mtrf-ridge-laplacian\External-External\weights\stims_Standarize_EEG_Standarize\same_alpha\tmin-0.2_tmax0.6\{band}\{stimulus}\total_weights_per_subject.pkl"
        )# TODO va con distinct_alpha, pero no se corrió todavía
            
        try:
            trfs = load_pickle(
                path=data_path
            )["average_weights_subjects"]
            print(f"\n\nProcessing {stimulus} in {band} band\n")
        except Exception as e:
            files_not_found.append((stimulus, band))
            continue        
        
        # Apply norm to all subjects, so they are comparable
        trfs_normalized = []
        for subj_trf in trfs:
            norm = np.linalg.norm(
                x=subj_trf
            )
            if norm == 0:
                trfs_normalized.append(subj_trf)
            else:
                trfs_normalized.append(subj_trf / norm)
        
        trfs_normalized = np.stack(
            trfs_normalized
        )
        if trfs_normalized.shape[2] == 1:
            average_trfs_normalized = trfs_normalized.mean(axis=0).mean(axis=1)
            fig = plt.figure(
                figsize=(6, 4),
                constrained_layout=True
            )
            ax = plt.subplot(111)
            
            evoked = mne.EvokedArray(
                data=average_trfs_normalized, 
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
                axes=ax,
            )
            
            # Eliminar la etiqueta "Nave"
            for txt in fig.findobj(mtext.Text):
                if "ave" in txt.get_text():
                        txt.remove()
                    
            ax.plot(
                config.times*1e3, #ms
                evoked._data.mean(axis=0),
                zorder=130,
                linewidth=1.2,
                label='mean',
                color='black'
            )
            ax.set_xlabel('Time (ms)') 
            ax.set_ylabel('mTRFs (U.A.)')
            ax.set_title(f'Stimulus: {stimulus} - Band: {band}', fontsize=14)
            ax.grid(True)
            
            ax.legend(
                loc='upper right', 
                fontsize=12, 
                frameon=False
            )
            fig.savefig(
                rf'figures\analysis\average_trfs_by_norm\{stimulus}_{band}.png',
                bbox_inches='tight',
                dpi=600
            )
        else:
            average_channels_trfs_normalized = trfs.mean(
                axis=(0)
            ).mean(axis=1)
            feat_weights = trfs.mean(
                axis=(0)
            ).mean(axis=0)
            
            fig = plt.figure(
                figsize=(8, 8),
                constrained_layout=True
            )
            
            ax_chan = plt.subplot(211)
            evoked = mne.EvokedArray(
                data=average_channels_trfs_normalized, 
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
                axes=ax_chan,
            )
            for txt in fig.findobj(mtext.Text):
                if "ave" in txt.get_text():
                        txt.remove()
            ax_chan.plot(
                config.times*1e3, #ms
                evoked._data.mean(axis=0),
                zorder=130,
                linewidth=1.2,
                label='mean',
                color='black'
            )
            ax_chan.set_xlabel('Time (ms)') 
            ax_chan.set_ylabel('mTRFs (U.A.)')
            ax_chan.set_title(f'Stimulus: {stimulus} - Band: {band}', fontsize=14)
            ax_chan.grid(True)
            
            ax_chan.legend(
                loc='upper right', 
                fontsize=12, 
                frameon=False
            )

            ax_feat = plt.subplot(212)
            
            order, null_indexes = clustering_by_correlation(weights=feat_weights)
            feat_weights = feat_weights[order]

            im = ax_feat.pcolormesh(
                config.times[:] * 1e3,
                np.arange(feat_weights.shape[0]),
                feat_weights[:, :],
                cmap='RdBu_r',
                shading='auto',
                vmin=-np.abs(feat_weights).max(),
                vmax=np.abs(feat_weights).max()
                )

            # Set figure configuration
            # tags = config.Exp_info().phonemes_phonet
            # tags.remove('/sil/')
            # ticks = np.arange(feat_weights.shape[0])
            # tags = tags if order is None else [tags[i] for i in order]

            # ax_feat.set(
            #     xlabel='Tiempo (ms)',
            #     xticks=[0, 100, 200, 300, 400, 500],
            #     xticklabels=[0, 100, 200, 300, 400, 500],
            #     xlim=(5,550),
            #     ylabel='Fonemas',
            #     yticks=ticks,
            #     yticklabels=tags,
            #     )
            # ax_feat.set_xlabel('Tiempo (ms)', fontsize=14)
            ax_feat.set_xlabel('', fontsize=14)
            ax_feat.xaxis.labelpad = 0

            ax_feat.set_ylabel(f'{stimulus}', fontsize=14)
            # ax_feat.set_xticks([0, 100, 200, 300, 400, 500], labels = [0, 100, 200, 300, 400, 500], fontsize=14)
            ax_feat.set_xticks([0, 100, 200, 300, 400, 500], labels = ["","","", "", "", ""], fontsize=14)
            ax_feat.set_xlim(5,550)
            # ax_feat.set_yticks(ticks, labels =tags, fontsize=14)

            # Configure colorbar
            cbar = fig.colorbar(
                im,
                ax=ax_feat,
                orientation='horizontal',
                shrink=1,
                fraction=.075,
                aspect=20
                )
            cbar.set_label(label='Amplitud (U.A)',size=14)
            cbar.ax.tick_params(labelsize=14)
            
            fig.savefig(
                rf'figures\analysis\average_trfs_by_norm\{stimulus}_{band}.png',
                bbox_inches='tight',
                dpi=600
            )
print("\n\nFiles not found:")
for stimulus, band in files_not_found:
    print(f"Stimulus: {stimulus}, Band: {band}")