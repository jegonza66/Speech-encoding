"""
Generate and save correlation matrix heatmaps for different stimuli under specified situations.
"""

from pathlib import Path
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from load import load_data
import config

# Configuration --> It's invariant to band, because stimuli is the same in all bands
band = 'Theta' 
# situation = ['Internal', 'External']
situation = 'External'
stimuli = [
    'Envelope', 
    'Pitch-Log-Raw', 
    'Spectrogram', 
    'Phonemes-Discrete', 
    'Phonological'
]

# Mapping for stimulus labels
stimulus_labels = {
    'Envelope': 'Env',
    'Pitch-Log-Raw': 'Pitch', 
    'Spectrogram': 'Spec',
    'Phonemes-Discrete': 'Phon',
    'Phonological': 'PhonFeat'
}

# Load and concatenate all stimulus data
preprocessed_data_path = f'saves/preprocessed_data/{situation}/tmin{config.tmin}_tmax{config.tmax}/'

correlation_matrix = []
for session in config.sessions:
    feature_names = []
    all_features = []

    for stimulus in stimuli:
        subject_1, subject_2, _ = load_data(
            preprocessed_data_path=preprocessed_data_path,
            situation=situation,
            session=session,
            stimuli=stimulus,
            band=band
        )

        # Combine both subjects
        stimulus_session = np.vstack([subject_1[stimulus], subject_2[stimulus]])
                
        # Add each dimension
        for dim in range(stimulus_session.shape[1]):
            all_features.append(stimulus_session[:, dim])
            feature_names.append(f"{stimulus_labels[stimulus]}_{dim+1}")

    # Compute correlation matrix
    feature_matrix = np.column_stack(all_features)
    correlation_matrix.append(
        np.corrcoef(
            feature_matrix, 
            rowvar=False
        )
    )

# Average correlation matrix across sessions
correlation_matrix = np.mean(
    correlation_matrix, 
    axis=0
)

# Plot
plt.figure(figsize=(9, 9))
sns.heatmap(
    correlation_matrix, 
    xticklabels=feature_names, 
    yticklabels=feature_names,
    cmap='RdBu_r', 
    center=0, 
    fmt='.2f',
    vmin=-1, vmax=1,
    square=True,
    cbar_kws={'label': 'Correlation'}
)
cbar = plt.gca().collections[0].colorbar
cbar.set_label('Correlation', fontsize=12)
cbar.ax.tick_params(labelsize=12)

plt.title(f'Stimulus correlation matrix - {situation}', fontsize=16)
plt.xticks(rotation=90, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

# Save
output_dir = Path('figures/analysis/correlation_matrix_stimuli/')
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir/f'{situation}.png', 
    bbox_inches='tight',
    dpi=300 
)
# plt.show(block=False)