
from pathlib import Path
import numpy as np

from utils.general_functions import load_pickle
from utils.processing import shifted_matrix
import config

samples_info_dir = Path(r'saves\preprocessed_data\tmin-0.2_tmax0.6\samples_info')
situations = [
    'External',
    'External_BS'
]
samples_statistics = {
    situation: [] for situation in situations
}

# External and internal are complementary, when subject is speaking, interlocutor is not and vice versa, so they have same statistics
for session in config.sessions:
    samples_info = load_pickle(
        path=samples_info_dir / 'External' / f'samples_info_{session}.pkl'
    )
    samples_info_BS = load_pickle(
        path=samples_info_dir / 'External_BS' / f'samples_info_{session}.pkl'
    )
    samples_statistics['External'].append(len(samples_info['keep_indexes1']))
    samples_statistics['External'].append(len(samples_info['keep_indexes2']))
    samples_statistics['External_BS'].append(len(samples_info_BS['keep_indexes1']))
    samples_statistics['External_BS'].append(len(samples_info_BS['keep_indexes2']))
    

# Calculate and print statistics
for situation, lengths in samples_statistics.items():
    text = f"{situation}: min={np.min(lengths):.2f}, mean={np.mean(lengths):.2f}, max={np.max(lengths):.2f}"
    print(text)

    # # Plot histogram
    # plt.figure()
    # plt.hist(lengths, bins=20, color='blue', alpha=0.7)
    # plt.title(f"Histogram of {situation} Sample Lengths")
    # plt.xlabel("Length")
    # plt.ylabel("Frequency")
    # plt.grid(axis='y', alpha=0.75)
    # plt.show()
    