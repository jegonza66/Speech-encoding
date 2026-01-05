import numpy as np
import mne

# General configuration
stable_version = False
parallel_load, number_of_workers = True, 9
use_gpu = True

same_validation_subjects = True # same hyperparameter for all subjects 
external_validation = False # whether to use External hyperparameter or the one that maximize specific condition
default_alpha, set_alpha = 400, None
substract_mean = True

solver = 'ridge' # "ridge-laplacian"
n_folds = 10 # with 5 folds (remain 20% as validation set, then interchange to cross validate)

statistical_test, perform_tfce = False, False

# Logging and directory configuration
LOG_LEVEL = "INFO"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
figures_dir = "figures"
output_dir = "output"
saves_dir = "saves"
save_results, save_figures = True, True
no_figures = False

# Time lags and delays
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)


ROI = False # whether to use Regions of Interest (ROI) data or full EEG data
if ROI:
    info_mne = mne.create_info(
        ch_names=['ROI1', 'ROI2', 'ROI3', 'ROI4', 'ROI5', 'ROI6', 'ROI7', 'ROI8', 'ROI9', 'ROI10', 'ROI11', 'ROI12', 'ROI13', 'ROI14', 'ROI15', 'ROI16'], 
        sfreq=sr, 
        ch_types='eeg'
    )
else:
    montage = mne.channels.make_standard_montage('biosemi128')
    relevant_channels = [f'C{i+1}' for i in range(32)] # frontal/ frontal right
    # relevant_channels += [f'D{i+1}' for i in range(13)]# frontal left
    # relevant_channels = ['C23','C2', 'A1', 'D1']
    # relevant_channels = montage.ch_names
    info_mne = mne.create_info(
        ch_names=relevant_channels, 
        sfreq=sr, 
        ch_types='eeg'
    ).set_montage(montage)
    channels_index = [info_mne.ch_names.index(ch) for ch in relevant_channels]

# =======================
# Configurable parameters
sessions = [
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    29,
    30
]
stimuli = [
    'Envelope',
    # 'Pitch-Log-Raw',
    # 'Spectrogram-21',
    # 'Phonemes-Discrete',
    # 'Phonological',
    # 'Phonemes',
    # 'Phones',
    # '21DNNs5-wavlm'
]
situations = [
    'External', # Predicts own EEG from interlocutor speech
    # 'Internal', # Predicts own EEG from own speech
    # 'External_BS', # Predicts own EEG from interlocutor speech when both speaking
    # 'Internal_BS', # Predicts own EEG from own speech when both speaking
    # 'All', # Predicts own EEG from whole recording
    # 'External_Silence_80' # Predicts own EEG from interlocutor speech adding XX% of total silence
]
bands = [
    'Unfiltered', # No filtering
    # 'All', # 1-40 Hz
    'Broad', # 1-15 Hz
    # 'Delta', # 1-4 Hz
    # 'Theta', # 4-8 Hz
    # 'Alpha', # 8-13 Hz
    # 'Beta', # 13-25 Hz
    # 'Beta1', # 13-19 Hz
    # 'Beta2', # 19-25 Hz
    # 'Gamma', # 30-45 Hz
    # 'Delta_Theta' # 1-8 Hz
]

# =================
# Static parameters

# Logging parameters
LOG_DIR = "saves/detailed_logs"
LOG_TO_FILE = False

stims_preprocess, eeg_preprocess = 'Standarize', 'Standarize'
model = 'mtrf' # 'mtrf_ridge'

# TFCE parameters
n_permutations, significance, number_of_jobs = 4096, .05, -1

# Validation parameters
val_correlation_limit_percentage = 0.01
min_order, max_order, steps, base_log = -4, 8, 48, 10
alphas_swept = np.logspace(
        min_order, 
        max_order, 
        steps, 
        base=base_log
)
alpha_step = np.diff(np.log(alphas_swept))[0]

# Permutation test and statistical parameters
random_permutations = 3000
correlation_length_samples = 104
power_n_bootstrap_samples = 1000
significance_threshold = 0.05/128 # Bonferroni correction (the test is in # channels) #TODO

# Other parameters
hierarchical_clustering = True
temporal_shift = None
praat_executable_path = r"C:\Users\jocta\Downloads\programas\Praat.exe"
figure_format = '.png'

class ExpInfo:
    def __init__(self):
        """A class used to represent experimental information for speech encoding.
        """
        self.phones = [
            '<p:>', 'B', 'D', 'F', 'G', 'J', 'L', 'N', 'S', 'T', 'Z', 'a', 'b', 'd', 'e',
            'f', 'g', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 'sil',
            't', 'tS', 'u', 'w', 'x', 'z'
        ]
        self.phonemes = [
            '/a/', '/b/', '/d/', '/e/', '/f/', '/g/', '/i/', '/k/', '/l/', '/m/', '/n/',
            '/o/', '/p/', '/r/', '/s/', '/t/', '/tS/', '/u/', '/x/', '/R/', '/L/','/sil/'
        ]
        self.phones_to_phonemes = {
            'a' : '/a/', 'e' : '/e/', 'i' : '/i/', 'o' : '/o/', 'j' : '/i/', 'w' : '/u/', 'u' : '/u/',
            'l' : '/l/', 'r' : '/R/', 'rr': '/r/', 't' : '/t/', 'd' : '/d/', 'D' : '/d/', 'sil' : '/sil/',
            '<p:>' : '/sil/', 'm' : '/m/', 'n' : '/n/', 'N' : '/n/', 'k' : '/k/', 'g' : '/g/', 'G' : '/g/',
            'tS': '/tS/', 'T' : '/tS/', 'f' : '/f/', 'F' : '/f/', 's' : '/s/', 'S' : '/s/', 'z' : '/s/',
            'Z' : '/s/', 'p' : '/p/', 'b' : '/b/', 'B' : '/b/', 'L' : '/L/', 'x' : '/x/', 'jj': '/x/', 'J' : '/x/'
        }
        self.phonological_labels = {
            "vocalic" : ["a","e","i","o","u", "w", "j"],
            "consonantal" : ["b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z"],
            "back" : ["a","o","u", "w"],
            "anterior" : ["e","i","j"],
            "open" : ["a","e","o"],
            "close" : ["j","i","u", "w"],
            "nasal" : ["m","n", "N"],
            "stop" : ["p","b", "B","t","k","g", "G","tS","d", "D"],
            "continuant" : ["f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z"],
            "lateral" :["l"],
            "flap" :["r"],
            "trill" :["rr"],
            "voice" :["a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j"],
            "strident" :["tS","f", "F","s", "Z", "T", "z",  "S"],
            "labial" :["m","p","b", "B","f", "F"],
            "dental" :["t","d", "D"],
            "velar" :["k","g", "G"],
            "pause" :  ["sil", "<p:>"]
        }
        self.phonological_labels1 = ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal']
        self.phonological_labels2 = ['dental', 'consonantal', 'velar', 'flap', 'close', 'strident', 'continuant']
exp_info = ExpInfo()

# ===========
# Old configs
leadership_kind_of_subsampling = 'optimized_trials' #'ordered_trials' #'random_trials'
tollerance = 0.1