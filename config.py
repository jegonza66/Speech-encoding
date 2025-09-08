import numpy as np, mne
number_of_workers = 8
saves_dir = 'saves'
output_dir = 'output'
figures_dir = 'figures'

# Logging configuration
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE = True
LOG_DIR = "saves/detailed_logs"

# ==========================================
# SESSIONS, STIMULI, SITUATION AND EEG BANDS
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
        # # 'Mistakes-Separated_Control-Separated',

        # # Redundant features
        # 'Spectrogram_Mfccs',
        # 'Phones_Phonemes',

        # # Combined 1st level features
        # 'Envelope_Pitch-Log-Raw',
        # 'Envelope_Spectrogram',
        # 'Pitch-Log-Raw_Spectrogram',
        # 'Envelope_Pitch-Log-Raw_Spectrogram',

        # # Combined best performance features
        # 'Phonological_Spectrogram',
        # 'Phonemes_Spectrogram',
        # 'Phonological_Phonemes',
        # 'Phonological_Phonemes_Spectrogram',

        # # Simples
        'Envelope',
        # 'Turn',
        # 'Pitch-Log-Raw',
        # 'Spectrogram',
        # 'Mfccs',
        # 'Phonological',
        # 'Phonemes',
        # 'Phonemes-Discrete',
        # 'Phonemes-Frequency',
        # 'Phones-Discrete',
        # 'Phones',
        # # # 'Jitter', #TODO jitter y shimmer no los pudiste calcular bien
        # # # 'Shimmer',
        # 'DNNs1',
        # 'DNNs2',
        # 'DNNs3',
        # 'DNNs4',
        # 'DNNs5',
        # 'DNNs6',
        # 'DNNs7',
        # 'DNNs8',
        # 'DNNs9',
        # 'DNNs10',
        # 'DNNs11',
        # 'DNNs12',
        # 'DNNs13',
        # 'DNNs14',
        # 'DNNs15',
        # 'DNNs16', #wav2vec hasta 23, whisper 3
        # 'DNNs17',
        # 'DNNs18',
        # 'DNNs19',
        # 'DNNs20',
        # 'DNNs21',
        # 'DNNs22',
        # 'DNNs23',
        # 'Turn',
        # # Simples but non-standard
        # 'Envelope2',
        # 'Phonological1',
        # 'Phonological2'
]
situations = [
        'All',
        'External',
        'Internal',
        'External_BS', #TODO 
        'Internal_BS',
        # 'External_Silence_100',
        # 'External_Silence_10',
        # 'External_Silence_20',
        # 'External_Silence_30',
        # 'External_Silence_40',
        # 'External_Silence_50',
        # 'External_Silence_60',
        # 'External_Silence_70',
        # 'External_Silence_80',
        # 'External_Silence_90',
]
# Hizo hasta alpha de external inclusive hasta ahora
bands = [
        # 'Delta', # 1-4 Hz
        # 'Theta', # 4-8 Hz
        # 'Alpha', # 8-13 Hz
        # 'Beta', # 13-25 Hz # elegir qe se mantenha el ancho del filtro
        'Broad', # 1-15 Hz
        # 'All', # 1-40 Hz
        # 'Delta_Theta', # 1-8 Hz
        # 'Beta1', # 13-19 Hz
        # 'Beta2', # 19-25 Hz
        # 'Unfiltered' # None
]

temporal_shift = None
# ==========================================
# LOADING/SAVING DATA, FIGURE CONFIGURATIONS
praat_executable_path = r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"#
save_results, save_figures = True, True
hierarchical_clustering = True
just_load_data = False
figure_format = '.png'
display_interactive_mode, no_figures = False, False

# ==========================================
# MODEL AND NORMALIZATION OF STIMULI AND EEG
external_validation = True # whether to use External hyperparameter or the one that maximize specific condition
same_validation_subjects = True # same hyperparameter for all subjects 
statistical_test, perform_tfce = False, False
use_gpu = True

stims_preprocess, eeg_preprocess = 'Standarize', 'Standarize'
model = 'mtrf' # 'mtrf_ridge'
solver = 'ridge' # "ridge-laplacian"

# ==============================
# DEFAULT PENALIZATION PARAMETER
correlation_limit_percentage = 0.05
default_alpha, set_alpha = 400, None

# ====================================================================
# TFCE, T-TEST PARAMETERS, HIERARCHICAL_CLUSTERING and NUMBER OF FOLDS
n_permutations, significance, number_of_jobs = 4096, .05, -1
n_folds = 10 # with 5 folds (remain 20% as validation set, then interchange to cross validate)

# =====================
# VALIDATION PARAMETERS
min_order, max_order, steps, base_log = -4, 8, 48, 10
val_correlation_limit_percentage = 0.01
alphas_swept = np.logspace(
        min_order, 
        max_order, 
        steps, 
        base=base_log
)
alpha_step = np.diff(np.log(alphas_swept))[0]
save_alphas = True

# =======================
# RANDOM PERMUTATION TEST
random_permutations = 3000
correlation_length_samples = 104
power_n_bootstrap_samples = 1000
significance_threshold = 0.05/128 # Bonferroni correction (the test is in # channels) #TODO

# =========================
# EEG SAMPLE RATE AND TIMES
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# =================
# MODEL COMPARISON
montage = mne.channels.make_standard_montage('biosemi128')
info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)
relevant_channels = 12 # None

# ============
# PLOTS LABELS
class Exp_info:
    def __init__(self):
        """A class used to represent experimental information for speech encoding.

        Attributes
        ----------
        ph_labels : list
                A list of phoneme labels.
        ph_labels_man : list
                A list of manually labeled phonemes.
        ph_labels : list
                A list of phonemes labeled using phonetic transcription.
        ph_labels_ordered : list
                An ordered list of phonemes labeled using phonetic transcription.
        mistakes : list
                A list of types of mistakes.
        control : list
                A list of control categories.
        phonological_labels : dict
                A dictionary categorizing phonemes into various phonological features.
        """
        self.phones = [
                '<p:>', 'B', 'D', 'F', 'G', 'J', 'L', 'N', 'S', 'T', 'Z', 'a', 'b', 'd', 'e',
                'f', 'g', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 'sil',
                't', 'tS', 'u', 'w', 'x', 'z'
        ]        
        # self.phonemes = [phoneme for phoneme in np.unique(list(self.phones_to_phonemes.values()))]
        self.phonemes  = [
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
        
        self.mistakes = ['Articulatory', 'Lexical', 'Discursive']
        self.control = ['Articulatory', 'Lexical', 'Discursive']
        self.phonological_labels={
            "vocalic" : ["a","e","i","o","u", "w", "j"],
            "consonantal" : ["b", "B","d", "D","f", "F","k","l","m","n", "N","p","r","rr","s", "Z", "T","t","g", "G","tS","S","x", "jj", "J", "L", "z"],
            "back"        : ["a","o","u", "w"],
            "anterior"    : ["e","i","j"],
            "open"        : ["a","e","o"],
            "close"       : ["j","i","u", "w"],
            "nasal"       : ["m","n", "N"],
            "stop"        : ["p","b", "B","t","k","g", "G","tS","d", "D"],
            "continuant"  : ["f", "F","b", "B","tS","d", "D","s", "Z", "T","x", "jj", "J","g", "G","S","L","x", "jj", "J", "z"],
            "lateral"     :["l"],
            "flap"        :["r"],
            "trill"       :["rr"],
            "voice"       :["a","e","i","o","u", "w","b", "B","d", "D","l","m","n", "N","rr","g", "G","L", "j"],
            "strident"    :["tS","f", "F","s", "Z", "T", "z",  "S"],
            "labial"      :["m","p","b", "B","f", "F"],
            "dental"      :["t","d", "D"],
            "velar"       :["k","g", "G"],
            "pause"       :  ["sil", "<p:>"]
        }
        self.phonological_labels1 = ['labial', 'lateral', 'open', 'vocalic', 'back', 'voice', 'nasal']
        self.phonological_labels2 = ['dental', 'consonantal', 'velar', 'flap', 'close', 'strident', 'continuant']

# =============================
# INSTANTIATE EXPERIMENTAL INFO
exp_info = Exp_info()

# ===========
# OLD CONFIGS

leadership_kind_of_subsampling = 'optimized_trials' #'ordered_trials' #'random_trials'
tollerance = 0.1