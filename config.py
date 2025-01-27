import numpy as np, mne

# ==========================================
# SESSIONS, STIMULI, SITUATION AND EEG BANDS
sesiones = [
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
        # 'Envelope',
        # 'Pitch-Log-Raw',
        # 'Spectrogram',
        # 'Mfccs',
        # 'Phonological',
        'Phonemes-Discrete-Phonet',
        # 'Phones-Discrete-Phonet',
        # 'Wav2vec2'
        ] # ['Pitch-Log-Raw', 'Envelope', 'Mfccs-Deltas', 'Spectrogram', 'Phonemes-Discrete-Phonet', 'Phonological']
situations = [
        'External', 
        # 'Internal', 
        # 'External_BS',
        # 'Internal_BS'
        ] # ['External' #'External' # 'Internal' # 'External_BS' #'Internal_BS']
bands = [
        # 'Delta', 
        'Theta', 
        # 'Alpha', 
        # 'Beta1', 
        # 'Beta2',
        # 'All'
        ] # ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2', 'All']

# ==========================================
# LOADING/SAVING DATA, FIGURE CONFIGURATIONS
praat_executable_path = r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"#
display_interactive_mode, save_results, save_figures, no_figures = False, True, True, False
figure_format = '.png'
just_load_data = True

# ==========================================
# MODEL AND NORMALIZATION OF STIMULI AND EEG
model = 'mtrf'
estimator = 'ridge_torch' # ridge, ridge_torch or time_delaying_ridge
statistical_test, perform_tfce, use_gpu = False, False, False
stims_preprocess, eeg_preprocess = 'Normalize', 'Standarize'

if estimator=='ridge':
	model = 'mtrf_ridge'
elif estimator == 'ridge_torch':
    model = 'mtrf_ridge_torch'
else:
    model = 'mtrf'
    
# ====================================================================
# TFCE, T-TEST PARAMETERS, HIERARCHICAL_CLUSTERING and NUMBER OF FOLDS
n_permutations, significance, number_of_jobs = 4096, .05, -1
hierarchical_clustering = True
n_folds = 5 # with 5 folds (remain 20% as validation set, then interchange to cross validate)

# =====================
# VALIDATION PARAMETERS
min_order, max_order, steps = -1, 6, 32 
val_correlation_limit_percentage = 0.01
alphas_swept = np.logspace(min_order, max_order, steps)
alpha_step = np.diff(np.log(alphas_swept))[0]
save_alphas = True

# =======================
# RANDOM PERMUTATION TEST
random_permutations = 3000
correlation_length_samples = 104
power_n_bootstrap_samples = 1000
significance_threshold = 0.05/128 # Bonferroni correction (the test is in # channels) #TODO

# ==============================
# DEFAULT PENALIZATION PARAMETER 
correlation_limit_percentage, default_alpha, set_alpha = 0.01, 400, None

# =========================
# EEG SAMPLE RATE AND TIMES
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# =================
# MODEL COMPARISON
montage = mne.channels.make_standard_montage('biosemi128')
info_mne = mne.create_info(ch_names=montage.ch_names[:], sfreq=sr, ch_types='eeg').set_montage(montage)
relevant_channels = None#12

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
			ph_labels_phonet : list
					A list of phonemes labeled using phonetic transcription.
			ph_labels_phonet_ordered : list
					An ordered list of phonemes labeled using phonetic transcription.
			mistakes : list
					A list of types of mistakes.
			control : list
					A list of control categories.
			phonological_labels : dict
					A dictionary categorizing phonemes into various phonological features.
                        
        Methods
        -------
			__init__():
					Initializes the Exp_info class with predefined phoneme labels, mistake types, control categories, and phonological features.
        """
         # Define ctf data path and files path
        self.ph_labels = ['CH', 'NY', 'R', 'a', 'b', 'd', 'e', 'f', 'g', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'x', 'y']

        self.ph_labels_man = ['(d)o', 'A', 'AH', 'CH', 'F', 'NY', 'R', 'Y', 'a', 'ap', 'b', 'br', 'c', 'chas', 'd','de', 'e', 'es', 'f', 'g', 'h', 'i', 'k', 'l', 'lg', 'm', 'n', 'ns', 'o', 'p', 'r', 's','si', 't', 'u', 'v', 'x', 'y']
        self.ph_labels_phonet = ['B', 'D', 'F', 'G', 'N', 'T', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'jj', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 'rr', 's', 't', 'tS', 'u', 'w', 'x', 'z', 'Z', 'g', 'S', 'J', 'L', 'sil', '<p:>']
        
        self.phones_to_phonemes = {
            'a' : '/a/',
            'e' : '/e/',
            'i' : '/i/',
            'o' : '/o/',
            'j' : '/i/',
            'w' : '/u/',
            'u' : '/u/',
            'l' : '/l/',
            'r' : '/R/',
            'rr': '/r/',
            't' : '/t/',
            'd' : '/d/',
            'D' : '/d/',
            'sil' : '/sil/',
            '<p:>' : '/sil/',
            'm' : '/m/',
            'n' : '/n/',
            'N' : '/n/',
            'k' : '/k/',
            'g' : '/g/',
            'G' : '/g/',
            'tS': '/tS/',
            'T' : '/tS/',
            'f' : '/f/',
            'F' : '/f/',
            's' : '/s/',
            'S' : '/s/',
            'z' : '/s/',
            'Z' : '/s/',
            'p' : '/p/',
            'b' : '/b/',
            'B' : '/b/',
            'L' : '/L/',
            'x' : '/x/',
            'jj': '/x/',
            'J' : '/x/'
            }
        self.phonemes_phonet = [phoneme for phoneme in np.unique(list(self.phones_to_phonemes.values()))]
        
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
