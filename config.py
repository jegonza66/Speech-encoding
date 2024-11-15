import numpy as np

# ==========================================
# SESSIONS, STIMULI, SITUATION AND EEG BANDS
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
stimuli = ['Envelope'] # ['Pitch-Log-Raw', 'Envelope', 'Mfccs-Deltas', 'Spectrogram', 'Phonemes-Discrete-Phonet', 'Phonological']
situation = 'External' # 'External' #'External' # 'Internal' # 'External_BS' #'Internal_BS'
bands = ['Theta'] # ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']

# ====================================================================
# TFCE, T-TEST PARAMETERS, HIERARCHICAL_CLUSTERING and NUMBER OF FOLDS
perform_tfce, n_permutations, significance, number_of_jobs = False, 2500, .05, -1
hierarchical_clustering = False
n_folds = 5 # with 5 folds (remain 20% as validation set, then interchange to cross validate)

# ==========================================
# LOADING/SAVING DATA, FIGURE CONFIGURATIONS
display_interactive_mode, save_results, save_figures = False, True, True
just_load_data = False
no_figures = False

# ==========================================
# MODEL AND NORMALIZATION OF STIMULI AND EEG
model, estimator = 'mtrf', 'time_delaying_ridge' # ridge or time_delaying_ridge
stims_preprocess, eeg_preprocess = 'Normalize', 'Standarize'

# ==============================================================
# DEFAULT PENALIZATION PARAMETER AND RANDOM PERMUTATION ANALYSIS
correlation_limit_percentage, default_alpha, set_alpha = 0.01, 400, None

umbral = 0.05/128 # TODO que onda con features no unidimensionales
statistical_test = False


# =========================
# EEG SAMPLE RATE AND TIMES
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)