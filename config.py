import numpy as np

# ==========================================
# SESSIONS, STIMULI, SITUATION AND EEG BANDS
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
stimuli = [
            'Envelope', 
           ] # ['Pitch-Log-Raw', 'Envelope', 'Mfccs-Deltas', 'Spectrogram', 'Phonemes-Discrete-Phonet', 'Phonological']
situation = 'External' # 'External' #'External' # 'Internal' # 'External_BS' #'Internal_BS'
bands = [
        'Delta', 
        'Theta', 
        'Alpha', 
        'Beta1', 
        'Beta2'
        ] # ['Delta', 'Theta', 'Alpha', 'Beta1', 'Beta2']

# ====================================================================
# TFCE, T-TEST PARAMETERS, HIERARCHICAL_CLUSTERING and NUMBER OF FOLDS
perform_tfce, n_permutations, significance, number_of_jobs = False, 2500, .05, -1
hierarchical_clustering = True
n_folds = 5 # with 5 folds (remain 20% as validation set, then interchange to cross validate)

# ==========================================
# LOADING/SAVING DATA, FIGURE CONFIGURATIONS
praat_executable_path = r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"#
display_interactive_mode, save_results, save_figures = False, True, True
just_load_data = False
no_figures = False

# =====================
# VALIDATION PARAMETERS
min_order, max_order, steps = -1, 6, 32 
val_correlation_limit_percentage = 0.01
alphas_swept = np.logspace(min_order, max_order, steps)
alpha_step = np.diff(np.log(alphas_swept))[0]
save_alphas = True

# =======================
# RANDOM PERMUTATION TEST
random_permutations = 200
correlation_length_samples = 104
power_n_bootstrap_samples = 1000

# ================
# STATISTICAL TEST
significance_threshold = 0.05/128 # Bonferroni correction (the test is in # channels)
statistical_test = False

# ==========================================
# MODEL AND NORMALIZATION OF STIMULI AND EEG
model, estimator = 'mtrf', 'time_delaying_ridge' # ridge or time_delaying_ridge
stims_preprocess, eeg_preprocess = 'Normalize', 'Standarize'

# ==============================
# DEFAULT PENALIZATION PARAMETER 
correlation_limit_percentage, default_alpha, set_alpha = 0.01, 400, None

# =========================
# EEG SAMPLE RATE AND TIMES
tmin, tmax, sr = -.2, .6, 128
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)