# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones  import load_pickle, dump_pickle, dict_to_csv, iteration_percentage
from mtrf_models import Receptive_field_adaptation
from plot import hyperparameter_selection
from load import load_data

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542
     
start_time = datetime.now()

# ==========
# PARAMETERS
# ==========

# Run setup
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

# EEG sample rate
sr = 128

# Run times
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# Stimuli, EEG frecuency band and dialogue situation
# Stimuli ranked according to corrrelation
# Phonological_Spectrogram','Phonological_Deltas','Phonological_Deltas_Spectrograma','Phonological',
# 'Deltas','Deltas_Spectrogram','Spectrogram','Mfccs_Deltas','Phonemes-Onset-Manual_Envelope','Envelope','Pitch-Log-Raw','Pitch-Log-Raw_Phoneme'

stimuli = ['Envelope', 'Spectrogram', 'Deltas', 'Phonological', 'Mfccs', 'Mfccs-Deltas', 'Phonological_Spectrogram','Phonological_Deltas','Phonological_Deltas_Spectrogram']
stimuli = ['Pitch-Log-Raw','Phonemes-Discrete-Manual','Phonemes-Onset-Manual']
stimuli = ['Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw', 'Envelope_Phonemes-Onset-Manual', 'Envelope_Phonemes-Discrete-Manual']
# excluding pitch-log-raw because of its sparsity in External_BS

stimuli = ['Envelope', 'Spectrogram', 'Deltas', 'Phonological', 'Mfccs', \
           'Mfccs-Deltas', 'Phonological_Spectrogram','Phonological_Deltas',\
           'Phonological_Deltas_Spectrogram','Pitch-Log-Raw','Phonemes-Discrete-Manual',\
           'Phonemes-Onset-Manual','Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', \
           'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw', \
           'Envelope_Phonemes-Onset-Manual', 'Envelope_Phonemes-Discrete-Manual']
bands = ['Theta']#
bands = ['Delta', 'Alpha', 'Beta1', 'Beta2']
situation = 'External' #'External' 'External_BS' 'Internal_BS' 'Internal'

# Model, estimator and normalization of input
estimator = 'time_delaying_ridge'
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
model = 'mtrf'

# Save figures
no_figures = False
save_figures = True
save_alphas = True

# Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
n_folds = 5

# Correlation limit percentage to select alpha
correlation_limit_percentage = 0.01

# Make gridsearch sweep
min_order, max_order, steps = -1, 6, 32 #
alphas_swept = np.logspace(min_order, max_order, steps)
alpha_step = np.diff(np.log(alphas_swept))[0]

# ============
# RUN ANALYSIS
# ============
just_load_data = False

for band in bands:
    for stim in stimuli:
        ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)
        
        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + situation+'\n',f'Time interval: ({tmin},{tmax})s\n','\n===========================\n')
        
        # Create relevant paths
        preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/{situation}/tmin{tmin}_tmax{tmax}/{band}/')
        figures_path = os.path.normpath(f'figures/{model}_trace/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}')
        alphas_directory = os.path.normpath(f'saves/alphas/{situation}/stims_{stims_preprocess}/EEG_{eeg_preprocess}//tmin{tmin}_tmax{tmax}/{band}/{stim}/') 
        alphas_path = os.path.join(alphas_directory, f'corr_limit_{correlation_limit_percentage}.pkl')
        
        # Try to access alphas
        try:
            alphas = load_pickle(path=alphas_path)
        except:
            alphas = {s: {} for s in sesiones} 
       
        # Iterate over sessions
        for sesion in sesiones:
            print(f'\n\n------->\tStart of session {sesion}\n')

            # Load data by subject, EEG and info
            sujeto_1, sujeto_2, samples_info = load_data(sesion=sesion,
                                                         stim=stim,
                                                         band=band,
                                                         sr=sr,
                                                         delays=delays,
                                                         preprocessed_data_path=preprocessed_data_path,
                                                        #  praat_executable_path=os.path.normpath(r"C:\Program Files\Praat\Praat.exe"),
                                                         praat_executable_path=os.path.normpath(r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe"),
                                                         situation=situation,
                                                         silence_threshold=0.03)
            eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']
            
            if just_load_data:
                continue

            # Load stimuli by subject (i.e: concatenated stimuli features)
            stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
            stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])

            n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
            delayed_length_per_stimuli = [n_feat*len(delays) for n_feat in n_feats]

            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()

            for subject, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\n\t······  Running model for Subject {subject}\n')

                # Take some metrics for each alpha
                standarized_betas = np.zeros(len(alphas_swept))
                errors = np.zeros(len(alphas_swept))
                correlations = np.zeros(len(alphas_swept))
                correlations_std = np.zeros(len(alphas_swept))
                
                # Make sweep, initializing empty variables to store relevant data of each fold in each alpha
                for i_alpha, alpha in enumerate(alphas_swept):
                    weights_per_fold = np.zeros((n_folds, info['nchan'], np.sum(n_feats), len(delays)), dtype=np.float16)
                    correlation_per_channel = np.zeros((n_folds, info['nchan']))
                    rmse_per_channel = np.zeros((n_folds, info['nchan']))

                    # Make the Kfold test
                    kf_test = KFold(n_folds, shuffle=False)

                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]

                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        # print(f'\n\t\t······  [{fold+1}/{n_folds}]')

                        # Determine wether to run the model in parallel or not
                        n_jobs=-1 if sum(n_feats)>1 else 1
                        
                        # Implement mne model
                        mtrf = Receptive_field_adaptation(
                            tmin=tmin, 
                            tmax=tmax, 
                            sample_rate=sr, 
                            alpha=alpha, 
                            relevant_indexes=np.array(relevant_indexes),
                            train_indexes=train_indexes, 
                            test_indexes=test_indexes, 
                            stims_preprocess=stims_preprocess, 
                            eeg_preprocess=eeg_preprocess,
                            fit_intercept=False,
                            n_jobs=n_jobs, 
                            estimator=estimator,
                            validation=True)
                        
                        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
                        mtrf.fit(stims, eeg)
                        
                        # Get weights coefficients shape n_chans, feats, delays
                        weights_per_fold[fold] = mtrf.coefs
                        
                        # Predict and save
                        predicted, eeg_val = mtrf.predict(stims)

                        # Calculates and saves correlation of each channel
                        correlation_matrix = np.array([np.corrcoef(eeg_val[:, j], predicted[:, j])[0,1] for j in range(eeg_val.shape[1])])
                        correlation_per_channel[fold] = correlation_matrix

                        # Calculates and saves root mean square error of each channel
                        root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_val), 2).mean(0)))
                        rmse_per_channel[fold] = root_mean_square_error

                    correlations[i_alpha] = correlation_per_channel.mean()
                    correlations_std[i_alpha] = correlation_per_channel.std()
                    print(f'\r·················· Sweeping progress  {int((i_alpha + 1) * 100 / steps)}% ··················', end='')
                
                # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
                relative_difference = abs((correlations.max() - correlations)/correlations.max())
                good_indexes_range = np.where(relative_difference < correlation_limit_percentage)[0]

                # Get the very last one, because the greater the alpha, the smoothest the signal gets
                alpha_subject = alphas_swept[int(good_indexes_range[-1])]
                
                # Make the alpha selection process plot
                hyperparameter_selection(alphas_swept=alphas_swept,correlations=correlations, correlations_std=correlations_std, alpha_subject=alpha_subject,
                                         correlation_limit_percentage=correlation_limit_percentage, session=sesion, subject=subject, stim=stim, band=band, 
                                         save_path=figures_path, save=save_figures, no_figures=no_figures)

                # Update dictionary
                alphas[sesion][subject] = alpha_subject

                # Save results
                os.makedirs(name=alphas_directory, exist_ok=True)
                if save_alphas:
                    dump_pickle(path=alphas_path, obj=alphas, rewrite=True)
                
                # Print the progress of the iteration
                iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=sesiones.index(sesion), length_of_iterator=len(sesiones))

# Get run time            
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = f'PARAMETERS  \nModel: ' + model +f'\nBands: {bands}'+'\nStimuli: ' + f'{stimuli}'+'\nStatus: ' +situation+f'\nTime interval: ({tmin},{tmax})s'
if just_load_data:
    text += '\n\n\tJUST LOADING DATA'
else:
    text += '\n\n\talpha selection'
text += f'\n\n\t\t RUN TIME \n\n\t\t{run_time} hours'
print(text)

# Dump metadata
metadata_path = f'saves/log/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
os.makedirs(metadata_path, exist_ok=True)
metadata = {'stimuli':stimuli, 'bands':bands, 'situation':situation, 
            'sesiones':sesiones, 'sr':sr, 'tmin':tmin, 'tmax':tmax, 
            'save_figures':save_figures, 'no_figures':no_figures, 
            'stims_preprocess':stims_preprocess, 'eeg_preprocess':eeg_preprocess, 'model':model, 
            'estimator':estimator, 'n_folds':n_folds, 'preprocessed_data_path':preprocessed_data_path,
            'figures_path':figures_path,'alphas_directory':alphas_directory,'alphas_path':alphas_path,
            'correlation_limit_percentage':correlation_limit_percentage, 'min_order':min_order, 
            'max_order':max_order, 'steps':steps, 'alphas_swept':alphas_swept, 'alpha_step':alpha_step,
            'metadata_path': metadata_path,
            'date': str(datetime.now()), 'run_time':str(run_time), 'just_loading_data':just_load_data}
dict_to_csv(path=metadata_path+'metadata.csv', 
            obj=metadata,
            rewrite=True)

# Send text to telegram bot
mensaje_tel(api_token=api_token,chat_id=chat_id, mensaje=text)