# Standard libraries
from datetime import datetime
import os, numpy as np

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from funciones  import load_pickle, dump_pickle, dict_to_csv, iteration_percentage
from mtrf_models import Receptive_field_adaptation
from load import load_data
from processing import tfce
from setup import exp_info
import plot

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542
     
start_time = datetime.now()

# ==========
# PARAMETERS
# ==========

# Stimuli, EEG frecuency band and dialogue situation
# stimuli = ['Mfccs', 'Mfccs-Deltas', 'Mfccs-Deltas-Deltas', 'Envelope']
#stimuli = ['Phonological', 'Deltas', 'Spectrogram']
# stimuli = ['Phonemes-Discrete-Manual', 'Phonemes-Onset-Manual', 'Phonemes-Envelope-Manual']
# stimuli = ['Pitch-Raw', 'Pitch-Log-Raw', 'Pitch-Log-Manual', 'Pitch-Log-Phonemes', 'Pitch-Log-Quad']

# stimuli = ['Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw']
# stimuli = ['Envelope_Phonemes-Discrete-Manual', 'Envelope_Phonemes-Onset-Manual', 'Envelope_Phonemes-Discrete-Manual']
# stimuli = ['Deltas_Phonological', 'Deltas_Spectrogram', 'Phonological_Spectrogram', 'Deltas_Phonological_Spectrogram',]
# stimuli = ['Phonemes-Discrete-Manual_Pitch-Log-Raw_Envelope', 'Phonemes-Discrete-Manual_Pitch-Log-Raw', 'Envelope_Pitch-Log-Raw','Envelope_Phonemes-Discrete-Manual', 'Envelope_Phonemes-Onset-Manual', 'Envelope_Phonemes-Discrete-Manual']
stimuli =  ['Envelope']
bands = ['Theta'] #, 'Delta', 'Alpha', 'Beta1', 'Beta2', 'All', 'Delta_Theta', 'Alpha_Delta_Theta']
situation = 'External_BS' #'Internal_BS' #'External' # 'Internal' # 'External_BS'
# situation = 'External' 
# stimuli = ['Pitch-Log-Raw']
# Run setup
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

# EEG sample rate
sr = 128

# Run times
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# Save / display Figures
display_interactive_mode = False
save_figures = True 
save_results = True

no_figures = False

# Include random permutations analaysis
statistical_test = False
umbral = 0.05/128 # TODO que onda con features no unidimensionales

# TFCE number of permutations
n_permutations = 1024#4096

# Model and normalization of input
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'
model = 'mtrf'
estimator = 'time_delaying_ridge' # ridge or time_delaying_ridge

# Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
n_folds = 5

# Preset alpha (penalization parameter)
set_alpha = None
default_alpha = 400
correlation_limit_percentage = 0.01

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
        
        # Relevant paths
        save_results_path = f'saves/{model}/{situation}/correlations/tmin{tmin}_tmax{tmax}/{band}/'
        preprocessed_data_path = f'saves/preprocessed_data/{situation}/tmin{tmin}_tmax{tmax}/{band}/'
        path_weights = f'saves/{model}/{situation}/weights/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
        path_null = f'saves/{model}/{situation}/null/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
        path_figures = f'figures/{model}/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/{band}/{stim}/'
        prat_executable_path = r"C:\Users\User\Downloads\programas_descargados_por_octavio\Praat.exe" #r"C:\Program Files\Praat\Praat.exe"
        alphas_directory = os.path.normpath(f'saves/alphas/{situation}/stims_{stims_preprocess}/EEG_{eeg_preprocess}//tmin{tmin}_tmax{tmax}/{band}/{stim}/') 
        alphas_path = os.path.join(alphas_directory, f'corr_limit_{correlation_limit_percentage}.pkl')
        
        # Make lists to store relevant data across sobjects
        average_weights_subjects = []
        average_correlation_subjects = []
        average_rmse_subjects = []
        pvalues_corr_subjects = []
        pvalues_rmse_subjects = []
        repeated_good_correlation_channels_subjects = []
        repeated_good_rmse_channels_subjects = []
        phoenemes_ocurrences = {sesion:{} for sesion in sesiones}

        # Iterate over sessions
        for sesion in sesiones:
            print(f'\n------->\tStart of session {sesion}\n')
            
            # Load data by subject, EEG and info
            sujeto_1, sujeto_2, samples_info = load_data(sesion=sesion,
                                                         stim=stim,
                                                         band=band,
                                                         sr=sr,
                                                         delays=delays,
                                                         preprocessed_data_path=preprocessed_data_path,
                                                         praat_executable_path=prat_executable_path,
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
            
            # Store phonemes ocurrences to make boxplot
            for stimulus in stim.split('_'):
                if stimulus.startswith('Phonemes'):
                    # Change to 1's every value that isn't 0. In this way the method works for every kind
                    matrix_1 = sujeto_1[stimulus].copy()
                    matrix_1[matrix_1!=0.] = 1
                    matrix_2 = sujeto_2[stimulus].copy()
                    matrix_2[matrix_2!=0.] = 1
                    matrix = matrix_1 + matrix_2
                    
                    # Identify the phonemes
                    phonemes = exp_info()
                    if stimulus.endswith('Manual'):
                        phonemes = phonemes.ph_labels_man
                    else:
                        phonemes = phonemes.ph_labels
                    phoenemes_ocurrences[sesion][stimulus] = {'phonemes':phonemes, 'count':np.sum(matrix, axis=0)}
                else:
                    pass

            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()

            # Initialize empty variables to store relevant data of each fold 
            weights_per_fold = np.zeros((n_folds, info['nchan'], np.sum(n_feats), len(delays)), dtype=np.float16)
            correlation_per_channel = np.zeros((n_folds, info['nchan']))
            rmse_per_channel = np.zeros((n_folds, info['nchan']))

            # Variable to store all channel's p-value
            topo_pvalues_corr = np.zeros((n_folds, info['nchan']))
            topo_pvalues_rmse = np.zeros((n_folds, info['nchan']))

            # Variable to store p-value of significant channels
            proba_correlation_per_channel = np.ones((n_folds, info['nchan']))
            proba_rmse_per_channel = np.ones((n_folds, info['nchan']))

            # Variable to store significant channels
            repeated_good_correlation_channels = np.zeros(info['nchan'])
            repeated_good_rmse_channels = np.zeros(info['nchan'])
        
            # Run model for each subject
            for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\t······  Running model for Subject {sujeto}\n')
                
                # Set alpha for specific subject
                if set_alpha is None:
                    try:
                        alphas = load_pickle(path=alphas_path)
                        alpha = alphas[sesion][sujeto]
                    except:
                        alpha = default_alpha
                else:
                    alpha = set_alpha

                # Make the Kfold test
                kf_test = KFold(n_folds, shuffle=False)

                # Keep relevant indexes for eeg
                relevant_eeg = eeg[relevant_indexes]

                for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                    print(f'\n\t······  [{fold+1}/{n_folds}]')
                    
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
                        estimator=estimator)
                    
                    # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
                    mtrf.fit(stims, eeg)
                    
                    # Get weights coefficients shape n_chans, feats, delays
                    weights_per_fold[fold] = mtrf.coefs

                    # Predict and save
                    predicted, eeg_test = mtrf.predict(stims)

                    # Calculates and saves correlation of each channel
                    correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
                    correlation_per_channel[fold] = correlation_matrix

                    # Calculates and saves root mean square error of each channel
                    root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                    rmse_per_channel[fold] = root_mean_square_error
                    
                    # Perform statistical test
                    if statistical_test:
                        null_correlation_per_channel, null_errors = load_pickle(path=path_null + f'Corr_Rmse_fake_Sesion{sesion}_Sujeto{sujeto}.pkl')
                        iterations =  null_correlation_per_channel.shape[1]
                        
                        # Correlation and RMSE
                        null_correlation_matrix = null_correlation_per_channel[fold]
                        null_root_mean_square_error = null_errors[fold]

                        # p-values for both tests
                        p_corr = ((null_correlation_matrix > correlation_matrix).sum(0) + 1) / (iterations + 1)
                        p_rmse = ((null_root_mean_square_error < root_mean_square_error).sum(0) + 1) / (iterations + 1)

                        # Threshold
                        proba_correlation_per_channel[fold][p_corr < umbral] = p_corr[p_corr < umbral]
                        proba_rmse_per_channel[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]
                        
                        # p-value topographic distribution
                        topo_pvalues_corr[fold] = p_corr
                        topo_pvalues_rmse[fold] = p_rmse

                print(f'\n\t······  Run model\n')

                # Take average weights, correlation and RMSE between folds of all channels
                average_weights = weights_per_fold.mean(axis=0) # info['nchan'], np.sum(n_feats), len(delays)
                average_correlation = correlation_per_channel.mean(axis=0)
                average_rmse = rmse_per_channel.mean(axis=0)
                
                # Channels that pass the tests
                corr_good_channel_indexes = []
                rmse_good_channel_indexes = []

                if statistical_test:
                    # Correlation and RMSE of channels that pass the test
                    corr_good_channel_indexes, = np.where(np.all((proba_correlation_per_channel < 1), axis=0))
                    rmse_good_channel_indexes, = np.where(np.all((proba_rmse_per_channel < 1), axis=0))

                    # Saves passing channels by subject
                    repeated_good_correlation_channels[corr_good_channel_indexes] += 1
                    repeated_good_rmse_channels[rmse_good_channel_indexes] += 1

                    # Plot shadows
                    plot.null_correlation_vs_correlation_good_channels(display_interactive_mode=display_interactive_mode, session=sesion, subject=sujeto,
                                              save_path=path_figures, good_channels_indexes=corr_good_channel_indexes, average_correlation=average_correlation, 
                                              save=save_figures, correlation_per_channel=correlation_per_channel, 
                                              null_correlation_per_channel=null_correlation_per_channel, no_figures=no_figures)

                # Adapt to yield average p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(axis=0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(axis=0)

                # Plot head topomap across al channel for correlation and rmse 
                plot.topomap(good_channels_indexes=corr_good_channel_indexes, average_coefficient=average_correlation, info=info,
                             coefficient_name='Correlation', save=save_figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, subject=sujeto, session=sesion, no_figures=no_figures)
                plot.topomap(good_channels_indexes=rmse_good_channel_indexes, average_coefficient=average_rmse, info=info,
                             coefficient_name='RMSE', save=save_figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, subject=sujeto, session=sesion, no_figures=no_figures)

                # Plot weights
                plot.channel_weights(info=info, save=save_figures, save_path=path_figures, average_correlation=average_correlation,
                                     average_rmse=average_rmse, best_alpha=alpha, average_weights=average_weights, times=times, 
                                     n_feats=n_feats, stim=stim, session=sesion, subject=sujeto,
                                     display_interactive_mode=display_interactive_mode, no_figures=no_figures)
                
                # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                average_weights_subjects.append(average_weights)
                average_correlation_subjects.append(average_correlation)
                average_rmse_subjects.append(average_rmse)
                pvalues_corr_subjects.append(topo_pval_corr_sujeto)
                pvalues_rmse_subjects.append(topo_pval_rmse_sujeto)
                repeated_good_correlation_channels_subjects.append(repeated_good_correlation_channels)
                repeated_good_rmse_channels_subjects.append(repeated_good_rmse_channels)
            
            # Print the progress of the iteration
            iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=sesiones.index(sesion), length_of_iterator=len(sesiones))
            
            # del average_weights, average_rmse, average_correlation, correlation_per_channel, rmse_per_channel, correlation_matrix, root_mean_square_error,\
            #     eeg_test, eeg, stims, stims_sujeto_1, stims_sujeto_2, sujeto_1, sujeto_2, eeg_sujeto_1, eeg_sujeto_2
        
        if just_load_data:
            continue

        # Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
        average_weights_subjects = np.stack(average_weights_subjects, axis=0)
        average_correlation_subjects = np.stack(average_correlation_subjects , axis=0)
        average_rmse_subjects = np.stack(average_rmse_subjects , axis=0)
        pvalues_corr_subjects = np.stack(pvalues_corr_subjects , axis=0)
        pvalues_rmse_subjects = np.stack(pvalues_rmse_subjects , axis=0)
        repeated_good_correlation_channels_subjects = np.stack(repeated_good_correlation_channels_subjects , axis=0)
        repeated_good_rmse_channels_subjects = np.stack(repeated_good_rmse_channels_subjects , axis=0)
        
        # Plot phoneme ocurrences
        if np.array([bool(d) for d in phoenemes_ocurrences.values()]).any():
            plot.phonemes_ocurrences(ocurrences=phoenemes_ocurrences, save_path=path_figures, save=save_figures, no_figures=no_figures)

        # Plot average topomap across each subject
        plot.average_topomap(average_coefficient_subjects=average_rmse_subjects, stim=stim, info=info, display_interactive_mode=display_interactive_mode,
                             save=save_figures, save_path=path_figures, coefficient_name='RMSE', no_figures=no_figures)
        plot.average_topomap(average_coefficient_subjects=average_correlation_subjects, stim=stim, display_interactive_mode=display_interactive_mode,
                             info=info, save=save_figures, save_path=path_figures, coefficient_name='Correlation', test_result=False, no_figures=no_figures) # USING ZERO METHOD PRATT
        
        # Plot topomap with relevant times
        plot.topo_map_relevant_times(average_weights_subjects=average_weights_subjects, info=info, n_feats=n_feats, band=band, stim=stim, times=times, 
                                sample_rate=sr, save_path=path_figures, save=save_figures, display_interactive_mode=display_interactive_mode, no_figures=no_figures)

        # Plot channel-wise correlation topomap 
        plot.channel_wise_correlation_topomap(average_weights_subjects=average_weights_subjects, info=info, stim=stim, save=save_figures, 
                                              save_path=path_figures, display_interactive_mode=display_interactive_mode, no_figures=no_figures)      

        # Plot weights
        plot.average_regression_weights(average_weights_subjects=average_weights_subjects, info=info, save=save_figures, save_path=path_figures, 
                                        times=times, n_feats=n_feats, stim=stim, display_interactive_mode=display_interactive_mode, no_figures=no_figures)
        
        # Plot correlation matrix between subjects 
        plot.correlation_matrix_subjects(average_weights_subjects=average_weights_subjects, stim=stim, n_feats=n_feats, save=save_figures, 
                                         save_path=path_figures, display_interactive_mode=display_interactive_mode, no_figures=no_figures)

        if statistical_test: 
            # Plot topomap of average p-values across all subject
            plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_corr_subjects, info=info, display_interactive_mode=display_interactive_mode,
                                   save=save_figures, save_path=path_figures, coefficient_name='correlation', no_figures=no_figures)
            plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_rmse_subjects, info=info, display_interactive_mode=display_interactive_mode,
                                   save=save_figures, save_path=path_figures, coefficient_name='RMSE', no_figures=no_figures)
            
            # Plot topomap of sum of repeated channels across all subject
            plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_correlation_channels_subjects, 
                                        info=info, display_interactive_mode=display_interactive_mode, save=save_figures, 
                                        save_path=path_figures, coefficient_name='correlation', no_figures=no_figures)
            plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_rmse_channels_subjects, 
                                        info=info, display_interactive_mode=display_interactive_mode, save=save_figures, 
                                        save_path=path_figures, coefficient_name='RMSE', no_figures=no_figures)

        # TFCE across subjects 
        tvalue_tfce, pvalue_tfce, trf_subjects_shape = tfce(average_weights_subjects=average_weights_subjects, n_jobs=1, n_permutations=n_permutations)

        # Plot t and p values
        plot.plot_tvalue_pvalue_tfce(tvalue=tvalue_tfce, pvalue=pvalue_tfce, trf_subjects_shape=trf_subjects_shape, times=times, 
                                     band=band, stim=stim, n_feats=n_feats, info=info, pval_tresh=.05, save_path=path_figures, 
                                     display_interactive_mode=display_interactive_mode, save=save_figures, no_figures=no_figures)
        
        plot.plot_pvalue_tfce(average_weights_subjects=average_weights_subjects, pvalue=pvalue_tfce, times=times, info=info,
                              trf_subjects_shape=trf_subjects_shape, n_feats=n_feats, band=band, stim=stim, pval_tresh=.05, 
                              save_path=path_figures, display_interactive_mode=display_interactive_mode, save=save_figures, 
                              no_figures=no_figures)
                
        # Save results
        if save_results:
            os.makedirs(save_results_path, exist_ok=True)
            os.makedirs(path_weights, exist_ok=True)
            dump_pickle(path=save_results_path+f'{stim}.pkl', 
                        obj={'average_correlation_subjects':average_correlation_subjects,
                            'repeated_good_correlation_channels_subjects':repeated_good_correlation_channels_subjects},
                        rewrite=True)
            dump_pickle(path=path_weights+'total_weights_per_subject.pkl', 
                        obj={'average_weights_subjects':average_weights_subjects},
                        rewrite=True)
            
# Get run time            
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = f'PARAMETERS  \nModel: ' + model +f'\nBands: {bands}'+'\nStimuli: ' + f'{stimuli}'+'\nStatus: ' +situation+f'\nTime interval: ({tmin},{tmax})s'
if just_load_data:
    text += '\n\n\tJUST LOADING DATA'
text += f'\n\n\t\t RUN TIME \n\n\t\t{run_time} hours'
print(text)

# Dump metadata
metadata_path = f'saves/log/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
os.makedirs(metadata_path, exist_ok=True)
metadata = {'stimuli':stimuli, 'bands':bands, 'situation':situation, 
            'sesiones':sesiones, 'sr':sr, 'tmin':tmin, 'tmax':tmax, 
            'save_figures':save_figures, 'save_results':save_results, 'tfce_permutations':n_permutations,
            'umbral':umbral, 'no_figures':no_figures, 'statistical_test':statistical_test, 
            'stims_preprocess':stims_preprocess, 'eeg_preprocess':eeg_preprocess, 'model':model, 
            'estimator':estimator, 'n_folds':n_folds, 'default_alpha':default_alpha,
            'save_results_path':save_results_path, 'preprocessed_data_path':preprocessed_data_path,
            'path_original':path_weights, 'path_null':path_null, 'path_figures':path_figures,
            'prat_executable_path':prat_executable_path, 'metadata_path': metadata_path,
            'date': str(datetime.now()), 'run_time':str(run_time), 'just_loading_data':just_load_data}
dict_to_csv(path=metadata_path+'metadata.csv', 
            obj=metadata,
            rewrite=True)

# Send text to telegram bot
mensaje_tel(api_token=api_token,chat_id=chat_id, mensaje=text)