# Standard libraries
import os, pickle, numpy as np
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
import Load, Models, Plot, Statistics, Funciones

# Notofication bot
from labos.notificacion_bot import mensaje_tel
api_token, chat_id = '5448153732:AAGhKraJQquEqMfpD3cb4rnTcrKB6U1ViMA', 1034347542
     
start_time = datetime.now()

# ==========
# PARAMETERS
# ==========

# Preset alpha (penalization parameter)
set_alpha = None
default_alpha = 1000
alpha_correlation_limit = 0.01
alphas_fname = f'saves/alphas/alphas_Corr{alpha_correlation_limit}.pkl'
try:
    f = open(alphas_fname, 'rb')
    alphas = pickle.load(f)
    f.close()
    del f
except:
    print('\n\nAlphas file not found.\n\n')

# Save / Display Figures
display_interactive_mode = False
save_figures = True 
save_results = True

# Random permutations
perform_statistical_test = False

# Standarization
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'

# Stimuli and EEG frecuency band
stims = ['Envelope']#,'Spectrogram']
bands = ['Theta']

# Dialogue situation
situation = 'Escucha'

# Model parameters ('Ridge' or 'mtrf') #TODO va a ser mtrf de ahora en más
model = 'mtrf'

# Run setup
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]

# EEG sample rate
sr = 128

# Run times
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# ============
# RUN ANALYSIS
# ============
just_load_data = False

for band in bands:
    for stim in stims:
        ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
        stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)

        # Update
        print('\n===========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','Band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + situation+'\n',f'Time interval: ({tmin},{tmax})s\n','\n===========================\n')
        
        # Relevant paths
        save_path = f'saves/{model}/{situation}/Final_Correlation/tmin{tmin}_tmax{tmax}/'
        procesed_data_path = f'saves/{model}/Preprocesed_Data/tmin{tmin}_tmax{tmax}/'
        path_original = f'saves/{model}/{situation}/Original/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'
        path_null = f'saves/{model}/{situation}/Fake_it/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'
        path_figures = f'figures/{model}/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'

        # Make lists to store relevant data across sobjects
        average_weights_subjects = []
        average_correlation_subjects = []
        average_rmse_subjects = []
        pvalues_corr_subjects = []
        pvalues_rmse_subjects = []
        repeated_good_correlation_channels_subjects = []
        repeated_good_rmse_channels_subjects = []

        # Iterate over sessions
        for sesion in sesiones:
            print(f'\n------->\tStart of session {sesion}\n')
            
            # Load data by subject, EEG and info
            sujeto_1, sujeto_2, samples_info = Load.Load_Data(sesion=sesion, 
                                                stim=stim, 
                                                band=band, 
                                                sr=sr,
                                                delays=delays,
                                                procesed_data_path=procesed_data_path, 
                                                situation=situation,
                                                SilenceThreshold=0.03)
            eeg_sujeto_1, eeg_sujeto_2, info = sujeto_1['EEG'], sujeto_2['EEG'], sujeto_1['info']

            if just_load_data:
                continue

            # Load stimuli by subject (i.e: concatenated stimuli features)
            stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
            stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])
            n_feats = [sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]

            # This used to be called len_estimulos and was calculated in another way but at the end it was this  
            delayed_length_per_stimuli = [len(delays)*sujeto_1[stimulus].shape[1] for stimulus in stim.split('_')]
                        
            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()
            
            # Run model for each subject
            for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\t······  Running model for Subject {sujeto}\n')
                
                # Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
                n_folds, iterations = 5, 3000

                # Initialize empty variables to store relevant data of each fold 
                weights_per_fold = np.zeros((n_folds, info['nchan'], np.sum(n_feats), len(delays)), dtype=np.float16)
                correlation_per_channel = np.zeros((n_folds, info['nchan']))
                rmse_per_channel = np.zeros((n_folds, info['nchan']))

                if perform_statistical_test:
                    null_weights = np.zeros((n_folds, iterations, info['nchan'], sum(delayed_length_per_stimuli)), dtype=np.float16)
                    null_correlation_per_channel = np.zeros((n_folds, iterations, info['nchan']))
                    null_errors = np.zeros((n_folds, iterations, info['nchan']))

                # Variable to store all channel's p-value
                topo_pvalues_corr = np.zeros((n_folds, info['nchan']))
                topo_pvalues_rmse = np.zeros((n_folds, info['nchan']))

                # Variable to store p-value of significant channels
                proba_correlation_per_channel = np.ones((n_folds, info['nchan']))
                proba_rmse_per_channel = np.ones((n_folds, info['nchan']))

                # Variable to store significant channels
                repeated_good_correlation_channels = np.zeros(info['nchan'])
                repeated_good_rmse_channels = np.zeros(info['nchan'])

                # Set alpha for specific subject
                if set_alpha is None:
                    try:
                        alpha = alphas[band][stim][sesion][sujeto]
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

                    # Implement mne model
                    mtrf = Models.MNE_MTRF(
                        tmin=tmin, 
                        tmax=tmax, 
                        sample_rate=sr, 
                        alpha=alpha, 
                        relevant_indexes=np.array(relevant_indexes),
                        train_indexes=train_indexes, 
                        test_indexes=test_indexes, 
                        stims_preprocess=stims_preprocess, 
                        eeg_preprocess=eeg_preprocess,
                        fit_intercept=False)
                    
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
                    
                    # Perform statistical test #TODO porque se llaman fake? Jamas se crean en ninguno archivo estos datos, donde los saco?
                    if perform_statistical_test:
                        try:
                            null_correlation_per_channel, null_errors = Funciones.load_pickle(path=path_null + f'Corr_Rmse_fake_Sesion{sesion}_Sujeto{sujeto}.pkl')
                        except:
                            perform_statistical_test = False

                        # Correlation and RMSE
                        null_correlation_matrix = null_correlation_per_channel[fold]
                        null_root_mean_square_error = null_errors[fold]

                        # p-values for both tests
                        p_corr = ((null_correlation_matrix > correlation_matrix).sum(0) + 1) / (iterations + 1)
                        p_rmse = ((null_root_mean_square_error < root_mean_square_error).sum(0) + 1) / (iterations + 1)

                        # Threshold
                        umbral = 0.05/128
                        proba_correlation_per_channel[fold][p_corr < umbral] = p_corr[p_corr < umbral]
                        proba_rmse_per_channel[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]
                        
                        # p-value topographic distribution
                        topo_pvalues_corr[fold] = p_corr
                        topo_pvalues_rmse[fold] = p_rmse

                # Saves model weights and correlations
                if save_results:
                    os.makedirs(path_original, exist_ok=True)
                    Funciones.dump_pickle(path=path_original + f'Pesos_Sesion{sesion}_Sujeto{sujeto}.pkl', 
                                          obj=weights_per_fold.mean(0),
                                          rewrite=True, 
                                          verbose=False)
                    Funciones.dump_pickle(path=path_original + f'Corr_Rmse_Sesion{sesion}_Sujeto{sujeto}.pkl', 
                                          obj=[correlation_per_channel, rmse_per_channel],
                                          rewrite=True, 
                                          verbose=False)
                print(f'\n\t······  Run model for Subject {sujeto}\n')

                # Take average weights, correlation and RMSE between folds of all channels
                average_weights = weights_per_fold.mean(axis=0) # info['nchan'], np.sum(n_feats), len(delays)
                average_correlation = correlation_per_channel.mean(axis=0)
                average_rmse = rmse_per_channel.mean(axis=0)
                
                # Channels that pass the tests
                corr_good_channel_indexes = []
                rmse_good_channel_indexes = []

                if perform_statistical_test:
                    # Correlation and RMSE of channels that pass the test
                    corr_good_channel_indexes, = np.where(np.all((proba_correlation_per_channel < 1), axis=0))
                    rmse_good_channel_indexes, = np.where(np.all((proba_rmse_per_channel < 1), axis=0))

                    # Saves passing channels by subject
                    repeated_good_correlation_channels[corr_good_channel_indexes] += 1
                    repeated_good_rmse_channels[rmse_good_channel_indexes] += 1

                    # Plot shadows
                    Plot.null_correlation_vs_correlation_good_channels(display_interactive_mode=display_interactive_mode, session=sesion, subject=sujeto,
                                              save_path=path_figures, good_channels_indexes=corr_good_channel_indexes, average_correlation=average_correlation, 
                                              save=save_figures, correlation_per_channel=correlation_per_channel, 
                                              null_correlation_per_channel=null_correlation_per_channel)

                # Adapt to yield average p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(axis=0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(axis=0)

                # Plot head topomap across al channel for correlation and rmse
                Plot.topomap(good_channels_indexes=corr_good_channel_indexes, average_coefficient=average_correlation, info=info,
                             coefficient_name='Correlation', save=save_figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, subject=sujeto, session=sesion)
                Plot.topomap(good_channels_indexes=rmse_good_channel_indexes, average_coefficient=average_rmse, info=info,
                             coefficient_name='RMSE', save=save_figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, subject=sujeto, session=sesion)

                # Plot weights
                Plot.channel_weights(info=info, save=save_figures, save_path=path_figures, average_correlation=average_correlation,
                                     average_rmse=average_rmse, best_alpha=alpha, average_weights=average_weights, times=times, 
                                     n_feats=n_feats, stim=stim, session=sesion, subject=sujeto,
                                     display_interactive_mode=display_interactive_mode)

                # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                average_weights_subjects.append(average_weights)
                average_correlation_subjects.append(average_correlation)
                average_rmse_subjects.append(average_rmse)
                pvalues_corr_subjects.append(topo_pval_corr_sujeto)
                pvalues_rmse_subjects.append(topo_pval_rmse_sujeto)
                repeated_good_correlation_channels_subjects.append(repeated_good_correlation_channels)
                repeated_good_rmse_channels_subjects.append(repeated_good_rmse_channels)
            
            # Print the progress of the iteration
            Funciones.iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=sesiones.index(sesion), length_of_iterator=len(sesiones))

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

        # Plot average topomap across each subject
        Plot.average_topomap(average_coefficient_subjects=average_rmse_subjects, info=info, display_interactive_mode=display_interactive_mode,
                             save=save_figures, save_path=path_figures, coefficient_name='RMSE')
        Plot.average_topomap(average_coefficient_subjects=average_correlation_subjects, display_interactive_mode=display_interactive_mode,
                             info=info, save=save_figures, save_path=path_figures, coefficient_name='Correlation', test_result=False) # USING ZERO METHOD PRATT
        
        # Plot topomap with relevant times
        Plot.topo_map_relevant_times(average_weights_subjects=average_weights_subjects, info=info, n_feats=n_feats, band=band, stim=stim, times=times, 
                                sample_rate=sr, save_path=path_figures, save=save_figures, display_interactive_mode=display_interactive_mode)

        # Plot channel-wise correlation topomap 
        Plot.channel_wise_correlation_topomap(average_weights_subjects=average_weights_subjects, info=info, save=save_figures, 
                                              save_path=path_figures, display_interactive_mode=display_interactive_mode)      

        # Plot weights
        condition_of_mesh = np.array([True for st in stim.split('_') if st.startswith('Phoneme') or st.startswith('Spectro')])
        Plot.average_regression_weights(average_weights_subjects=average_weights_subjects, info=info, save=save_figures, save_path=path_figures, 
                                        times=times, n_feats=n_feats, stim=stim, display_interactive_mode=display_interactive_mode, 
                                        colormesh_form=condition_of_mesh.any())
        
        # Plot correlation matrix between subjects # TODO ROTA LA DIAGONAL
        Plot.correlation_matrix_subjects(average_weights_subjects=average_weights_subjects, stim=stim, n_feats=n_feats, save=save_figures, 
                                         save_path=path_figures, display_interactive_mode=display_interactive_mode)

        if perform_statistical_test: 
            # Plot topomap of average p-values across all subject
            Plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_corr_subjects, info=info, display_interactive_mode=display_interactive_mode,
                                   save=save_figures, save_path=path_figures, coefficient_name='correlation')
            Plot.topo_average_pval(pvalues_coefficient_subjects=pvalues_rmse_subjects, info=info, display_interactive_mode=display_interactive_mode,
                                   save=save_figures, save_path=path_figures, coefficient_name='RMSE')
            
            # Plot topomap of sum of repeated channels across all subject
            Plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_correlation_channels_subjects, 
                                        info=info, display_interactive_mode=display_interactive_mode, save=save_figures, 
                                        save_path=path_figures, coefficient_name='correlation')
            Plot.topo_repeated_channels(repeated_good_coefficients_channels_subjects=repeated_good_rmse_channels_subjects, 
                                        info=info, display_interactive_mode=display_interactive_mode, save=save_figures, 
                                        save_path=path_figures, coefficient_name='RMSE')

        # TFCE across subjects
        n_permutations = 256#4096
        t_tfce, clusters, p_tfce, H0, trf_subjects = Statistics.tfce(average_weights_subjects=average_weights_subjects, n_permutations=n_permutations)
                
        # TODO CHECAR ESTO TICK LABELS y
        Plot.plot_t_p_tfce(t=t_tfce,p=p_tfce, trf_subjects_shape=trf_subjects.shape, band=band, stim=stim, pval_tresh=.05,
                           save_path=path_figures, display_interactive_mode=display_interactive_mode, save=save_figures)
        # TODO CHECAR ESTO TICK LABELS y
        Plot.plot_p_tfce(p=p_tfce, times=times, trf_subjects_shape=trf_subjects.shape, band=band, stim=stim, pval_tresh=.05,
                         save_path=path_figures, display_interactive_mode=display_interactive_mode, save=save_figures)

        # TODO CHECAR ESTO TICK LABELS y
        if 'Spectrogram' in stim:
            Plot.plot_trf_tfce(average_weights_subjects=average_weights_subjects, p=p_tfce, times=times,
                               trf_subjects_shape=trf_subjects.shape, save_path=path_figures, band=band,
                               stim=stim, n_permutations=n_permutations, pval_trhesh=.05,
                               display_interactive_mode=display_interactive_mode, save=save_figures)
        
        # Save results
        if save_results:
            os.makedirs(save_path, exist_ok=True)
            Funciones.dump_pickle(path=save_path+f'{stim}_EEG_{band}.pkl', 
                                  obj={'average_correlation_subjects':average_correlation_subjects,
                                       'repeated_good_correlation_channels_subjects':repeated_good_correlation_channels_subjects},
                                  rewrite=True)
            Funciones.dump_pickle(path=path_original+f'total_weights_per_subject_{stim}_{band}.pkl', 
                                  obj={'average_weights_subjects':average_weights_subjects},
                                  rewrite=True)
# Get run time            
run_time = datetime.now() - start_time
print(run_time)
mensaje_tel(api_token=api_token,chat_id=chat_id, mensaje=f'Main.py run in {run_time} hours')