# Standard libraries
import os, pickle, numpy as np
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
import Load, Models, Plot, Statistics, Funciones

startTime = datetime.now()

# ==========
# PARAMETERS
# ==========

# Save / Display Figures
display_interactive_mode = False
Display_Total_Figures = False
Save_Ind_Figures = True 
Save_Total_Figures = True
save_results = False

# Random permutations
perform_statistical_test = False

# Standarization
stims_preprocess = 'Normalize'
eeg_preprocess = 'Standarize'

# Stimuli and EEG frecuency band
stims = ['Envelope']
bands = ['Theta']

# Dialogue situation
situation = 'Escucha'

# Model parameters ('Ridge' or 'mtrf') #TODO va a ser mtrf de ahora en más
model = 'mtrf'

# Preset alpha (penalization parameter)
set_alpha = None
default_alpha = 1000
alpha_correlation_limit = 0.01
alphas_fname = f'saves/alphas/alphas_Corr{alpha_correlation_limit}.pkl'
try:
    f = open(alphas_fname, 'rb')
    alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

# Run setup
sesiones = [21, 22, 23, 24, 25, 26, 27, 29, 30]
total_subjects = len(sesiones)*2

# EEG sample rate
sr = 128

# Run times
tmin, tmax = -.2, .6
delays = np.arange(int(np.round(tmin * sr)), int(np.round(tmax * sr) + 1))
times = (delays/sr)

# ============
# RUN ANALYSIS
# ============

for band in bands:
    for stim in stims:
        # Update
        print('\n==========================\n','\tPARAMETERS\n\n','Model: ' + model+'\n','band: ' + str(band)+'\n','Stimulus: ' + stim+'\n','Status: ' + situation+'\n',f'tmin: {tmin} - tmax: {tmax}\n','\n==========================\n')
        
        # Relevant paths
        save_path = f'saves/{model}/{situation}/Final_Correlation/tmin{tmin}_tmax{tmax}/'
        procesed_data_path = f'saves/Preprocesed_Data/tmin{tmin}_tmax{tmax}/'
        path_original = f'saves/{model}/{situation}/Original/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'
        path_null = f'saves/{model}/{situation}/Fake_it/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'
        path_figures = f'figures/{model}/{situation}/stims_{stims_preprocess}_EEG_{eeg_preprocess}/tmin{tmin}_tmax{tmax}/stim_{stim}_EEG_band_{band}/'

        # Iterate over sessions
        sujeto_total = 0
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

            # Load stimuli by subject (i.e: concatenated stimuli features)#TODO PENSAR COMO IMPLEMENTAR EN CASO DE SER MÁS DE UN ESTIMULO. XEJ.: 'Envelope_Phonemes'. En ese caso, la delayed matrix de features que usa el modelo estará compuesta por el delayed de envelope y luego el de phonemes. Como está ahora toma como si todo fuera la misma variable
            stims_sujeto_1 = np.hstack([sujeto_1[stimulus] for stimulus in stim.split('_')]) 
            stims_sujeto_2 = np.hstack([sujeto_2[stimulus] for stimulus in stim.split('_')])

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
                weights_per_fold = np.zeros((n_folds, info['nchan'], sum(delayed_length_per_stimuli)), dtype=np.float16)
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
                    
                    # Flip coefficients to get wright order
                    weights_per_fold[fold] = np.flip(mtrf.coefs, axis=0)

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
                    
                # Take average weights, correlation and RMSE between folds of all channels
                average_weights = np.flip(weights_per_fold.mean(0), axis=1)
                average_correlation = correlation_per_channel.mean(0)
                average_rmse = rmse_per_channel.mean(0)
                
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
                    Plot.null_correlation_vs_correlation_good_channels(display_interactive_mode=display_interactive_mode, sesion=sesion, sujeto=sujeto,
                                              save_path=path_figures, good_channels_indexes=corr_good_channel_indexes, average_correlation=average_correlation, 
                                              save=Save_Ind_Figures, correlation_per_channel=correlation_per_channel, 
                                              null_correlation_per_channel=null_correlation_per_channel)

                # Adapt to yield average p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(0)

                # Plot cabezas y canales 
                Plot.topomap(good_channels_indexes=corr_good_channel_indexes, average_coefficient=average_correlation, info=info,
                             coefficient_name='Correlation', save=Save_Ind_Figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, sujeto=sujeto, sesion=sesion)
                Plot.topomap(good_channels_indexes=rmse_good_channel_indexes, average_coefficient=average_rmse, info=info,
                             coefficient_name='RMSE', save=Save_Ind_Figures, display_interactive_mode=display_interactive_mode, 
                             save_path=path_figures, sujeto=sujeto, sesion=sesion)
                
                # Plot weights 
                Plot.channel_weights(info=info, save=Save_Ind_Figures, save_path=path_figures, average_correlation=average_correlation,
                                     average_rmse=average_rmse, best_alpha=alpha, average_weights=average_weights, times=times, 
                                     delayed_length_per_stimuli=delayed_length_per_stimuli, stim=stim, sesion=sesion, sujeto=sujeto,
                                     display_interactive_mode=display_interactive_mode)
                #TODO HASTAS ACA

                # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                if not sujeto_total:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = average_weights
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = average_correlation
                    Rmse_totales_sujetos = average_rmse
                    # Save p-values for all subjects
                    pvalues_corr_subjects = topo_pval_corr_sujeto
                    pvalues_rmse_subjects = topo_pval_rmse_sujeto
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = repeated_good_correlation_channels
                    Canales_repetidos_rmse_sujetos = repeated_good_rmse_channels
                else:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, average_weights))
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, average_correlation))
                    Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, average_rmse))
                    # Save p-values for all subjects
                    pvalues_corr_subjects = np.vstack((pvalues_corr_subjects, topo_pval_corr_sujeto))
                    pvalues_rmse_subjects = np.vstack((pvalues_rmse_subjects, topo_pval_rmse_sujeto))
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = np.vstack((Canales_repetidos_corr_sujetos, repeated_good_correlation_channels))
                    Canales_repetidos_rmse_sujetos = np.vstack((Canales_repetidos_rmse_sujetos, repeated_good_rmse_channels))
                sujeto_total += 1
            
            # Print the progress of the iteration
            Funciones.iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=sesiones.index(sesion), length_of_iterator=len(sesiones))

            del average_weights, average_rmse, average_correlation, correlation_per_channel, rmse_per_channel, correlation_matrix, root_mean_square_error,\
                eeg_test, eeg, stims, stims_sujeto_1, stims_sujeto_2, sujeto_1, sujeto_2, eeg_sujeto_1, eeg_sujeto_2

        # Armo cabecita con correlaciones promedio entre sujetos
        _, lat_test_results_corr = Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_Total_Figures,
                                                              Save_Total_Figures, path_figures, title='Correlation', lat_max_chs=12)

        _, lat_test_results_rmse = Plot.Cabezas_corr_promedio(Rmse_totales_sujetos, info, Display_Total_Figures,
                                                              Save_Total_Figures, path_figures, title='Rmse')

        # Armo cabecita con canales repetidos
        if perform_statistical_test:
            Plot.topo_pval(pvalues_corr_subjects.mean(0), info, Display_Total_Figures,
                                     Save_Total_Figures, path_figures, title='Correlation')
            Plot.topo_pval(pvalues_rmse_subjects.mean(0), info, Display_Total_Figures,
                           Save_Total_Figures, path_figures, title='Rmse')

            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, path_figures, title='Correlation')
            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, path_figures, title='Rmse')

        # Grafico Pesos
        Pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                                Save_Total_Figures, path_figures, delayed_length_per_stimuli, stim, ERP=True)

        Plot.regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                       Save_Total_Figures, path_figures, delayed_length_per_stimuli, stim, band, ERP=True)
        

        t_int = datetime.now()
        # TFCE across subjects # TODO TARDA UN SIGLO ESTA FUNCION Q ONDA, AUMENTA CON sujeto_total. Para Envelope todas las sesiones tardo 27 min
        t_tfce, clusters, p_tfce, H0, trf_subjects, n_permutations = Statistics.tfce(Pesos_totales_sujetos_todos_canales, times, delayed_length_per_stimuli, n_permutations=4096)#, verbose=None)
        print(datetime.now()-t_int)
        Plot.plot_t_p_tfce(t=t_tfce, p=p_tfce, title='TFCE', mcc=True, shape=trf_subjects.shape,
                           graficos_save_path=path_figures, band=band, stim=stim, pval_trhesh=0.05, Display=Display_Total_Figures)
        Plot.plot_p_tfce(p=p_tfce, times=times, title='', mcc=True, shape=trf_subjects.shape,
                           graficos_save_path=path_figures, band=band, stim=stim, pval_trhesh=0.05, fontsize=17,
                           Display=Display_Total_Figures, Save=Save_Total_Figures)

        if stim == 'Spectrogram':
            Plot.plot_trf_tfce(Pesos_totales_sujetos_todos_canales=Pesos_totales_sujetos_todos_canales, p=p_tfce,
                               times=times, title='', mcc=True, shape=trf_subjects.shape, n_permutations=n_permutations,
                               graficos_save_path=path_figures, band=band, stim=stim,
                               pval_trhesh=0.05, fontsize=17, Display=Display_Total_Figures, Save=Save_Total_Figures)

        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, stim, delayed_length_per_stimuli, info, times, sesiones, Display_Total_Figures, Save_Total_Figures,
                                      path_figures)
        try:
            _ = Plot.Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, band, stim, times, sr, Display_Total_Figures,
                                            Save_Total_Figures, path_figures, delayed_length_per_stimuli)
        except:
            pass
        # Cabezas de correlacion de pesos por canal
        Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_Total_Figures,
                                              Save_Total_Figures, path_figures)

        # SAVE FINAL CORRELATION
        if save_results and sujeto_total == 18:
            os.makedirs(save_path, exist_ok=True)
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, band), 'wb')
            pickle.dump([Correlaciones_totales_sujetos, Canales_repetidos_corr_sujetos], f)
            f.close()

            # Save final weights
            f = open(path_original + 'Pesos_Totales_{}_{}.pkl'.format(stim, band), 'wb')
            pickle.dump(Pesos_totales, f)
            f.close()

        del Pesos_totales

print(datetime.now() - startTime)
