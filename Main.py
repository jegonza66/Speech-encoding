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
Display_Ind_Figures = False
Display_Total_Figures = False
Save_Ind_Figures = True 
Save_Total_Figures = True
Save_Results = False

# Random permutations
Statistical_test = False

# Standarization
Stims_preprocess = 'Normalize'
EEG_preprocess = 'Standarize'

# Stimuli and EEG frecuency band
Stims = ['Envelope']
Bands = ['Theta']

# Dialogue situation
situacion = 'Escucha'

# Model parameters ('Ridge' or 'mtrf')
model = 'Ridge'

# Preset alpha (penalization parameter)
set_alpha = None
default_alpha = 1000
Alpha_Corr_limit = 0.01
alphas_fname = 'saves/Alphas/Alphas_Corr{}.pkl'.format(Alpha_Corr_limit)
try:
    f = open(alphas_fname, 'rb')
    Alphas = pickle.load(f)
    f.close()
except:
    print('\n\nAlphas file not found.\n\n')

# Run setup
sesiones = [21]#, 22, 23, 24, 25, 26, 27, 29, 30]
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

for Band in Bands:
    for stim in Stims:
        print('\n==========================')
        print('\tPARAMETERS')
        print('\nModel: ' + model)
        print('Band: ' + str(Band))
        print('Stimulus: ' + stim)
        print('Status: ' + situacion)
        print(f'tmin: {tmin} - tmax: {tmax}')
        print('\n==========================')

        # Paths
        save_path = f'saves/{model}/{situacion}/Final_Correlation/tmin{tmin}_tmax{tmax}/'
        procesed_data_path = f'saves/Preprocesed_Data/tmin{tmin}_tmax{tmax}/'
        Run_graficos_path = f'gráficos/{model}/{situacion}/Stims_{Stims_preprocess}_EEG_{EEG_preprocess}/tmin{tmin}_tmax{tmax}/Stim_{stim}_EEG_Band_{Band}/'
        Path_original = f'saves/{model}/{situacion}/Original/Stims_{Stims_preprocess}_EEG_{EEG_preprocess}/tmin{tmin}_tmax{tmax}/Stim_{stim}_EEG_Band_{Band}/'
        Path_it = f'saves/{model}/{situacion}/Fake_it/Stims_{Stims_preprocess}_EEG_{EEG_preprocess}/tmin{tmin}_tmax{tmax}/Stim_{stim}_EEG_Band_{Band}/'

        # Iterate over sessions
        sujeto_total = 0
        for sesion in sesiones:
            print(f'\n------->\tStart of session {sesion}\n')
            
            # Load data by subject, EEG and info
            Sujeto_1, Sujeto_2, samples_info = Load.Load_Data(sesion=sesion, 
                                                stim=stim, 
                                                Band=Band, 
                                                sr=sr,
                                                delays=delays,
                                                procesed_data_path=procesed_data_path, 
                                                situacion=situacion,
                                                SilenceThreshold=0.03)
            eeg_sujeto_1, eeg_sujeto_2, info = Sujeto_1['EEG'], Sujeto_2['EEG'], Sujeto_1['info']

            # Load stimuli by subject (i.e: concatenated stimuli features)#TODO PENSAR COMO IMPLEMENTAR EN CASO DE SER MÁS DE UN ESTIMULO. XEJ.: 'Envelope_Phonemes'. En ese caso, la delayed matrix de features que usa el modelo estará compuesta por el delayed de envelope y luego el de phonemes. Como está ahora toma como si todo fuera la misma variable
            stims_sujeto_1 = np.hstack([Sujeto_1[stimulus] for stimulus in stim.split('_')]) 
            stims_sujeto_2 = np.hstack([Sujeto_2[stimulus] for stimulus in stim.split('_')])

            # This used to be called len_estimulos and was calculated in another way but at the end it was this  
            len_delays = [len(delays) for i in range(stims_sujeto_1.shape[1])]
            
            # Get relevant indexes
            relevant_indexes_1 = samples_info['keep_indexes1'].copy()
            relevant_indexes_2 = samples_info['keep_indexes2'].copy()
            
            # Run model for each subject
            for sujeto, eeg, stims, relevant_indexes in zip((1, 2), (eeg_sujeto_1, eeg_sujeto_2), (stims_sujeto_1, stims_sujeto_2), (relevant_indexes_1, relevant_indexes_2)):
                print(f'\n\t······  Running model for Subject {sujeto}\n')
                
                # Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
                Predicciones = {}
                n_folds = 5
                iteraciones = 3000

                # Initialize empty variables to store relevant data of each fold
                Pesos_ronda_canales = np.zeros((n_folds, info['nchan'], sum(len_delays)), dtype=np.float16)
                Corr_buenas_ronda_canal = np.zeros((n_folds, info['nchan']))
                Rmse_buenos_ronda_canal = np.zeros((n_folds, info['nchan']))

                if Statistical_test:
                    Pesos_fake = np.zeros((n_folds, iteraciones, info['nchan'], sum(len_delays)), dtype=np.float16)
                    Correlaciones_fake = np.zeros((n_folds, iteraciones, info['nchan']))
                    Errores_fake = np.zeros((n_folds, iteraciones, info['nchan']))

                # Variable to store all channel's p-value
                topo_pvalues_corr = np.zeros((n_folds, info['nchan']))
                topo_pvalues_rmse = np.zeros((n_folds, info['nchan']))

                # Variable to store p-value of significant channels
                Prob_Corr_ronda_canales = np.ones((n_folds, info['nchan']))
                Prob_Rmse_ronda_canales = np.ones((n_folds, info['nchan']))

                # Variable to store significant channels
                Canales_repetidos_corr_sujeto = np.zeros(info['nchan'])
                Canales_repetidos_rmse_sujeto = np.zeros(info['nchan'])

                # Set alpha for specific subject
                if set_alpha is None:
                    try:
                        alpha = Alphas[Band][stim][sesion][sujeto]
                    except:
                        alpha = default_alpha
                        # print(f'Alpha missing. Ussing default value: {alpha}')
                else:
                    alpha = set_alpha
                    # print(f'Ussing pre-set alpha value: {alpha}')
               
                # Make the Kfold test
                kf_test = KFold(n_folds, shuffle=False)

                # Keep relevant indexes for eeg
                # relevant_indexes = np.arange(eeg.shape[0])
                relevant_eeg = eeg[relevant_indexes]
                for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                    if model=='mtrf':
                        # Implement mne model
                        mtrf = Models.MNE_MTRF(
                            tmin=tmin, 
                            tmax=tmax, 
                            sample_rate=sr, 
                            alpha=alpha, 
                            relevant_indexes=np.array(relevant_indexes),
                            train_indexes=train_indexes, 
                            test_indexes=test_indexes, 
                            stims_preprocess=Stims_preprocess, 
                            eeg_preprocess=EEG_preprocess,
                            fit_intercept=False)
                        
                        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
                        mtrf.fit(stims, eeg)
                        
                        # Flip coefficients to get wright order #TODO
                        # Pesos_ronda_canales[fold] = np.flip(mtrf.coefs, axis=0)
                        Pesos_ronda_canales[fold] = mtrf.coefs

                        # Predict and save
                        predicted = mtrf.predict(stims)
                        Predicciones[fold] = predicted
                    else:
                        ridge_model = Models.ManualRidge(
                            delays=delays, 
                            relevant_indexes=np.array(relevant_indexes),
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,
                            stims_preprocess=Stims_preprocess, 
                            eeg_preprocess=EEG_preprocess,
                            alpha=alpha)
                        
                        X_train, y_train, X_pred, eeg_test = ridge_model.normalization(stims, eeg)
                        
                        # The fit already already consider relevant indexes of train and test data and applies standarization|normalization
                        ridge_model.fit(X_train, y_train)
                        
                        # Flip coefficients to get wright order #TODO
                        # Pesos_ronda_canales[fold] = np.flip(ridge_model.coefs, axis=0)
                        Pesos_ronda_canales[fold] = ridge_model.coef_

                        # Predict and save
                        predicted = ridge_model.predict(X_pred)
                        Predicciones[fold] = predicted
                        

                    # Calculates and saves correlation of each channel
                    # eeg_test = relevant_eeg[test_indexes]
                    Rcorr = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])
                    Corr_buenas_ronda_canal[fold] = Rcorr

                    # Calculates and saves root mean square error of each channel
                    Rmse = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))
                    Rmse_buenos_ronda_canal[fold] = Rmse
                    
                    # Perform statistical test #TODO porque se llaman fake? Jamas se crean en ninguno archivo estos datos, donde los saco?
                    if Statistical_test:
                        try:
                            Correlaciones_fake, Errores_fake = Funciones.load_pickle(path=Path_it + f'Corr_Rmse_fake_Sesion{sesion}_Sujeto{sujeto}.pkl')
                        except:
                            Statistical_test = False

                        # Correlation and RMSE
                        Rcorr_fake = Correlaciones_fake[fold]
                        Rmse_fake = Errores_fake[fold]

                        # p-values for both tests
                        p_corr = ((Rcorr_fake > Rcorr).sum(0) + 1) / (iteraciones + 1)
                        p_rmse = ((Rmse_fake < Rmse).sum(0) + 1) / (iteraciones + 1)

                        # Threshold
                        umbral = 0.05/128
                        Prob_Corr_ronda_canales[fold][p_corr < umbral] = p_corr[p_corr < umbral]
                        Prob_Rmse_ronda_canales[fold][p_rmse < umbral] = p_rmse[p_rmse < umbral]
                        
                        # p-value topographic distribution
                        topo_pvalues_corr[fold] = p_corr
                        topo_pvalues_rmse[fold] = p_rmse

                # Saves model weights and correlations
                if Save_Results:
                    os.makedirs(Path_original, exist_ok=True)
                    Funciones.dump_pickle(path=Path_original + f'Pesos_Sesion{sesion}_Sujeto{sujeto}.pkl', 
                                          obj=Pesos_ronda_canales.mean(0),
                                          rewrite=True, 
                                          verbose=False)
                    Funciones.dump_pickle(path=Path_original + f'Corr_Rmse_Sesion{sesion}_Sujeto{sujeto}.pkl', 
                                          obj=[Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal],
                                          rewrite=True, 
                                          verbose=False)
                    
                # Take average weights, correlation and RMSE between folds of all channels
                Pesos_promedio = np.flip(Pesos_ronda_canales.mean(0), axis=1)
                Corr_promedio = Corr_buenas_ronda_canal.mean(0)
                Rmse_promedio = Rmse_buenos_ronda_canal.mean(0)
                
                # Channels that pass the tests
                Canales_sobrevivientes_corr = []
                Canales_sobrevivientes_rmse = []
                if Statistical_test:
                    # Correlation and RMSE of channels that pass the test
                    Canales_sobrevivientes_corr, = np.where(np.all((Prob_Corr_ronda_canales < 1), axis=0))
                    Canales_sobrevivientes_rmse, = np.where(np.all((Prob_Rmse_ronda_canales < 1), axis=0))

                    # Saves passing channels by subject
                    Canales_repetidos_corr_sujeto[Canales_sobrevivientes_corr] += 1
                    Canales_repetidos_rmse_sujeto[Canales_sobrevivientes_rmse] += 1

                    # Plot shadows
                    Plot.plot_grafico_shadows(Display_Ind_Figures, sesion, sujeto, alpha,
                                              Canales_sobrevivientes_corr, info, sr,
                                              Corr_promedio, Save_Ind_Figures, Run_graficos_path,
                                              Corr_buenas_ronda_canal, Correlaciones_fake)

                # Adapt to yield average p-values
                topo_pval_corr_sujeto = topo_pvalues_corr.mean(0)
                topo_pval_rmse_sujeto = topo_pvalues_rmse.mean(0)

                # Plot cabezas y canales #TODO TICK
                Plot.plot_cabezas_canales(info.ch_names, info, sesion, sujeto, Corr_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Correlación', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_corr)
                Plot.plot_cabezas_canales(info.ch_names, info, sesion, sujeto, Rmse_promedio, Display_Ind_Figures,
                                          info['nchan'], 'Rmse', Save_Ind_Figures, Run_graficos_path,
                                          Canales_sobrevivientes_rmse)

                # Plot weights #TODO TICK
                Plot.plot_grafico_pesos(Display_Ind_Figures, sesion, sujeto, alpha, Pesos_promedio,
                                        info, times, Corr_promedio, Rmse_promedio, Save_Ind_Figures,
                                        Run_graficos_path, len_delays, stim)

                # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
                if not sujeto_total:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = Pesos_promedio
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = Corr_promedio
                    Rmse_totales_sujetos = Rmse_promedio
                    # Save p-values for all subjects
                    pvalues_corr_subjects = topo_pval_corr_sujeto
                    pvalues_rmse_subjects = topo_pval_rmse_sujeto
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = Canales_repetidos_corr_sujeto
                    Canales_repetidos_rmse_sujetos = Canales_repetidos_rmse_sujeto
                else:
                    # Save TRFs for all subjects
                    Pesos_totales_sujetos_todos_canales = np.dstack((Pesos_totales_sujetos_todos_canales, Pesos_promedio))
                    # Save topographic distribution of correlation and rmse for all subjects
                    Correlaciones_totales_sujetos = np.vstack((Correlaciones_totales_sujetos, Corr_promedio))
                    Rmse_totales_sujetos = np.vstack((Rmse_totales_sujetos, Rmse_promedio))
                    # Save p-values for all subjects
                    pvalues_corr_subjects = np.vstack((pvalues_corr_subjects, topo_pval_corr_sujeto))
                    pvalues_rmse_subjects = np.vstack((pvalues_rmse_subjects, topo_pval_rmse_sujeto))
                    # Save significant channels for all subjects
                    Canales_repetidos_corr_sujetos = np.vstack((Canales_repetidos_corr_sujetos, Canales_repetidos_corr_sujeto))
                    Canales_repetidos_rmse_sujetos = np.vstack((Canales_repetidos_rmse_sujetos, Canales_repetidos_rmse_sujeto))
                sujeto_total += 1
            
            # Print the progress of the iteration
            Funciones.iteration_percentage(txt=f'\n------->\tEnd of session {sesion}\n', i=sesiones.index(sesion), length_of_iterator=len(sesiones))

            del Pesos_promedio, Rmse_promedio, Corr_promedio, Corr_buenas_ronda_canal, Rmse_buenos_ronda_canal, Rcorr, Rmse,\
                eeg_test, eeg, stims, stims_sujeto_1, stims_sujeto_2, Sujeto_1, Sujeto_2, eeg_sujeto_1, eeg_sujeto_2

        # Armo cabecita con correlaciones promedio entre sujetos
        _, lat_test_results_corr = Plot.Cabezas_corr_promedio(Correlaciones_totales_sujetos, info, Display_Total_Figures,
                                                              Save_Total_Figures, Run_graficos_path, title='Correlation', lat_max_chs=12)

        _, lat_test_results_rmse = Plot.Cabezas_corr_promedio(Rmse_totales_sujetos, info, Display_Total_Figures,
                                                              Save_Total_Figures, Run_graficos_path, title='Rmse')

        # Armo cabecita con canales repetidos
        if Statistical_test:
            Plot.topo_pval(pvalues_corr_subjects.mean(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Correlation')
            Plot.topo_pval(pvalues_rmse_subjects.mean(0), info, Display_Total_Figures,
                           Save_Total_Figures, Run_graficos_path, title='Rmse')

            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Correlation')
            Plot.Cabezas_canales_rep(Canales_repetidos_corr_sujetos.sum(0), info, Display_Total_Figures,
                                     Save_Total_Figures, Run_graficos_path, title='Rmse')

        # Grafico Pesos
        Pesos_totales = Plot.regression_weights(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                                Save_Total_Figures, Run_graficos_path, len_delays, stim, ERP=True)

        Plot.regression_weights_matrix(Pesos_totales_sujetos_todos_canales, info, times, Display_Total_Figures,
                                       Save_Total_Figures, Run_graficos_path, len_delays, stim, Band, ERP=True)
        

        t_int = datetime.now()
        # TFCE across subjects # TODO TARDA UN SIGLO ESTA FUNCION Q ONDA, AUMENTA CON sujeto_total
        t_tfce, clusters, p_tfce, H0, trf_subjects, n_permutations = Statistics.tfce(Pesos_totales_sujetos_todos_canales, times, len_delays, n_permutations=4096)#, verbose=None)
        print(datetime.now()-t_int)
        Plot.plot_t_p_tfce(t=t_tfce, p=p_tfce, title='TFCE', mcc=True, shape=trf_subjects.shape,
                           graficos_save_path=Run_graficos_path, Band=Band, stim=stim, pval_trhesh=0.05, Display=Display_Total_Figures)
        Plot.plot_p_tfce(p=p_tfce, times=times, title='', mcc=True, shape=trf_subjects.shape,
                           graficos_save_path=Run_graficos_path, Band=Band, stim=stim, pval_trhesh=0.05, fontsize=17,
                           Display=Display_Total_Figures, Save=Save_Total_Figures)

        if stim == 'Spectrogram':
            Plot.plot_trf_tfce(Pesos_totales_sujetos_todos_canales=Pesos_totales_sujetos_todos_canales, p=p_tfce,
                               times=times, title='', mcc=True, shape=trf_subjects.shape, n_permutations=n_permutations,
                               graficos_save_path=Run_graficos_path, Band=Band, stim=stim,
                               pval_trhesh=0.05, fontsize=17, Display=Display_Total_Figures, Save=Save_Total_Figures)

        # Matriz de Correlacion
        Plot.Matriz_corr_channel_wise(Pesos_totales_sujetos_todos_canales, stim, len_delays, info, times, sesiones, Display_Total_Figures, Save_Total_Figures,
                                      Run_graficos_path)
        try:
            _ = Plot.Plot_cabezas_instantes(Pesos_totales_sujetos_todos_canales, info, Band, stim, times, sr, Display_Total_Figures,
                                            Save_Total_Figures, Run_graficos_path, len_delays)
        except:
            pass
        # Cabezas de correlacion de pesos por canal
        Plot.Channel_wise_correlation_topomap(Pesos_totales_sujetos_todos_canales, info, Display_Total_Figures,
                                              Save_Total_Figures, Run_graficos_path)

        # SAVE FINAL CORRELATION
        if Save_Results and sujeto_total == 18:
            os.makedirs(save_path, exist_ok=True)
            f = open(save_path + '{}_EEG_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump([Correlaciones_totales_sujetos, Canales_repetidos_corr_sujetos], f)
            f.close()

            # Save final weights
            f = open(Path_original + 'Pesos_Totales_{}_{}.pkl'.format(stim, Band), 'wb')
            pickle.dump(Pesos_totales, f)
            f.close()

        del Pesos_totales

print(datetime.now() - startTime)