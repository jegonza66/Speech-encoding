# Standard libraries
import numpy as np, copy

# Specific libraries
from multiprocessing import cpu_count, Pool
from itertools import repeat

# Modules
from mtrf_models import Receptive_field_adaptation

def permutations(iteration:int, eeg:np.ndarray, stims:np.ndarray, tmin:float, tmax:float, sr:int, alpha:float, relevant_indexes:list, 
                     train_indexes:np.ndarray, test_indexes:np.ndarray, stims_preprocess:float, eeg_preprocess:float):
        
        # Define null model
        null_model = Receptive_field_adaptation(
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
                        n_jobs=1,
                        shuffle=True)

        # The fit already already consider relevant indexes of train and test data and applies shuffle and standarization|normalization
        null_model.fit(stims, eeg)

        # Predict and save
        predicted, eeg_test = null_model.predict(stims)

        # Calculates and saves correlation of each channel
        correlation_matrix = np.array([np.corrcoef(eeg_test[:, j], predicted[:, j])[0,1] for j in range(eeg_test.shape[1])])

        # Calculates and saves root mean square error of each channel
        root_mean_square_error = np.array(np.sqrt(np.power((predicted - eeg_test), 2).mean(0)))        
        return null_model.coefs, correlation_matrix, root_mean_square_error, iteration

# TODO CHECK DESCRIPTION
def simulation_mtrf(iterations:int,
                    fold:int,
                    stims:np.ndarray, 
                    eeg:np.ndarray,
                    sr:int, 
                    tmin:float, 
                    tmax:float,
                    relevant_indexes:list,
                    alpha:float,
                    train_indexes:np.ndarray,
                    test_indexes:np.ndarray, 
                    stims_preprocess:str,
                    eeg_preprocess:str,
                    null_correlation:np.ndarray, 
                    null_weights:np.ndarray, 
                    null_errors:np.ndarray,
                    n_jobs:int=-1):
    # Define iterations
    iterations=np.arange(iterations)

    if n_jobs!=1:
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(permutations, zip(iterations, repeat(eeg), repeat(stims), repeat(tmin), repeat(tmax), repeat(sr), repeat(alpha),\
                      repeat(relevant_indexes), repeat(train_indexes), repeat(test_indexes), repeat(stims_preprocess), repeat(eeg_preprocess)))
        for i in iterations:
            null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i], itera = results[i]
            print('Todo bien') if (itera==i) else print('Todo mal')
    else:
        for i in iterations:
            null_weights[fold, i], null_correlation[fold, i], null_errors[fold, i], itera = permutations(iteration=i, 
                                                                                                  eeg=eeg, 
                                                                                                  stims=stims, 
                                                                                                  tmin=tmin, 
                                                                                                  tmax=tmax, 
                                                                                                  sr=sr, 
                                                                                                  alpha=alpha, 
                                                                                                  relevant_indexes=relevant_indexes, 
                                                                                                  train_indexes=train_indexes, 
                                                                                                  test_indexes=test_indexes, 
                                                                                                  stims_preprocess=stims_preprocess, 
                                                                                                  eeg_preprocess=eeg_preprocess)
            if i in iterations[::int(len(iterations)/10)]:
                print("\n\t\t\rProgress {}%".format(int((i + 1) * 100 / len(iterations))), end='')
    return null_weights, null_correlation, null_errors


# def simular_iteraciones_Ridge_plot(info, times, situacion, alpha, iteraciones, sesion, sujeto, fold,
#                                    dstims_train_val, eeg_train_val, dstims_test, eeg_test, fmin, fmax, stim, Band,
#                                    save_path, Display=False):
#     print("\nSesion {} - Sujeto {} - Test round {}".format(sesion, sujeto, fold + 1))
#     psds_rand_correlations = []
#     for iteracion in range(iteraciones):
#         # Random permutation of stimuli
#         dstims_train_random = copy.deepcopy(dstims_train_val)
#         np.random.shuffle(dstims_train_random)

#         # Fit Model
#         Fake_Model = mtrf_models.Ridge(alpha)
#         Fake_Model.fit(dstims_train_random, eeg_train_val)

#         # Test
#         predicho_fake = Fake_Model.predict(dstims_test)

#         # Correlacion
#         Rcorr_fake = np.array(
#             [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicho_fake[:, ii]).ravel())[0, 1] for ii in
#              range(eeg_test.shape[1])])

#         # Error
#         Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test), 2).mean(0)))

#         # PSD
#         psds_test, freqs_mean = mne.time_frequency.psd_array_welch(eeg_test.transpose(), info['sfreq'], fmin, fmax)
#         psds_random, freqs_mean = mne.time_frequency.psd_array_welch(predicho_fake.transpose(), info['sfreq'], fmin,
#                                                                      fmax)

#         psds_channel_corr = np.array([np.corrcoef(psds_test[ii].ravel(), np.array(psds_random[ii]).ravel())[0, 1]
#                                       for ii in range(len(psds_test))])
#         psds_rand_correlations.append(np.mean(psds_channel_corr))

#         # PLOTS
#         if Display:
#             plt.ion()
#         else:
#             plt.ioff()

#         # Plot weights
#         if not iteracion % iteraciones:
#             fig, ax = plt.subplots()
#             fig.suptitle('Sesion {} - Sujeto {} - Corr {:.2f}'.format(sesion, sujeto, np.mean(Rcorr_fake)))

#             evoked = mne.EvokedArray(Fake_Model.coefs, info)
#             evoked.times = times
#             evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='ms', show=False,
#                         spatial_colors=True, unit=False, units='w', axes=ax)

#             ax.plot(times * 1000, evoked._data.mean(0), 'k--', label='Mean', zorder=130, linewidth=2)

#             ax.xaxis.label.set_size(13)
#             ax.yaxis.label.set_size(13)
#             ax.legend(fontsize=13)
#             ax.grid()

#             # Plot signal and fake prediction
#             eeg_x = np.linspace(0, len(eeg_test) / 128, len(eeg_test))
#             fig = plt.figure()
#             fig.suptitle('Random prediction')
#             plt.plot(eeg_x, eeg_test[:, 0], label='Signal')
#             plt.plot(eeg_x, predicho_fake[:, 0], label='Prediction')
#             plt.title('Pearson Correlation = {}'.format(Rcorr_fake[0]))
#             plt.xlim([18, 26])
#             plt.ylim([-3, 3])
#             plt.xlabel('Time [ms]')
#             plt.ylabel('Amplitude')
#             plt.grid()
#             plt.legend()

#             if save_path:
#                 os.makedirs(save_path+'Fake/Stim_{}_EEG_Band{}/'.format(stim,Band), exist_ok=True)
#                 plt.savefig(save_path+'Fake/Stim_{}_EEG_Band{}/Sesion{}_Sujeto_{}.png'.format(stim, Band,sesion, sujeto))
#                 plt.savefig(save_path + 'Fake/Stim_{}_EEG_Band{}/Sesion{}_Sujeto_{}.svg'.format(stim, Band, sesion, sujeto))

#             # Plot PSD
#             fig, ax = plt.subplots()
#             fig.suptitle('Sesion {} - Sujeto {} - Situacion {}'.format(sesion, sujeto, situacion))
#             evoked = mne.EvokedArray(psds_random, info)
#             evoked.times = freqs_mean
#             evoked.plot(scalings=dict(eeg=1, grad=1, mag=1), zorder='std', time_unit='s', show=False,
#                         spatial_colors=True, unit=False, units='w', axes=ax)
#             ax.set_xlabel('Frequency [Hz]')
#             ax.grid()
#             if True:
#                 save_path_graficos = 'gráficos/PSD/Zoom/Fake/{}/'.format(Band)
#                 os.makedirs(save_path_graficos, exist_ok=True)
#                 plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.png'.format(sesion, sujeto, Band))
#                 plt.savefig(save_path_graficos + 'Sesion{} - Sujeto{}.svg'.format(sesion, sujeto, Band))

#         print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')
#     return psds_rand_correlations


# def simular_iteraciones_Ridge(Fake_Model, iteraciones, sesion, sujeto, fold, dstims_train_val, eeg_train_val,
#                               dstims_test, eeg_test, Pesos_fake, Correlaciones_fake, Errores_fake):
#     print("\nSesion {} - Sujeto {} - Fold {}".format(sesion, sujeto, fold + 1))
#     for iteracion in np.arange(iteraciones):
#         # Random permutations of stimuli
#         dstims_train_random = copy.deepcopy(dstims_train_val)
#         np.random.shuffle(dstims_train_random)

#         # Fit Model
#         Fake_Model.fit(dstims_train_random, eeg_train_val)  # entreno el modelo
#         Pesos_fake[fold, iteracion] = Fake_Model.coefs

#         # Test
#         predicho_fake = Fake_Model.predict(dstims_test)

#         # Correlacion
#         Rcorr_fake = np.array(
#             [np.corrcoef(eeg_test[:, ii].ravel(), np.array(predicho_fake[:, ii]).ravel())[0, 1] for ii in
#              range(eeg_test.shape[1])])
#         Correlaciones_fake[fold, iteracion] = Rcorr_fake

#         # Error
#         Rmse_fake = np.array(np.sqrt(np.power((predicho_fake - eeg_test), 2).mean(0)))
#         Errores_fake[fold, iteracion] = Rmse_fake

#         print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')
#     return Pesos_fake, Correlaciones_fake, Errores_fake

def simular_iteraciones_decoding(Fake_Model, iteraciones, sesion, sujeto, fold, dstims_train_val, eeg_train_val,
                              dstims_test, eeg_test, Pesos_fake, Patterns_fake, Correlaciones_fake, Errores_fake):
    print("\nSesion {} - Sujeto {} - Fold {}".format(sesion, sujeto, fold + 1))
    for iteracion in np.arange(iteraciones):
        # Random permutations of stimuli
        eeg_train_random = copy.deepcopy(eeg_train_val)
        np.random.shuffle(eeg_train_random)

        # Fit Model
        Fake_Model.fit(eeg_train_random, dstims_train_val)  # entreno el modelo
        Pesos_fake[fold, iteracion] = Fake_Model.coefs
        Patterns_fake[fold, iteracion] = Fake_Model.patterns

        # Test
        predicho_fake = Fake_Model.predict(eeg_test)

        # Correlacion
        Rcorr_fake = np.array(
            [np.corrcoef(dstims_test[:, -1].ravel(), np.array(predicho_fake).ravel())[0, 1]])
        Correlaciones_fake[fold, iteracion] = Rcorr_fake

        # Error
        Rmse_fake = np.sqrt(np.power((predicho_fake.ravel() - dstims_test[:, -1]), 2).mean(0))
        Errores_fake[fold, iteracion] = Rmse_fake

        print("\rProgress: {}%".format(int((iteracion + 1) * 100 / iteraciones)), end='')
    return Pesos_fake, Patterns_fake, Correlaciones_fake, Errores_fake
