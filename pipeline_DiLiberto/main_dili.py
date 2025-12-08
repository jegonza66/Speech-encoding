# Standard libraries
from datetime import datetime
import os, numpy as np

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from utils.general_functions import iteration_percentage
from model_implementations import fold_model
import config, utils.plot as plot

# Notification bot
from utils.notification_telegram import tel_message, generate_completion_message
from utils.telegram_config import API_TOKEN, CHAT_ID


# ============
# RUN ANALYSIS
# ============
start_time = datetime.now()
band, stim = 'unfilt', 'Phonemes'
subjects = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19]

# Relevant paths
preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/DiLib/')
results_path = os.path.normpath(f'output/DiLib/')

path_eeg = os.path.join(preprocessed_data_path, 'EEG', band)
path_stimulus = os.path.join(preprocessed_data_path, 'Phonemes', 'phonemes.pkl')
stimulus = load_pickle(path=path_stimulus)

path_results = os.path.normpath(os.path.join(results_path, f'correlations/{band}/{stim}'))
path_weights = os.path.normpath(os.path.join(results_path, f'weights/{band}/{stim}')) 
path_figures = os.path.normpath(os.path.join(f'figures/diliberto/', band, stim)) 

path_validation = os.path.join(results_path, 'validation', stim, band)
alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')

# Make lists to store relevant data across sobjects
average_weights_subjects = []
average_correlation_subjects = []
average_rmse_subjects = []

# Store total number of subjects (18) to save figures and results just in this case
total_number_of_subjects = 0

# Iterate over sessions
# for subject in subjects:
for subject in [10]:

    print(f'\n------->\tStart of session {subject}\n')

    # Load data by subject, EEG and info
    eeg = np.array(load_pickle(path=os.path.join(path_eeg, f'sub-{str(subject).zfill(3)}.pkl')))
    
    stimulus = stimulus[:eeg.shape[0]]
    
    n_feats = [stimulus.shape[1]]
    delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

    # Get relevant indexes
    relevant_indexes = np.arange(len(stimulus))

    # Initialize empty variables to store relevant data of each fold
    weights_per_fold = np.zeros((config.n_folds, 128, np.sum(n_feats), len(config.delays)), dtype=np.float16)
    correlation_per_channel = np.zeros((config.n_folds, 128))
    rmse_per_channel = np.zeros((config.n_folds, 128))

    # Set alpha for specific subject
    if config.set_alpha is None:
        try:
            alphas = load_pickle(path=alphas_path)
            alpha = alphas[subject][subject]
        except:
            alpha = config.default_alpha
    else:
        alpha = config.set_alpha

    # Make the Kfold test
    kf_test = KFold(config.n_folds, shuffle=False)

    # Keep relevant indexes for eeg
    relevant_eeg = eeg[relevant_indexes]

    # Run folds
    k_models_output = []
    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
        print(f'\n\t······  [{fold+1}/{config.n_folds}]')
        k_models_output.append(
                        fold_model(
                            fold=fold,
                            alpha=np.float32(alpha),#TODO adapt inside
                            stims=stimulus,
                            eeg=eeg,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,
                            validation=False,
                            statistical_test=False,
                            session=subject,
                            subject=subject, 
                            )
                        )
        
    # Store model output
    for output_k in k_models_output:
        fold, weights, correlation_matrix, root_mean_square_error = output_k[:4]
        
        # Update weights and metrics per fold
        weights_per_fold[fold] = weights
        correlation_per_channel[fold] = correlation_matrix
        rmse_per_channel[fold] = root_mean_square_error 
    print(f'\n\t······  Run model\n')

    # Take average weights, avoiding folds entirely filled with zeros
    for k, weight in enumerate(weights_per_fold):
        if (weight==0).all():
            weights_per_fold[k] = np.full(shape=weight.shape, fill_value=np.nan)
            print(
                f'\n\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>\n'
                f'\t\tFold {k+1}/{config.n_folds} weights are empty\n'
                f'\t\t>>>>>>>>>>>>>>>>>>>>>>>>>>'
            )
            
    average_weights = np.nanmean(weights_per_fold, axis=0) # 128, np.sum(n_feats), len(delays)
    average_weights = np.nan_to_num(average_weights)
                    
    # Take average correlation and RMSE between folds of all channels
    average_correlation = np.nanmean(correlation_per_channel, axis=0)
    average_correlation = np.nan_to_num(average_correlation)
    average_rmse = rmse_per_channel.mean(axis=0)

    # Channels that passed the tests
    corr_good_channel_indexes = []
    rmse_good_channel_indexes = []

    # Variable to store significant channels
    repeated_good_correlation_channels = np.zeros(128)
    repeated_good_rmse_channels = np.zeros(128)
    
    # Plot head topomap across al channel for correlation and rmse
    if stim=='Phonemes':
        stimi="phonemes-dili"
    else:
        stimi="Envelope"
    plot.topomap(
        good_channels_indexes=corr_good_channel_indexes, 
        average_coefficient=average_correlation, 
        info=config.info_mne,
        coefficient_name='Correlation', 
        save=config.save_figures, 
        save_path=path_figures, 
        subject=subject, 
        session=subject, 
        no_figures=config.no_figures
        )
    plot.topomap(
        good_channels_indexes=rmse_good_channel_indexes, 
        average_coefficient=average_rmse, 
        info=config.info_mne,
        coefficient_name='RMSE', 
        save=config.save_figures, 
        save_path=path_figures, 
        subject=subject, 
        session=subject, 
        no_figures=config.no_figures #TODO: remove all config. parameters and put them in plot module
        )

    # Plot weights
    plot.channel_weights(
        info=config.info_mne, 
        save=config.save_figures, 
        save_path=path_figures, 
        average_correlation=average_correlation,
        average_rmse=average_rmse, 
        best_alpha=alpha, 
        average_weights=average_weights, 
        times=config.times,
        n_feats=n_feats, 
        stim=stimi, 
        session=subject, 
        subject=subject, 
        hierarchical_clustering=config.hierarchical_clustering,
        no_figures=config.no_figures
        )

    # Saves average correlation, RMSE and weights between folds of each channel of each subject to take average above subjects channels
    average_weights_subjects.append(average_weights)
    average_correlation_subjects.append(average_correlation)
    average_rmse_subjects.append(average_rmse)

    # Update the number of subjects
    total_number_of_subjects+=1

    # Print the progress of the iteration
    iteration_percentage(txt=f'\n------->\tEnd of session {subject}\n', i=subjects.index(subject), length_of_iterator=len(subjects))

# Get desire shape n_subject, shape of array. For ex.: shape(average_weights_subjects) = n_subj, n_chans, n_feats, n_delays
average_weights_subjects = np.stack(average_weights_subjects, axis=0) # n_subj, n_chans, n_feats, n_delays
average_correlation_subjects = np.stack(average_correlation_subjects , axis=0) # n_subj, n_chans
average_rmse_subjects = np.stack(average_rmse_subjects , axis=0) # n_subj, n_chans

# Save results
if config.save_results and total_number_of_subjects==16:
    os.makedirs(path_results, exist_ok=True)
    os.makedirs(path_weights, exist_ok=True)
    dump_pickle(
            path=os.path.join(path_results,f'{stim}.pkl'),
            obj={'average_correlation_subjects':average_correlation_subjects},
            rewrite=True,
            verbose=True
            )
    dump_pickle(
            path=os.path.join(path_weights, 'total_weights_per_subject.pkl'),
            obj={'average_weights_subjects':average_weights_subjects},
            rewrite=True
            )

# Plot phoneme ocurrences
# if np.array([bool(d) for d in phonemes_occurrences.values()]).any():
#     plot.phonemes_occurrences(occurrences=phonemes_occurrences, save_path=path_figures, save=save_figures, no_figures=config.no_figures)

# Plot average results only if all subjects are analyzed
# config.no_figures=True if (total_number_of_subjects!=16) else config.no_figures

# Plot average topomap metrics across each subject
plot.average_topomap(
    average_coefficient_subjects=average_rmse_subjects, 
    stim=stimi, 
    info=config.info_mne, 
    save=config.save_figures, 
    save_path=path_figures, 
    coefficient_name='RMSE', 
    no_figures=config.no_figures
    )
plot.average_topomap(
    average_coefficient_subjects=average_correlation_subjects, 
    stim=stimi, 
    info=config.info_mne, 
    save=config.save_figures, 
    save_path=path_figures,
    coefficient_name='Correlation', 
    test_result=False, 
    no_figures=config.no_figures
    ) 

# Plot topomap with relevant times
plot.topo_map_relevant_times(
    average_weights_subjects=average_weights_subjects, 
    info=config.info_mne, 
    n_feats=n_feats,
    band=band,
    stim=stimi, 
    times=config.times,
    sample_rate=config.sr, 
    save_path=path_figures, 
    save=config.save_figures, 
    no_figures=config.no_figures
    )

# Plot channel-wise correlation topomap
plot.channel_wise_correlation_topomap(
    average_weights_subjects=average_weights_subjects,
    info=config.info_mne,
    stim=stimi, 
    save=config.save_figures,
    save_path=path_figures, 
    no_figures=config.no_figures
    )

# Plot weights
plot.average_regression_weights(
    average_weights_subjects=average_weights_subjects, 
    info=config.info_mne, 
    save=config.save_figures, 
    save_path=path_figures, 
    hierarchical_clustering=config.hierarchical_clustering,
    times=config.times, 
    n_feats=n_feats, 
    stim=stimi, 
    no_figures=config.no_figures
    )

# Plot correlation matrix between subjects
plot.correlation_matrix_subjects(
    average_weights_subjects=average_weights_subjects,
    stim=stimi, 
    n_feats=n_feats, 
    save=config.save_figures,
    save_path=path_figures, 
    no_figures=config.no_figures
    )

# Get run time
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = f'\n\n\t\t\tPARAMETERS  \n\n\tModel: ' + config.model +f'\n\tBands: {config.bands}'+'\n\tStimuli: ' + f'{config.stimuli}'+'\n\tCondition: ' f'\n\tTime interval: ({config.tmin},{config.tmax})s'+f'\n\tNumber of subjects analyzed: {total_number_of_subjects}. \n\tSessions: {subjects}'
text += '\n\n\t\t\tmain_dili.py'
text += f'\n\n\t\t\tRUN TIME:{run_time}'

# Dump metadata
metadata_path = f'saves/log/main_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
os.makedirs(metadata_path, exist_ok=True)
metadata = {
        name: getattr(config, name) for name in dir(config) 
        if (not name.startswith("__")) and (not callable(getattr(config, name)) and (name not in ['phonemes_to_ipa','ordered_phonemes']))
        }
dict_to_csv(
            path=metadata_path+'metadata.csv',
            obj=metadata,
            rewrite=True
            )

print(text)