# Standard libraries
import numpy as np, os
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold
from tqdm import tqdm 

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage, Suppress_print
from model_implementations import fold_model
from utils.plot import hyperparameter_selection
from load import load_data
import config

# Notofication bot
from utils.notification_telegram import tel_message, generate_completion_message
from utils.telegram_config import API_TOKEN, CHAT_ID

# ============
# RUN ANALYSIS
# ============
start_time = datetime.now()
stim, band = 'Envelope', "Theta"
subjects = [1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19]

# Relevant paths
preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/DiLib/')
results_path = os.path.normpath(f'output/DiLib/')

path_eeg = os.path.join(preprocessed_data_path, 'EEG', band)
path_stimulus = os.path.join(preprocessed_data_path, 'Envelope', 'envelope.pkl')
stimulus = load_pickle(path=path_stimulus)

path_results = os.path.normpath(os.path.join(results_path, f'correlations/{band}/{stim}'))
path_weights = os.path.normpath(os.path.join(results_path, f'weights/{band}/{stim}')) 
path_figures = os.path.normpath(os.path.join(f'figures/diliberto/', band, stim)) 

path_validation = os.path.join(results_path, 'validation', stim, band)
alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')

# # Reduce data to a third to save time
# cutoff = int(stimulus.shape[0]//3.5)
# stimulus = stimulus[:cutoff, :]

figures_path = os.path.normpath(f'figures/diliberto/validation/{band}/{stim}')
os.makedirs(figures_path, exist_ok=True)

# Try to access alphas
try:
    alphas = load_pickle(path=alphas_path)
except:
    alphas = {s: {} for s in subjects} 

# Iterate over sessions
for subject in subjects:
    print(f'\n\n------->\tStart of subject {subject}\n')

    # Load data by subject, EEG and info
    eeg = np.array(load_pickle(path=os.path.join(path_eeg, f'sub-{str(subject).zfill(3)}.pkl')))
    stimulus = stimulus[:eeg.shape[0], :]  # Ensure stimulus matches EEG length

    n_feats = [stimulus.shape[1]]
    delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

    # Get relevant indexes
    relevant_indexes = np.arange(len(stimulus))

    print(f'\n\n\t······  Running model for Subject {subject}\n')

    # Take some metrics for each alpha
    correlations_per_fold = np.zeros(
                        (config.n_folds, len(config.alphas_swept))
                    )
    rmse_per_fold = np.zeros(
        (config.n_folds, len(config.alphas_swept))
    )
    trfs_per_fold = np.zeros(
        (config.n_folds, len(config.alphas_swept), len(config.times))
    )
    
    # Make sweep 
    correlations_per_fold_train = np.zeros(
        (config.n_folds, len(config.alphas_swept))
    )
    rmse_per_fold_train = np.zeros(
        (config.n_folds, len(config.alphas_swept))
    )

    # Make the Kfold test
    kf_test = KFold(config.n_folds, shuffle=False)

    # Keep relevant indexes for eeg
    relevant_eeg = eeg[relevant_indexes]
    
    # Run folds 
    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
        trfs_per_fold[fold], correlations_per_fold[fold], rmse_per_fold[fold], correlations_per_fold_train[fold], rmse_per_fold_train[fold] = fold_model(
            relevant_indexes=relevant_indexes,
            train_indexes=train_indexes,
            alpha=config.alphas_swept,
            test_indexes=test_indexes,  
            validation=True,
            stims=stimulus,
            fold=fold,
            eeg=eeg
        ) 
    
    # Calculate mean correlation, rmse and std
    correlations = np.nan_to_num(np.nanmean(correlations_per_fold, axis=0))
    correlations_std = np.nan_to_num(np.nanstd(correlations_per_fold, axis=0))
    rmse = np.nan_to_num(np.nanmean(rmse_per_fold, axis=0))
    rmse_std = np.nan_to_num(np.nanstd(rmse_per_fold, axis=0))
    
    # Same for training
    correlations_train = np.nan_to_num(np.nanmean(correlations_per_fold_train, axis=0))
    rmse_train = np.nan_to_num(np.nanmean(rmse_per_fold_train, axis=0))
    
    # Calculate mean TRFs
    trfs = np.nanmean(trfs_per_fold, axis=0)
    
    # Find all indexes where the relative difference between the correlation and its maximum is within corr_limit_percent
    relative_difference = abs((correlations.max() - correlations)/correlations.max())
    good_indexes_range = np.where(relative_difference < config.val_correlation_limit_percentage)[0]

    # Get the very last one, because the greater the alpha, the smoothest the signal gets
    alpha_subject = config.alphas_swept[int(good_indexes_range[-1])]
    
    # Make the alpha selection process plot
    hyperparameter_selection(
        alphas_swept=config.alphas_swept,
        correlations=correlations, 
        correlations_std=correlations_std,
        correlations_train=correlations_train, 
        rmse=rmse,
        rmse_std=rmse_std,
        rmse_train=rmse_train,
        trfs=trfs,
        alpha_subject=alpha_subject,
        correlation_limit_percentage=config.val_correlation_limit_percentage, 
        session=subject, 
        subject=subject, 
        stim=stim, 
        band=band, 
        save_path=figures_path, 
        save=config.save_figures, 
        no_figures=config.no_figures
    )

    # Update dictionary
    alphas[subject] = alpha_subject
    
    # Print the progress of the iteration
    iteration_percentage(txt=f'\n------->\tEnd of session {subject}\n', i=subjects.index(subject), length_of_iterator=len(subjects))

# Save results
os.makedirs(name=path_validation, exist_ok=True)
if config.save_alphas:
    dump_pickle(path=alphas_path, obj=alphas, rewrite=True)
        
# Get run time            
run_time = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
text = rf'PARAMETERS  \nModel: ' + config.model +f'\nBands: {config.bands}'+'\nStimuli: ' + f'{config.stimuli}'+'\nCondition: ' +f'\nTime interval: ({config.tmin},{config.tmax})s'
text += fr'\n\n_dili.py'
text += f'\n\n\t\t RUN TIME \n\n\t\t{run_time} hours'
print(text)

# Dump metadata
metadata_path = f'saves/log/validation_{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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

# # Send text to telegram bot
# with Suppress_print():
#     mensaje_tel(api_token=api_token, chat_id=chat_id, mensaje=text)
