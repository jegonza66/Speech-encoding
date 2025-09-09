# Standard libraries
import os, numpy as np
from datetime import datetime

# Specific libraries
from sklearn.model_selection import KFold

# Modules
from utils.general_functions import load_pickle, dump_pickle, dict_to_csv, iteration_percentage
from model_implementations import fold_model
from load import load_data
import config

# Notification bot
from utils.notification_telegram import tel_message, generate_permutation_completion_message
from utils.telegram_config import API_TOKEN, CHAT_ID

# Logging
from utils.logs import setup_logger

# Initialize logger
logger = setup_logger(
    name='random_permutations',
    log_to_file=config.LOG_TO_FILE,
    log_dir=config.LOG_DIR if config.LOG_TO_FILE else None,
    level=config.LOG_LEVEL
)

# ============
# RUN ANALYSIS
# ============
for situation in config.situations:
    start_time = datetime.now()
    stimulus_runtimes = {}
    total_permutations_run = 0
    
    for band in config.bands:
        for stim in config.stimuli:
            stim_start_time = datetime.now()
            ordered_stims, ordered_band = sorted(stim.split('_')), sorted(band.split('_'))
            stim, band = '_'.join(ordered_stims), '_'.join(ordered_band)

            # Update
            logger.info(
                '\n===========================\n'
                '\tPARAMETERS\n\n'
                f'Model: {config.model}-{config.solver}\n'
                f'Band: {band}\n'
                f'Stimulus: {stim}\n'
                f'Condition: {situation}\n'
                f'Time interval: ({config.tmin},{config.tmax})s\n'
                f'Permutations: {config.random_permutations}\n'
                '\n===========================\n'
            )
            
            # Relevant paths
            preprocessed_data_path = os.path.normpath(f'saves/preprocessed_data/tmin{config.tmin}_tmax{config.tmax}/')
            if config.same_validation_subjects:
                if config.external_validation:
                    path_null = f'output/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                else:
                    path_null = f'output/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/same_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            else:
                if config.external_validation:
                    path_null = f'output/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                else:
                    path_null = f'output/{config.model}-{config.solver}/External-{situation}/null_model/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/distinct_alpha/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
                
            if config.external_validation:
                path_validation = f'output/{config.model}-{config.solver}/External/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            else:
                path_validation = f'output/{config.model}-{config.solver}/{situation}/validation/stims_{config.stims_preprocess}_EEG_{config.eeg_preprocess}/tmin{config.tmin}_tmax{config.tmax}/{band}/{stim}/'
            alphas_path = os.path.join(path_validation, f'corr_limit_{config.val_correlation_limit_percentage}.pkl')
                            
            # Iterate over sessions
            for session in config.sessions:
                print(f'\n------->\tStart of session {session}\n')
                
                # Load data by subject, EEG and info
                subject_1, subject_2, samples_info = load_data(
                    preprocessed_data_path=preprocessed_data_path,
                    situation=situation,
                    session=session,
                    stimuli=stim,
                    band=band
                )
                
                eeg_subject_1, eeg_subject_2 = subject_1['EEG'], subject_2['EEG']

                if config.just_load_data:
                    continue

                # Load stimuli by subject (i.e: concatenated stimuli features)
                stims_subject_1 = np.hstack([subject_1[stimulus] for stimulus in stim.split('_')]) 
                stims_subject_2 = np.hstack([subject_2[stimulus] for stimulus in stim.split('_')])
                n_feats = [subject_1[stimulus].shape[1] for stimulus in stim.split('_')]
                delayed_length_per_stimuli = [n_feat*len(config.delays) for n_feat in n_feats]

                # Get relevant indexes
                relevant_indexes_1 = samples_info['keep_indexes1'].copy()
                relevant_indexes_2 = samples_info['keep_indexes2'].copy()

                # Initialize empty variables to store relevant data of each fold 
                null_weights_per_fold = np.zeros((config.n_folds, config.random_permutations, config.info_mne['nchan'], np.sum(n_feats), len(config.delays)), dtype=np.float32)
                null_correlation_per_channel_per_fold = np.zeros((config.n_folds, config.random_permutations, config.info_mne['nchan']), dtype=np.float32)
                null_errors_per_channel_per_fold = np.zeros((config.n_folds, config.random_permutations, config.info_mne['nchan']), dtype=np.float32)
                
                # Run model for each subject
                for subject, eeg, stims, relevant_indexes in zip((1, 2), (eeg_subject_1, eeg_subject_2), (stims_subject_1, stims_subject_2), (relevant_indexes_1, relevant_indexes_2)):
                    print(f'\n\t······  Running permutations for Subject {subject}\n')
                    
                    # Set alpha for specific subject
                    if config.set_alpha is None:
                        try:
                            alphas = load_pickle(path=alphas_path)
                            alpha = alphas[session][subject]
                        except:
                            alpha = config.default_alpha
                    else:
                        alpha = config.set_alpha
                
                    # Make k-fold test with 5 folds (remain 20% as validation set, then interchange to cross validate)
                    kf_test = KFold(config.n_folds, shuffle=False)
                    
                    # Keep relevant indexes for eeg
                    relevant_eeg = eeg[relevant_indexes]

                    for fold, (train_indexes, test_indexes) in enumerate(kf_test.split(relevant_eeg)):
                        logger.debug(f'\n\t······  [{fold+1}/{config.n_folds}]\t-->\t α:{alpha:.2f}\t🎲 {config.random_permutations} permutations')

                        # Run permutations 
                        null_weights_per_fold[fold], null_correlation_per_channel_per_fold[fold], null_errors_per_channel_per_fold[fold]  = fold_model(
                            fold=fold,
                            alpha=alpha,
                            stims=stims,
                            eeg=eeg,
                            relevant_indexes=relevant_indexes,
                            train_indexes=train_indexes,
                            test_indexes=test_indexes,
                            validation=False,
                            shuffle=True,
                            statistical_test=False,
                            )                        
                    
                    # Save permutations
                    os.makedirs(path_null, exist_ok=True)
                    dump_pickle(
                                path=path_null+ f'null_metrics_ses_{session}_sub_{subject}_{config.random_permutations}.pkl',
                                obj={
                                    'null_correlation_per_channel_per_fold':null_correlation_per_channel_per_fold, 
                                    'null_errors_per_fold':null_errors_per_channel_per_fold
                                    },
                                rewrite=True
                                )
                    dump_pickle(
                                path=path_null+ f'null_weights_ses_{session}_sub_{subject}_{config.random_permutations}.pkl',
                                obj=null_weights_per_fold.mean(axis=0),
                                rewrite=True
                                )
                    
                    # Update permutations counter
                    total_permutations_run += config.random_permutations * config.n_folds
                    
                    print(f'\n\t······ ✓ Completed permutations for Subject {subject}\n')

                # Print the progress of the iteration
                iteration_percentage(
                    txt=f'\n------->\tEnd of session {session}\n', 
                    i=config.sessions.index(session), 
                    length_of_iterator=len(config.sessions)
                )

            # Calculate runtime for this stimulus
            stim_runtime = datetime.now().replace(microsecond=0) - stim_start_time.replace(microsecond=0)
            stimulus_runtimes[f"{band}_{stim}"] = stim_runtime
            
            # Print stimulus completion time
            logger.info(
                f"\n\t{'='*40}\n"
                f"\t✅ STIMULUS COMPLETED: {band}_{stim}\n\n"
                f"\t⏱️  Runtime: {stim_runtime}\n"
                f"\t🎲 Permutations run: {config.random_permutations * config.n_folds * len(config.sessions) * 2}\n"
                f"\t📊 Subjects processed: {len(config.sessions) * 2}\n"
                f"\t{'='*40}\n"
            )

    # Get total run time
    total_runtime = datetime.now().replace(microsecond=0) - start_time.replace(microsecond=0)
    
    # Generate completion message
    text = generate_permutation_completion_message(
        situation=situation,
        total_permutations_run=total_permutations_run,
        stimulus_runtimes=stimulus_runtimes,
        total_runtime=str(total_runtime)
    )
    
    # Send text to telegram bot
    tel_message(
        api_token=API_TOKEN,
        chat_id=CHAT_ID, 
        message=text,
        caption='Random Permutations Analysis Completed'
    )

    # Dump metadata
    metadata_path = f'saves/log/permutations/{datetime.now().strftime("%Y-%m-%d--%H-%M-%S")}/'
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
    
    # Print the completion message
    logger.info(text)

if True and __name__ == "__main__":
    from pathlib import Path
    import subprocess
    import sys
    import os
    
    # Get project root (assumes script is in project directory)
    project_root = Path(__file__).parent.absolute()

    # Find Python executable in current environment
    python_exe = sys.executable
    
    # Cross-platform command execution
    main_script = str(project_root / "main.py")
    
    # Run the main script to get significant channels
    logger.info(
            f"Running main script: {main_script+python_exe}"
        )
    try:
        if os.name == 'nt':  # Windows
            command = [python_exe, main_script]
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                encoding='utf-8', 
                errors='replace'
            )
            # Only show output if there are errors
            if result.returncode != 0:
                logger.error(f"Main script failed with return code: {result.returncode}")
                if result.stderr:
                    print("STDERR:")
                    print(result.stderr)
                if result.stdout:
                    print("STDOUT:")
                print(result.stdout)
            else:
                # Success - no output needed
                logger.info("Main script completed successfully")
        else:  # Unix-like systems
            command = [python_exe, main_script]
            subprocess.run(command)
            
    except Exception as e:
        logger.error(f"Error running main script: {e}")