"""
The objective of this script is to find source location of different ROIs (Regions of Interest):
- Primary Auditory Cortex (A1)
- Superior Temporal Gyrus (STG)
- Inferior Frontal Gyrus (IFG)
- Inferior Parietal Lobule (IPL)
These ROIs will be used to extract time series data from EEG recordings made with a Biosemi 128-channel system.
"""
from mne.datasets import fetch_fsaverage
from pathlib import Path
from tqdm import tqdm
import numpy as np
import itertools
import json
import mne

mne.set_log_level("CRITICAL")  # Reduce verbosity for clarity

from utils.general_functions import (
    dump_pickle, load_pickle, convert_numpy_keys
)
from utils.load_utils import (
    get_trials, labeling
)
from validation import main as main_validation
from main import main as main_main

from utils.logs import setup_logger
import config

#TODO hablar con Joaco que estuvo corriendo estas cosas
#TODO probar con todo el raw o secciones sin cortes; TAMBIEN POR TRIAL COMPLETO QUE Sï va a entrar. PROBAR PCA y SILENCIO POR TRIAL TMB
#TODO SNR isn't this pre-computable?
#TODO agregar auditivas tempranas como 
#TODO probar pca_flip

SAVE_CHECKPOINT_PATH = Path(r"output\mtrf-ridge\analysis\source_location\correlations.json")

# Path to average head already downloaded
SUBJECT = 'fsaverage'
SUBJECTS_DIR = fetch_fsaverage(
    subjects_dir=Path('data'),
    verbose=False
)

CONFIGURATIONS = {
    # 'length':['trial', 'session'],
    'length':['session'],
    'situation':['External', 'All'],
    'time_course_mode':['mean_flip', 'pca_flip'],
    'tmax_noise_cov':[None, 5, 'Silences'],
}
configurations = [
    combination for combination in itertools.product(*CONFIGURATIONS.values())
    if not (combination[0]=='session' and combination[1]=='All')
]

# These are specific to the atlas
ROI_MAP = {
    'A1': 'transversetemporal', # Giro de Heschl / Córtex Auditivo Primario
    'STG': 'superiortemporal', # Giro Temporal Superior
    'IFG': 'parsopercularis',  # Parte de Broca / Giro Frontal Inferior
    'IPL': 'inferiorparietal'  # Lóbulo Parietal Inferior
}
INFO_MNE_ROI = mne.create_info(
    ch_names=[
        'transversetemporal-lh', 'transversetemporal-rh',
        'superiortemporal-lh', 'superiortemporal-rh',
        'parsopercularis-lh', 'parsopercularis-rh',
        'inferiorparietal-lh', 'inferiorparietal-rh'
    ], 
    sfreq=config.sr, 
    ch_types='eeg'
)
# Configure inverse method parameters
METHOD = "dSPM" # "sLORETA" and "eLORETA" are alternatives
SNR = 3.0  # Standard assumption of Signal-to-Noise Ratio 
LAMBDA2 = 1.0 / SNR**2

# Define source space geometry (cortical surface), BEM, and compute forward solution
src = mne.setup_source_space(
    SUBJECT, 
    spacing='oct6', # standard resolution
    subjects_dir=SUBJECTS_DIR.parent, 
    add_dist=False
)

# Conductivity model (Boundary Element Model - BEM)
model = mne.make_bem_model(
    subject=SUBJECT, 
    conductivity=(0.3, 0.006, 0.3), # brain, skull, scalp
    subjects_dir=SUBJECTS_DIR.parent
)
bem = mne.make_bem_solution(model)

# Alignment between sensors and brain: use 'fsaverage' alignment since we use a template brain
trans = SUBJECT

# Finally, compute the forward solution: for this sensor, where in the brain does the signal come from?
fwd = mne.make_forward_solution(
    config.info_mne, # TODO check if this is correct
    trans=trans, 
    src=src, 
    bem=bem,
    meg=False, 
    eeg=True, 
)
labels = mne.read_labels_from_annot(
    subject=SUBJECT,
    parc='aparc',
    subjects_dir=SUBJECTS_DIR.parent
)
labels_by_name = {label.name: label for label in labels}

# Reorder the ROI time courses to match the desired order and hemispheres
ordered_labels = []
ordered_roi_names = [] # Esta será tu lista de "canales"
for key, roi_name in ROI_MAP.items():
    for hemi in ('lh', 'rh'):
        l_name = f'{roi_name}-{hemi}'
        label = labels_by_name.get(l_name)
        if label is None:
            continue
        ordered_labels.append(label)
        ordered_roi_names.append(f'{key}_{hemi}')
# Load checkpoint data and process
try:
    with open(SAVE_CHECKPOINT_PATH, 'r') as f:
        correlations = json.load(f)
        start_index = len(correlations)
except FileNotFoundError:
    SAVE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    correlations = []
    start_index = 0

# # Check if there are any repeated configurations in the checkpoint
# seen_configurations = set()
# for corr in correlations:
#     config_tuple = tuple(sorted((k, v) for k, v in corr['configuration'].items() if not isinstance(v, dict))) 
#     # if config_tuple in seen_configurations:
#     #     raise ValueError(f"Repeated configuration found in checkpoint: {corr['configuration']}")
#     seen_configurations.add(config_tuple)
#     # >>> seen_configurations.add(config_tuple)
#     # Traceback (most recent call last):
#     # File "<stdin>", line 1, in <module>
#     #     import platform
#     # TypeError: unhashable type: 'dict'
#     # >>> type(config_tuple)
#     # <class 'tuple'>

# Iterate over all configurations
for j, (length, situation, time_course_mode, tmax_noise_cov) in tqdm(enumerate(configurations), total=len(configurations)-start_index, desc="Configurations"):
    if j < start_index:
        continue
    for session in config.sessions:
        trials = get_trials(
            session=session
        )
        correlations_channels = []
        configurations_channels = []
        for channel in [1, 2]:
            eeg = load_pickle(
                rf"saves\preprocessed_data\tmin-0.2_tmax0.6\EEG\Broad\Sesion{session}.pkl"
            )[channel-1]
            
            samples_info = load_pickle(
                path=rf"saves\preprocessed_data\tmin-0.2_tmax0.6\samples_info\External\samples_info_{session}.pkl"
            )
            if length == 'session':
                silences = []
                for i, trial in enumerate(trials):
                    speaker = labeling( # 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)
                        session=session,
                        trial=trial,
                        channel=(channel - 3) * -1,
                        sr=config.sr,
                    )
                    minimum = samples_info[f'trial_lengths{channel}'][trial]
                    silences.extend(speaker[:minimum]==0)

                silences = np.array(silences)
                
                # Source location works better with reference to average
                raw = mne.io.RawArray(
                    data=eeg.T, 
                    info=config.info_mne.copy(), 
                )
                _ = raw.set_eeg_reference(
                    'average', 
                    projection=True
                )
                _ = raw.apply_proj()
                
                # Create a copy of the raw data containing only silence periods for noise covariance estimation
                raw_silences = raw.get_data().copy()[:, silences]
                raw_silences = mne.io.RawArray(raw_silences, raw.info.copy())
                if situation == 'External':
                    raw_data = raw.get_data().copy()[:, samples_info[f'keep_indexes{channel}']]
                else:  
                    raw_data = raw.get_data().copy()
                raw = mne.io.RawArray(raw_data, raw.info.copy(), )
               
                # Noise covariance matrix estimation from raw data (use all data for better estimate) 
                # -> Note that this is not ideal, because of data leaking
                tmax_noise_cov_touse = tmax_noise_cov
                if tmax_noise_cov=='Silences':
                    noise_cov = mne.compute_raw_covariance(
                        raw_silences, 
                        tmax=None, 
                        method=['shrunk', 'empirical'],
                    )
                elif isinstance(tmax_noise_cov, (int, float)):
                    noise_cov = mne.compute_raw_covariance(
                        raw, 
                        tmax=tmax_noise_cov, 
                        method=['shrunk', 'empirical'],
                    )
                else: # None # TODO REVISAR 
                    noise_cov = mne.compute_raw_covariance(
                        raw, 
                        tmax=None, 
                        method=['shrunk', 'empirical'],
                    )
                
                # Map to source space using Minimum Norm Estimation (MNE)
                inverse_operator = mne.minimum_norm.make_inverse_operator(
                    raw.info, 
                    fwd, 
                    noise_cov, 
                    loose=0.2,
                    depth=0.8
                )

                # Transform 'raw' from 128 channels to ~10,000 cortical vertices. (stc) - 'SourceEstimate'
                stc = mne.minimum_norm.apply_inverse_raw(
                    raw, 
                    inverse_operator, 
                    LAMBDA2, 
                    method=METHOD, 
                    pick_ori=None, 
                    verbose=False
                )
                # Time series for each ROI
                roi_time_courses = mne.extract_label_time_course( 
                    stc, 
                    ordered_labels, 
                    src, 
                    mode=time_course_mode, 
                ).T # --> (n_times, n_rois)             
                if situation != 'External':
                    roi_time_courses = roi_time_courses[
                        samples_info[f'keep_indexes{channel}'], :
                    ]
                else:
                    roi_time_courses = roi_time_courses   
            else:
                keep_indexes = np.zeros(sum(samples_info[f'trial_lengths{channel}']))
                keep_indexes[samples_info[f'keep_indexes{channel}']] = 1
                roi_time_courses = []
                for i, trial in enumerate(trials):
                    speaker = labeling( # 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)
                        session=session,
                        trial=trial,
                        channel=(channel - 3) * -1,
                        sr=config.sr,
                    )
                    minimum = samples_info[f'trial_lengths{channel}'][trial]
                    
                    previous_samples = sum(samples_info[f'trial_lengths{channel}'][:trial])
                    relative_minimum = previous_samples + samples_info[f'trial_lengths{channel}'][trial]
                    valid_idx = keep_indexes[previous_samples:relative_minimum].astype(bool)
                    if valid_idx.sum() == 0:
                        tmax_noise_cov_touse=5
                    else:
                        tmax_noise_cov_touse=tmax_noise_cov
                    raw_trial = mne.io.RawArray(
                        data=eeg[previous_samples:relative_minimum, :].T, 
                        info=config.info_mne.copy(), 
                    )

                    # Source location works better with reference to average
                    _ = raw_trial.set_eeg_reference(
                        'average', 
                        projection=True
                    )
                    _ = raw_trial.apply_proj()
                    
                    # Create a copy of the raw data containing only silence periods for noise covariance estimation
                    raw_silences = raw_trial.get_data().copy()[:, speaker[:minimum]==0]
                    raw_silences = mne.io.RawArray(raw_silences, raw_trial.info.copy())
                    if situation == 'External':
                        raw_data = raw_trial.get_data().copy()[:, valid_idx]
                        if raw_data.shape[1]==0:
                            continue
                    else:  
                        raw_data = raw_trial.get_data().copy()
                    raw = mne.io.RawArray(raw_data, raw_trial.info.copy(), )
                    
                    # Noise covariance matrix estimation from raw data (use all data for better estimate) 
                    # -> Note that this is not ideal, because of data leaking
                    if tmax_noise_cov=='Silences':
                        noise_cov = mne.compute_raw_covariance(
                            raw_silences, 
                            tmax=None, 
                            method=['shrunk', 'empirical'],
                        )
                    elif isinstance(tmax_noise_cov, (int, float)):
                        noise_cov = mne.compute_raw_covariance(
                            raw, 
                            tmax=tmax_noise_cov, 
                            method=['shrunk', 'empirical'],
                        )
                    else: # None
                        noise_cov = mne.compute_raw_covariance(
                            raw, 
                            tmax=None, 
                            method=['shrunk', 'empirical'],
                        )
                    
                    # Map to source space using Minimum Norm Estimation (MNE)
                    inverse_operator = mne.minimum_norm.make_inverse_operator(
                        raw_trial.info, 
                        fwd, 
                        noise_cov, 
                        loose=0.2,
                        depth=0.8
                    )
                    
                    # Configure inverse method parameters
                    method = "dSPM" # "sLORETA" and "eLORETA" are alternatives
                    snr = 3.0  # Standard assumption of Signal-to-Noise Ratio 
                    lambda2 = 1.0 / snr**2

                    # Transform 'raw' from 128 channels to ~10,000 cortical vertices. (stc) - 'SourceEstimate'
                    stc = mne.minimum_norm.apply_inverse_raw(
                        raw, 
                        inverse_operator, 
                        lambda2, 
                        method=method, 
                        pick_ori=None, 
                        verbose=False
                    )
                    
                    # Time series for each ROI
                    roi_time_courses_trial = mne.extract_label_time_course( 
                        stc, 
                        ordered_labels, 
                        src, 
                        mode=time_course_mode, 
                    ).T # --> (n_times, n_rois) 

                    if situation != 'External':
                        roi_time_courses_trial = roi_time_courses_trial[
                            valid_idx, :
                        ]
                    else:
                        roi_time_courses_trial = roi_time_courses_trial
                    roi_time_courses.append(roi_time_courses_trial)
                roi_time_courses = np.concatenate(roi_time_courses, axis=0)

            # Save ROI signals (bilateral average)
            save_dir = Path(rf"data\ROIs\S{session}")
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"s{session}-{channel}-roi-signals-bilateral.pkl"
            save_path_labels = save_path.with_name(save_path.stem + "-labels.pkl")
            
            # Save the ordered ROI names to a text file for reference
            dump_pickle(
                path=save_path,
                obj=roi_time_courses,  
                rewrite=True
            )
            dump_pickle(
                path=save_path_labels,
                obj=ordered_roi_names,
                rewrite=True
            )

            # Try the ROI
            logger = setup_logger(
                name='source_location',
                log_to_file=False,
                log_dir=None,
                level="WARNING",
            )
               
            _ = main_validation(
                stimuli=['Envelope'],
                situations=['External'],
                bands=['Broad'],
                ROI=True,
                save_results=True,
                logger_val=logger
            )
            main_results = main_main(
                same_validation_subjects=False,
                save_results=False,
                ROI=True,
                logger_main=logger,
                info_mne=INFO_MNE_ROI,
            )['External']['Broad']['Envelope']

            correlations_channels.append(main_results['average_correlation_subjects'])
            configurations_channels.append({
                'length': length,
                'situation': situation,
                'time_course_mode': time_course_mode,
                'tmax_noise_cov': tmax_noise_cov_touse,
                'roi_map': ROI_MAP
            })

        correlations.append(
            {
            ch:{
                'correlation': correlations_channels[ch-1],
                'configuration': configurations_channels[ch-1]
            } for ch in [1,2]
            }
        )
        with open(SAVE_CHECKPOINT_PATH, 'w') as f:
            json.dump(
                convert_numpy_keys(correlations), 
                f, 
                indent=4
            )

# roi_names = [
#     'transversetemporal-lh', 'transversetemporal-rh',
#     'superiortemporal-lh', 'superiortemporal-rh',
#     'parsopercularis-lh', 'parsopercularis-rh',
#     'inferiorparietal-lh', 'inferiorparietal-rh'
# ]
# roi_labels = [label for label in labels if label.name in roi_names]

# # Carga el cerebro 'fsaverage'
# brain = mne.viz.Brain(
#     subject=SUBJECT,
#     subjects_dir=SUBJECTS_DIR.parent,
#     hemi='split',  # Mostrar ambos hemisferios
#     surf='pial',  # Superficie pial (la más externa del córtex)
#     background='white' # Fondo blanco para más claridad
# )

# # Mapeo de colores para tus ROIs (así es más robusto que una lista)
# # (A1=Rojo, STG=Azul, IFG=Verde, IPL=Amarillo)
# color_map = {
#     'transversetemporal': 'red',
#     'superiortemporal': 'blue',
#     'parsopercularis': 'green',
#     'inferiorparietal': 'yellow'
# }

# print("Agregando ROIs al cerebro para visualización...")
# # Agrega cada etiqueta (ROI) al cerebro
# for label in roi_labels:
#     # Extrae el nombre base (ej: 'transversetemporal') de 'transversetemporal-lh'
#     base_name = label.name.split('-')[0] 
    
#     # Asigna el color o 'gray' si no lo encuentra
#     color = color_map.get(base_name, 'gray') 
    
#     brain.add_label(
#         label, 
#         color=color, 
#         alpha=0.7 # Un poco de transparencia
#     )

# # Como es un script, lo mejor es guardar imágenes desde vistas específicas
# print("Guardando imágenes de las ROIs...")

# brain.show_view(view='lateral') # 'lateral' (o 'lat') es el string correcto
# brain.save_image('rois_lateral_view.png')

# # 2. Mostrar y guardar la vista medial
# brain.show_view(view='medial') # 'medial' (o 'med')
# brain.save_image('rois_medial_view.png')

# # 3. (Opcional) Mostrar y guardar la vista frontal
# brain.show_view(view='frontal')
# brain.save_image('rois_frontal_view.png')

# brain.close() 

# Invert the problem: from sensors to sources
avg_correlations = []
for cor in correlations:
    for ch in [1, 2]:
        avg_correlations.append(np.mean(cor[str(ch)]['correlation']))
print(
    '\n\n\tMax(avg_correlations): ',     max(avg_correlations)     
)