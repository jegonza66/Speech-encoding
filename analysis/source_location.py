"""
The objective of this script is to find source location of different ROIs (Regions of Interest):
- Primary Auditory Cortex (A1)
- Superior Temporal Gyrus (STG)
- Inferior Frontal Gyrus (IFG)
- Inferior Parietal Lobule (IPL)
These ROIs will be used to extract time series data from EEG recordings made with a Biosemi 128-channel system.
"""
from pathlib import Path
from mne.datasets import fetch_fsaverage
import numpy as np
import mne

mne.set_log_level("CRITICAL")  # Reduce verbosity for clarity

from utils.general_functions import dump_pickle, load_pickle
from utils.load_utils import get_trials
from utils.load_utils import labeling
import config

# Path to average head already downloaded
subjects_dir = fetch_fsaverage(
    subjects_dir=Path('data/fsaverage'),
    verbose=False
)
subject = 'fsaverage'

for session in config.sessions:
    trials = get_trials(session=session)
    for channel in [1, 2]:
        raw_fnames=[
            Path(f"data/EEG/S{session}/s{session}-{channel}-Trial{trial}-Deci-Filter-Trim-ICA-Pruned.set")
            for trial in trials
        ]
        samples_info = load_pickle(
            path=rf"saves\preprocessed_data\tmin-0.2_tmax0.6\samples_info\External\samples_info_{session}.pkl"
        )
        # Load and concatenate data
        situations = []
        raw = []
        for i, (raw_fname, trial) in enumerate(zip(raw_fnames, trials)):
            raw_trial = mne.io.read_raw_eeglab(
                input_fname=raw_fname, 
                preload=True
            )
            raw_trial = raw_trial.filter(
                l_freq=1,
                h_freq=15,
                method="iir",
                iir_params={
                    "ftype": "cheby2",       # Filter type: Chebyshev Type II
                    "order": 4,              # Filter order
                    "rs": 20,                # Stopband attenuation (dB)
                },
                phase='zero'  # Zero-phase filtering to avoid phase distortion
            )
            raw_trial = raw_trial.resample(
                sfreq=config.sr, 
                npad=0, 
                window='hamming', 
                method='fft'
            )
            speaker = labeling( # 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)
                session=session,
                trial=trial,
                channel=channel,
                sr=config.sr,
            )
            minimum = samples_info['trial_lengths1'][trial]
            situation = speaker==0
            raw_trial = mne.io.RawArray(raw_trial[:, :minimum][0], raw_trial.info)

            situations.extend(situation[:minimum])
            raw.append(raw_trial)
        raw = mne.concatenate_raws(raw)

        # Source location works better with reference to average
        raw.set_eeg_reference(
            'average', 
            projection=True
        )


        # Define source space geometry (cortical surface), BEM, and compute forward solution
        src = mne.setup_source_space(
            subject, 
            spacing='oct6', # standard resolution
            subjects_dir=subjects_dir, 
            add_dist=False
        )

        # Conductivity model (Boundary Element Model - BEM): brain, scull, skin
        model = mne.make_bem_model(
            subject=subject, 
            conductivity=(0.3,), 
            subjects_dir=subjects_dir
        )
        bem = mne.make_bem_solution(model)

        # Alignment between sensors and brain: use 'fsaverage' alignment since we use a template brain
        trans = 'fsaverage' 

        # Finally, compute the forward solution: for this sensor, where in the brain does the signal come from?
        fwd = mne.make_forward_solution(
            raw.info, 
            trans=trans, 
            src=src, 
            bem=bem,
            meg=False, 
            eeg=True, 
        )

        # Invert the problem: from sensors to sources
        
        # Use silences as times for noise covariance estimation
        situations = []
        for trial in trials:
            speaker = labeling( # 3 (both speak), 2 (interlocutor), 1 (channel), 0 (silence)
                session=session,
                trial=trial,
                channel=channel,
                sr=config.sr,
            )
            situations.extend(speaker)
        silences_samples = np.array(situations) == 0
        
        # Noise covariance matrix estimation from raw data (use all data for better estimate) 
        # -> Note that this is not ideal, because of data leaking
        noise_cov = mne.compute_raw_covariance(
            raw, 
            tmax=None, #TODO USAR EL RAW ENTERO.
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
        
        # Configure inverse method parameters
        method = "dSPM" # "sLORETA" and "eLORETA" are alternatives
        snr = 3.0  # Standard assumption of Signal-to-Noise Ratio #TODO isn't this pre-computable?
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
        
        # Extract ROI time series from source estimates
        labels = mne.read_labels_from_annot(
            subject,  
            parc='aparc', # 'Desikan-Killiany' atlas ('aparc')  #TODO CHEQUEAR CUAL USAN EN HABLA --> las parcelaciones se hacen en funcion de que queres ver
            subjects_dir=subjects_dir
        )
        labels_by_name = {label.name: label for label in labels}

        # These are specific to the atlas
        roi_map = {
            'A1': 'transversetemporal', # Giro de Heschl / Córtex Auditivo Primario
            'STG': 'superiortemporal', # Giro Temporal Superior
            'IFG': 'parsopercularis',  # Parte de Broca / Giro Frontal Inferior
            'IPL': 'inferiorparietal'  # Lóbulo Parietal Inferior
            # TODO agregar auditivas tempranas como 
        }
        # (Nota: IFG completo es 'parsopercularis', 'parstriangularis', 'parsorbitalis'. # TODO entender neuroanatomia
        #  Empezar con 'parsopercularis' está bien).

        # Search for MNE labels that match (for both hemispheres)
        rois_to_extract_names = []
        for roi in roi_map.values():
            rois_to_extract_names.extend([f'lh.{roi}', f'rh.{roi}'])
        rois_to_extract_labels = [l for l in labels if l.name in rois_to_extract_names]

        # Time series for each ROI
        roi_time_courses = mne.extract_label_time_course( # --> (n_rois, n_times)
            stc, 
            rois_to_extract_labels, 
            src, 
            mode='mean_flip', #TODO check best mode 
        )
        # 'max'
        # Maximum absolute value across vertices at each time point within each label.
        # 'mean'
        # Average across vertices at each time point within each label. Ignores orientation of sources for standard source estimates, which varies across the cortical surface, which can lead to cancellation. Vector source estimates are always in XYZ / RAS orientation, and are thus already geometrically aligned.
        # 'mean_flip'
        # Finds the dominant direction of source space normal vector orientations within each label, applies a sign-flip to time series at vertices whose orientation is more than 90° different from the dominant direction, and then averages across vertices at each time point within each label.
        # 'pca_flip'
        # Applies singular value decomposition to the time courses within each label, and uses the first right-singular vector as the representative label time course. This signal is scaled so that its power matches the average (per-vertex) power within the label, and sign-flipped by multiplying by np.sign(u @ flip), where u is the first left-singular vector and flip is the same sign-flip vector used when mode='mean_flip'. This sign-flip ensures that extracting time courses from the same label in similar STCs does not result in 180° direction/phase changes.
        # 'auto' (default)
        # Uses 'mean_flip' when a standard source estimate is applied, and 'mean' when a vector source estimate is supplied.
        # None
        # No aggregation is performed, and an array of shape (n_vertices, n_times) is returned. 
        # New in v0.21: Support for 'auto', vector, and volume source estimates.
        
        ordered_labels = []
        ordered_roi_names = [] # Esta será tu lista de "canales"

        for key, roi_name in roi_map.items():
            # Hemisferio Izquierdo (Left)
            lh_name = f'lh.{roi_name}'
            if lh_name in labels_by_name:
                ordered_labels.append(labels_by_name[lh_name])
                # Nomenclatura: 'A1_lh', 'STG_lh', etc.
                ordered_roi_names.append(f'{key}_lh') 
            
            # Hemisferio Derecho (Right)
            rh_name = f'rh.{roi_name}'
            if rh_name in labels_by_name:
                ordered_labels.append(labels_by_name[rh_name])
                # Nomenclatura: 'A1_rh', 'STG_rh', etc.
                ordered_roi_names.append(f'{key}_rh')
        
        # Save ROI signals (bilateral average)
        save_dir = Path(rf"data\ROIs\S{session}").mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"s{session}-{channel}-roi-signals-bilateral.pkl"
        save_path_labels = save_path.with_name(save_path.stem + "-labels.txt")
        
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

        