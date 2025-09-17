"""
Este script tiene que correr para Hubert y Wav2Vec2.
Para cada DNN, debe variar las n_components entre 12 y 26 de a pasos de 2, tomando los extremos.
Debe registrar, únicamente el valor promedio de la correlación en cada capa (ni los pesos, ni los atributos) para ahorrar almacenamiento
La idea es graficar muchas curvas (una por n_components), en las cuales se plotee la correlación promedio por capa
"""
from pathlib import Path
import numpy as np
import shutil
import json
import os

import config
from utils.general_functions import load_pickle, dump_pickle

from load import main_parallel as main_load
from validation import main as main_val
from main import main as main_main

def convert_numpy_keys(obj):
    """Convert numpy integers to Python integers for JSON serialization"""
    if isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_numpy_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_keys(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return obj

# Total: 2*8*23 = 368 analyses
backbones = ["hubert", "wav2vec2"] # 2
components = np.arange(12, 28, 2) # 8
layers = np.arange(1, 24) # 23 

correlations = {
    backbone: {
        n_components: {
                layer: None for layer in layers
            } for n_components in components
    } for backbone in backbones
}
correlations_std = {
    backbone: {
        n_components: {
                layer: None for layer in layers
            } for n_components in components
    } for backbone in backbones
}
save_path = Path("output/mtrf-ridge/analysis/DNN_component_analysis")
try:
    data = load_pickle(path=save_path / "checkpoint_correlations.pkl")
    correlations = data["correlations"]
    correlations_std = data["correlations_std"]
except FileNotFoundError:
    pass

for backbone in backbones:
    for n_components in components:  
        for layer in layers:
            stimuli = f'{n_components}DNNs{layer}-{backbone}'

            if correlations[backbone][n_components][layer] is not None:
                print(f"Skipping already computed {stimuli}")
                continue
            
            print(
                f'\n\n\n\tProcessing {n_components} components, layer {layer}, backbone {backbone}\n',
                f'\n\tStimuli:\t{stimuli}\n'
            )
            # Run the validation script with arguments for backbone and n_components
            _ = main_load(
                situations=['External'],
                bands=['Broad'],
                stimuli=[stimuli],
                save_results=True,
                number_of_workers=7
            )

            validation_path = Path(rf'output\mtrf-ridge\External\validation\stims_Standarize_EEG_Standarize\tmin-0.2_tmax0.6\Broad\{stimuli}')
            if not validation_path.exists():
                alphas = main_val(
                    situations=['External'],
                    stimuli=[stimuli],
                    bands=['Broad'],
                    save_results=True,
                    no_figures=True
                )['External']['Broad'][stimuli]
            else:
                alphas = load_pickle(path=validation_path / 'corr_limit_0.01.pkl')
                print(f"Validation found for {stimuli}, loading from disk.")

            alphas_total = []
            for session in config.sessions:
                for subject in [1, 2]:
                    alphas_total.append(alphas[session][subject])
            alphas_total = np.array(alphas_total)
            set_alpha = 10**(np.median(np.log10(alphas_total)))
            
            main_results = main_main(
                situations=['External'],
                stimuli=[stimuli],
                bands=['Broad'],
                save_results=False,
                set_alpha=set_alpha,
                no_figures=True
            )['External']['Broad'][stimuli]
            correlations[backbone][n_components][layer] = main_results['average_correlation_subjects'].mean(axis=1)
            correlations_std[backbone][n_components][layer] = main_results['average_correlation_subjects'].std(axis=1)/np.sqrt(128)

            # Save checkpoint
            
            save_path.mkdir(parents=True, exist_ok=True)
            dump_pickle(
                path=save_path / "checkpoint_correlations.pkl",
                obj={
                    "correlations": correlations,
                    "correlations_std": correlations_std
                },
                rewrite=True,
                verbose=True
            )
            # Save json to legible format
            with open(save_path / "checkpoint_correlations.json", 'w') as f:
                json_data = {
                    "correlations": convert_numpy_keys(correlations),
                    "correlations_std": convert_numpy_keys(correlations_std)
                }
                json.dump(json_data, f, indent=4)
            # Remove saved data to save space
            try:
                dir_to_remove = os.path.normpath(rf'saves\preprocessed_data\tmin-0.2_tmax0.6\{stimuli}')
                shutil.rmtree(dir_to_remove, ignore_errors=True)
            except Exception as e:
                raise(f"Could not remove directory {dir_to_remove}: {e}")
            
# Make plots
