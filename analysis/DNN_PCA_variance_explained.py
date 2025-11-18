"""
This script analyzes the variance explained by PCA components
for different DNN backbones and layers. It loads precomputed variance
"""
import matplotlib.pyplot as plt
from matplotlib import rc
from pathlib import Path
import scienceplots
import numpy as np

rc('text', usetex=True)
plt.style.use(['science'])

from load import main_parallel

NUUMBER_COMPONENTS = [int(n) for n in np.logspace(np.log10(5), np.log10(512), num=20)]
MATRIX_REDUCTION_PATH = Path("data/DNNs_cache/matrix_reduction/")
VARIANCE_EXPLAINED_PATH = Path("figures/analysis/DNN_PCA_variance_explained/variance_explained_over_components.png")
VARIANCE_EXPLAINED_PATH.parent.mkdir(parents=True, exist_ok=True)

variance_explained = {
    component: {} for component in NUUMBER_COMPONENTS
}
for component in NUUMBER_COMPONENTS:
    for backbone in MATRIX_REDUCTION_PATH.iterdir():
        backbone_name = backbone.name
        if not backbone_name.endswith(f'-{component}'):
            continue
        else:
            backbone_name = backbone_name.split('-')[0]
            variance_explained[component][backbone_name] = []
        for layer in backbone.iterdir():
            try:
                variance_files = [file for file in layer.iterdir() if file.suffix == '.txt']
                for file in variance_files:
                    with open(file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if "Explained variance (cumulative)" in line:
                                # extract number after the colon, remove surrounding quotes and whitespace
                                num = float(line.strip().split(':', 1)[1].strip().strip("'\""))
                                variance_explained[component][backbone_name].append(num)
            except Exception as e:
                print(f"Failed for component {component}, backbone {backbone_name}, layer {layer.name}: {e}")
                stimuli = [
                    f'{component}DNNs{layer}-{backbone}'
                    for backbone in ['wav2vec2']
                    for layer in np.arange(1,24)
                ]
                main_parallel(
                    situations = ['External'],
                    stimuli = stimuli,
                    bands = ['Broad'],
                    number_of_workers = 12,
                    save_results = False
                )
                # Retry loading after computation
                variance_files = [file for file in layer.iterdir() if file.suffix == '.txt']
                for file in variance_files:
                    with open(file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            if "Explained variance (cumulative)" in line:
                                # extract number after the colon, remove surrounding quotes and whitespace
                                num = float(line.strip().split(':', 1)[1].strip().strip("'\""))
                                variance_explained[component][backbone_name].append(num)
        variance_explained[component][backbone_name] = np.array(variance_explained[component][backbone_name])

# Plot mean and std variances for wavlm and hubert across possible components:
plt.figure(figsize=(10, 6))
for color, backbone in enumerate(['hubert', 'wav2vec2', 'wavlm']):
    means = []
    sems = []
    for component in NUUMBER_COMPONENTS:
        try: 
            mean_var = 100*variance_explained[component][backbone].mean()
            sems_ = 100*variance_explained[component][backbone].std(ddof=1)
            means.append(mean_var)
            sems.append(sems_)
        except Exception as e:
            print(f"Skipping {backbone} with {component} components due to missing component: {e}")
            means.append(np.nan)
            sems.append(np.nan)
    means = np.array(means)
    sems = np.array(sems)
    plt.scatter(NUUMBER_COMPONENTS, means, label=backbone.capitalize(), color=f'C{color}')
    plt.fill_between(
        NUUMBER_COMPONENTS, 
        means-sems, means+sems, 
        alpha=0.2, color=f'C{color}'
    )

# keep default tick locations but ensure there's a tick at 25
plt.xticks([0,25,50,75,100,200,300,400,500], fontsize=14)
plt.tick_params(axis='y', which='major', labelsize=14)

plt.xlabel('Number of PCA Components', fontsize=14)
plt.ylabel(r'Explained Variance (\%)', fontsize=14)
plt.legend(fontsize=14, loc='lower right', frameon=True)
plt.grid()

plt.savefig(
    VARIANCE_EXPLAINED_PATH,
    dpi=300,
    bbox_inches='tight'
)
print(f"Figure saved in {VARIANCE_EXPLAINED_PATH.resolve()}")