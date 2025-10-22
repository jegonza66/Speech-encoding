import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from load import main_parallel
import config


n_components = [int(n) for n in np.logspace(np.log10(5), np.log10(512), num=20)]
for n_components in n_components:
    stimuli = [
        f'{n_components}DNNs{layer}-{backbone}'
        for backbone in ['wavlm', 'hubert']#, 'wav2vec2']
        for layer in np.arange(1,24)
    ]
    main_parallel(
        situations = ['External'],
        stimuli = stimuli,
        bands = ['Broad'],
        number_of_workers = 12,
        save_results = False
    )

matrix_reduction_path = Path("data/DNNs_cache/matrix_reduction/")
# n_components = sorted(list(set(int(component.name.split('-')[-1]) for component in matrix_reduction_path.iterdir())))

variance_explained = {
    component: {} for component in n_components
}

for component in n_components:
    for backbone in matrix_reduction_path.iterdir():
        backbone_name = backbone.name
        if not backbone_name.endswith(f'-{component}'):
            continue
        else:
            backbone_name = backbone_name.split('-')[0]
            variance_explained[component][backbone_name] = []
        for layer in backbone.iterdir():
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

interest_backbone = 'wavlm'
interest_component = 128
mean_var = 100*variance_explained[interest_component][interest_backbone].mean()
std_var = 100*variance_explained[interest_component][interest_backbone].std()
print(
    f"Variance explained for {interest_backbone} with {interest_component} components:",
    rf" ({mean_var:.2f} ± {std_var:.2f})%"
)

# Plot mean and std variances for wavlm and hubert across possible components:
plt.figure(figsize=(10, 6))
for backbone in ['wavlm', 'hubert']:
    means = []
    stds = []
    for component in n_components:
        try: 
            mean_var = 100*variance_explained[component][backbone].mean()
            std_var = 100*variance_explained[component][backbone].std()
            means.append(mean_var)
            stds.append(std_var)
        except Exception as e:
            print(f"Skipping {backbone} with {component} components due to missing component: {e}")
            means.append(np.nan)
            stds.append(np.nan)
    means = np.array(means)
    stds = np.array(stds)
    plt.scatter(n_components, means, label=backbone)
    plt.fill_between(n_components, means - stds, means + stds, alpha=0.2)
plt.xlabel('Number of PCA Components')
plt.ylabel('Explained Variance (%)')
plt.title('Explained Variance vs Number of PCA Components')
plt.legend()
plt.grid()
# plt.savefig('DNN_PCA_variance_explained.png')
plt.show()

    
