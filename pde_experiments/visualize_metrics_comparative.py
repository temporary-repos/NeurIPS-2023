import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Which model to use, NODE or HyperNet
exp_folders = [
    "metanode_regFalse",
    "metahyper_regTrue",
    # "metargnres_regTrue"
]

colors = [
    "pastel",
    "Set2",
    "flare"
]

length_st = 6
num_samples = 4
plt.figure(figsize=(10, 5))

for exp_folder, color in zip(exp_folders, colors):
    # Dictionaries across sample sizes and environment sizes
    environments_mean = {
        "validation_mse": [0] * length_st,
        "indomain": [0] * length_st,
        "outdomain": [0] * length_st
    }

    environments_std = {
        "validation_mse": [0] * length_st,
        "indomain": [0] * length_st,
        "outdomain": [0] * length_st
    }

    model_name, reg_used = exp_folder.split("_")
    for folder in os.listdir(f"experiments/{exp_folder}/"):
        if model_name not in folder or "lips" in folder or reg_used not in folder:
            continue

        # Extract parameters of test
        split_folder = folder.split("_")
        num_hidden = int(split_folder[1].replace('hidden', ''))
        regularized = bool(split_folder[2].replace('reg', ''))
        samples_per_env = int(split_folder[3].replace('perenv', ''))
        indomain_envs = int(split_folder[4].replace('indomainenvs', ''))
        outdomain_envs = int(split_folder[5].replace('outdomainenvs', ''))

        if samples_per_env != num_samples:
            continue

        # Get rid of the in-between environment training setups
        if indomain_envs == 4 or outdomain_envs == 4:
            continue

        print(num_hidden, regularized, samples_per_env, indomain_envs, outdomain_envs)

        # Get metrics of current test
        try:
            metrics = json.load(open(f"experiments/{exp_folder}/{folder}/{model_name}/version_1/test_files/test_mapegrid_excel.json"))
            formatted = open(f"experiments/{exp_folder}/{folder}/{model_name}/version_1/test_files/test_excel.txt").read()
        except FileNotFoundError:
            continue
        print(formatted)

        # Get the environment index in the dictionaries based on environments
        if indomain_envs == 0 and outdomain_envs == 0:
            env_idx = 0
        elif indomain_envs == 8 and outdomain_envs == 0:
            env_idx = 1
        elif indomain_envs == 0 and outdomain_envs == 8:
            env_idx = 2
        elif indomain_envs == 0 and outdomain_envs == 16:
            env_idx = 3
        elif indomain_envs == 8 and outdomain_envs == 8:
            env_idx = 4
        elif indomain_envs == 8 and outdomain_envs == 16:
            env_idx = 5

        # Add metrics to environment dictionary
        # environments_mean["validation_mse"][env_idx] = metrics['recon_mse_mean']
        # environments_std["validation_mse"][env_idx] = metrics['recon_mse_std']

        environments_mean["indomain"][env_idx] = metrics['rel_indomain_mean']
        environments_std["indomain"][env_idx] = metrics['rel_indomain_std']

        environments_mean["outdomain"][env_idx] = metrics['rel_outdomain_mean']
        environments_std["outdomain"][env_idx] = metrics['rel_outdomain_std']

    # PLot the metrics across the different environment setups and data-per-env setup
    data = {'vals': np.concatenate((
                            # environments_mean["validation_mse"],
                            environments_mean["indomain"],
                            environments_mean["outdomain"]
                    )),
            'x': np.concatenate((
                # range(length_st),
                range(length_st), range(length_st)
            )),
            'cat':
                # [f'{model_name}']*length_st +
                   [f'{model_name}']*length_st + [f'{model_name}']*length_st,
            'type':
                # ['MSE']*length_st +
                ['In-Domain']*length_st + ['Out-Domain']*length_st
    }

    # plot the data
    sns.lineplot(data=data, x='x', y='vals', hue='cat', style='type', markers='o', palette=color)


plt.xticks(range(6), ["9/0", "17/0", "9/8", "9/16", "17/8", "17/16"])
plt.xlabel("Base Envs | In-Domain Envs | Out-Domain Envs")

# move the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# plt.ylim([0.02, 0.11])
plt.title(f"Dataset Size/Range Ablation")
plt.grid()
plt.tight_layout()
plt.show()
