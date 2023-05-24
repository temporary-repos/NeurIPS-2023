import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Which model to use, NODE or HyperNet
model_name = "metanode"
reg_used = "regFalse"
exp_folder = "metanode_regFalse"

# Dictionaries across sample sizes and environment sizes
environments_mean = {
    4: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    },
    8: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    },
    16: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    }
}

environments_std = {
    4: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    },
    8: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    },
    16: {
        "validation_mse": [0, 0, 0, 0],
        "indomain": [0, 0, 0, 0],
        "outdomain": [0, 0, 0, 0]
    }
}

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

    # Get rid of the in-between environment training setups
    if indomain_envs == 4 or outdomain_envs == 4:
        continue

    print(num_hidden, regularized, samples_per_env, indomain_envs, outdomain_envs)

    # Get metrics of current test
    try:
        metrics = json.load(open(f"experiments/{exp_folder}/{folder}/{model_name}/version_1/test_files/test_metrics.json"))
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
    elif indomain_envs == 8 and outdomain_envs == 8:
        env_idx = 3

    # Add metrics to environment dictionary
    environments_mean[samples_per_env]["validation_mse"][env_idx] = metrics['recon_mse_mean']
    environments_std[samples_per_env]["validation_mse"][env_idx] = metrics['recon_mse_std']

    environments_mean[samples_per_env]["indomain"][env_idx] = metrics['rel_indomain_mean']
    environments_std[samples_per_env]["indomain"][env_idx] = metrics['rel_indomain_std']

    environments_mean[samples_per_env]["outdomain"][env_idx] = metrics['rel_outdomain_mean']
    environments_std[samples_per_env]["outdomain"][env_idx] = metrics['rel_outdomain_std']

# PLot the metrics across the different environment setups and data-per-env setup
plt.figure()
length_st = 4
num_samples = 4
data = {'vals': np.concatenate((
                        environments_mean[num_samples]["validation_mse"],
                        environments_mean[num_samples]["indomain"],
                        environments_mean[num_samples]["outdomain"]
                )),
        'x': np.concatenate((range(4), range(4), range(4))),
        'cat': [f'{num_samples}/ENV']*length_st + [f'{num_samples}/ENV']*length_st + [f'{num_samples}/ENV']*length_st,
        'type': ['MSE']*length_st + ['In-Domain']*length_st + ['Out-Domain']*length_st
}

# plot the data
p = sns.lineplot(data=data, x='x', y='vals', hue='cat', style='type', markers='o')
plt.xticks(range(4), ["0|0", "8|0", "0|8", "8|8"])
plt.xlabel("In-Domain Envs | Out-Domain Envs")

# move the legend
p.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.ylim([0, 0.11])
plt.title(f"Metric outputs for {model_name}")
plt.grid()
plt.tight_layout()
plt.show()
