import os
import json

# Define the number of samples per environment and how many environments to use
sample_sizes = [8]
indomainenvs = [0, ]
outdomainenvs = [0, ]

experiment_name = "reg_ablation"
model_hidden_size = 8  # 8 for HyperNet 16 for NODE

# Define which regularization term to use
# reg = 'No'
# reg = 'L1'
reg = 'L2'
# reg = 'GP'
# reg = 'cGan'
# reg = 'MX'
model_name = f"metahyper_{model_hidden_size}hidden"

# Load in the sample config file
sample_config = json.load(open("MetaHyperNet.json", 'r'))

# Make experiment folders to hold the config files
if not os.path.exists(f"{experiment_name}/"):
    os.mkdir(f"{experiment_name}/")

if not os.path.exists(f"{experiment_name}/{model_name}/"):
    os.mkdir(f"{experiment_name}/{model_name}/")

# Inside the data experiments folder, make the overall folder
if not os.path.exists(f"../experiments/{experiment_name}/"):
    os.mkdir(f"../experiments/{experiment_name}/")

# For each sample size set, generate the config over the given environments
for sample_size in sample_sizes:
    for indomainenv in indomainenvs:
        for outdomainenv in outdomainenvs:
            exptype = f"{model_name}_reg{reg}_{sample_size}perenv_{indomainenv}indomainenvs_{outdomainenv}outdomainenvs"

            sample_config["exptype"] = f"{experiment_name}/{exptype}"
            sample_config["ckpt_path"] = f"experiments/{experiment_name}/{exptype}/{model_name.split('_')[0]}/version_1/"
            sample_config["dataset"] = f"lotka_volterra_{sample_size}perenv_{indomainenv}indomainenvs_{outdomainenv}outdomainenvs"
            sample_config["dataset_ver"] = f"lotka_volterra_{sample_size}perenv_{indomainenv}indomainenvs_{outdomainenv}outdomainenvs"
            sample_config["num_hidden"] = model_hidden_size
            sample_config["model"] = model_name.split("_")[0]

            if reg == 'No':
                sample_config["hypernet_reg_beta"] = 0
                sample_config["code_reg_beta"] = 0
                sample_config["gradient_beta"] = 0
                sample_config["cgan_reg_beta"] = 0
                sample_config["mixed_up_beta"] = 0
            elif reg == 'L1':
                sample_config["hypernet_reg_beta"] = 1e-6
                sample_config["code_reg_beta"] = 0
                sample_config["gradient_beta"] = 0
                sample_config["cgan_reg_beta"] = 0
                sample_config["mixed_up_beta"] = 0
            elif reg == 'L2':
                sample_config["hypernet_reg_beta"] = 0
                sample_config["code_reg_beta"] = 1e-4
                sample_config["gradient_beta"] = 0
                sample_config["cgan_reg_beta"] = 0
                sample_config["mixed_up_beta"] = 0
            elif reg == 'GP':
                sample_config["hypernet_reg_beta"] = 0
                sample_config["code_reg_beta"] = 0
                sample_config["gradient_beta"] = 10
                sample_config["cgan_reg_beta"] = 0
                sample_config["mixed_up_beta"] = 0
            elif reg == 'cGan':
                sample_config["hypernet_reg_beta"] = 0
                sample_config["code_reg_beta"] = 0
                sample_config["gradient_beta"] = 0
                sample_config["cgan_reg_beta"] = 1e-2
                sample_config["mixed_up_beta"] = 0
            elif reg == 'MX':
                sample_config["hypernet_reg_beta"] = 0
                sample_config["code_reg_beta"] = 0
                sample_config["gradient_beta"] = 0
                sample_config["cgan_reg_beta"] = 0
                sample_config["mixed_up_beta"] = 1e-2

            with open(f"{experiment_name}/{model_name}/{exptype}.json", 'w') as f:
                json.dump(sample_config, f, indent=4)
