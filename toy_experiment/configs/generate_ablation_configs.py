import os
import json

# Define the number of samples per environment and how many environments to use
dims = range(1, 21)
for dim in dims:
    model_type = "multiplicative"
    func_type = "additive"
    experiment_name = f"{model_type}_{func_type}"
    model_hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2042, 4084, 8126]

    # Load in the sample config file
    sample_config = json.load(open("ExampleConfig.json", 'r'))

    # Name of the ablation dir
    model_name = f"model{model_type.title()}_func{func_type.title()}_dim{dim}"

    # Make experiment folders to hold the config files
    if not os.path.exists(f"{experiment_name}/"):
        os.mkdir(f"{experiment_name}/")

    if not os.path.exists(f"{experiment_name}/{model_name}/"):
        os.mkdir(f"{experiment_name}/{model_name}/")

    # Inside the data experiments folder, make the overall folder
    if not os.path.exists(f"../experiments/{experiment_name}/"):
        os.mkdir(f"../experiments/{experiment_name}/")

    # For each sample size set, generate the config over the given environments
    for hidden_dim in model_hidden_sizes:
        exptype = f"model{model_type.title()}_func{func_type.title()}_dim{dim}_{hidden_dim}hidden"

        sample_config["exptype"] = f"{experiment_name}/{exptype}"
        sample_config["ckpt_path"] = f"experiments/{experiment_name}/{exptype}/{model_type}/version_1/"

        sample_config["model"] = model_type
        sample_config["func_type"] = func_type

        sample_config["hidden_dim"] = hidden_dim
        sample_config["in_dim"] = dim
        sample_config["out_dim"] = dim
        sample_config["control_dim"] = dim

        with open(f"{experiment_name}/{model_name}/{exptype}.json", 'w') as f:
            json.dump(sample_config, f, indent=4)
