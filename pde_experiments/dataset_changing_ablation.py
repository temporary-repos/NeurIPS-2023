import os

base_path = "configs/dataset_ablation/metahyper_8hidden_regTrue/"
for idx, config in enumerate(os.listdir(base_path)):
    if os.path.exists(f"experiments/{config.replace('.json', '')}/"):
        continue

    os.system(f"python pde_main.py --config {base_path}/{config} --train True")
    os.system(f"python pde_main.py --config {base_path}/{config} --train False")
    os.system(f"python phase_space_example.py --config {base_path}/{config}")
