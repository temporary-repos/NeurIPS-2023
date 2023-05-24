import os

configs = [
    # "DKF",
    # "VRNN",
    # "NeuralODE",
    "DVBF",
]

for config in configs:
    os.system(f"python meta_main.py --config configs/grav16_baselines/{config}.json --train True")
    os.system(f"python meta_main.py --config configs/grav16_baselines/{config}.json --train False")
    os.system(f"python meta_finetune.py --config configs/grav16_baselines/{config}.json")
