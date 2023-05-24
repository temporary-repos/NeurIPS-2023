import os
import json
import numpy as np
import pandas as pd

from tensorboard.backend.event_processing import event_accumulator


def parse_tensorboard(path):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )

    _absorb_print = ea.Reload()
    return pd.DataFrame(ea.Scalars("train_recon_mse")), pd.DataFrame(ea.Scalars("total_num_parameters"))


base_exps = [
    # "additive_additive",
    # "multiplicative_additive",

    "additive_multiplicative",
    "multiplicative_multiplicative"
]

for dim in range(1, 21):
    print(f"---- Dim {dim} ----")
    for base_exp in base_exps:
        print(f"---- {base_exp} ----")
        for folder in sorted(os.listdir(f"experiments/{base_exp}/"), key=lambda i: int(i.split("_")[3].replace("hidden", ""))):

            if int(folder.split('_')[2].replace("dim", "")) != dim:
                continue

            tb_file = None
            for file in os.listdir(f"experiments/{base_exp}/{folder}/{base_exp.split('_')[0]}/version_1/"):
                if "event" in file:
                    tb_file = file
                    break

            tb, params = parse_tensorboard(f"experiments/{base_exp}/{folder}/{base_exp.split('_')[0]}/version_1/{tb_file}")
            if len(np.where(tb.value < 1e-3)[0]) > 0:
                step = np.where(tb.value < 1e-3)[0][0]
            else:
                step = "None"


            try:
                # Load in the json of test results
                metric = json.load(open(f"experiments/{base_exp}/{folder}/{base_exp.split('_')[0]}/version_1/test_files/test_metrics.json", 'r'))['recon_mse_mean']
            except FileNotFoundError:
                metric = 999

            print(f"{base_exp.split('_')[0].title() + ' ' + folder.split('_')[3]:25}: {metric:0.5f} | {step:5} epoch | {int(params.value[0]):9} params")

        print("")