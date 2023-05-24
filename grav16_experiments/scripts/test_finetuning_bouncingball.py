"""

"""
import os
import json
import shutil

import numpy as np

models = [
    # "node_meta",
    # "vrnn_meta",
    # "rgnres_meta",
    # "lstm_meta",
    # "dvbf_meta"
    # "leads_meta"
    # "dkf_meta"
    "coda_meta"
]

ckpt_paths = [
    # "experiments/grav16_node/node_meta/version_1",
    # "experiments/grav16_vrnn/vrnn_meta/version_1",
    # "experiments/grav16_leads/leads_meta/smallerODE"
    # "experiments/grav16_rgnres/rgnres_meta/version_1"
    # "experiments/grav16_lstm/lstm_meta/version_1",
    # "experiments/grav16_dvbf/dvbf_meta/smallerODE"
    # "experiments/grav16_dkf/dkf_meta/version_1"
    "experiments/grav16_coda/coda_meta/version_4"
]

checkpts = [
    "None",
    "None"
]

queries = [f"qry_{i}" for i in [0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 15]] + \
          [f"unknown_qry_{i}" for i in [4, 9, 10, 12]]

supports = [f"spt_{i}" for i in [0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 15]] + \
          [f"unknown_spt_{i}" for i in [4, 9, 10, 12]]

for model, ckpt_path, checkpt in zip(models, ckpt_paths, checkpts):
    # Get metrics on all queries
    for query, support in zip(queries, supports):
        os.system(f"python meta_finetune.py --model {model} --ckpt_path {ckpt_path} --dataset_ver bouncing_ball/bouncingball_16 --dataset_split {support} --checkpt {checkpt} --exptype grav16_{model}_finetuned_{support} --batch_size 16 --dataset_percent 0.032 --meta False")
        os.system(f"python meta_test.py --model {model} --ckpt_path experiments/grav16_{model}_finetuned_{support}/{model}/version_1/ --dataset_ver bouncing_ball/bouncingball_16 --dataset_split {query} --meta False")
        os.rename(f"experiments/grav16_{model}_finetuned_{support}/{model}/version_1/test_{query}/", f"{ckpt_path}/test_finetuned_{query}//")
        shutil.rmtree(f"experiments/grav16_{model}_finetuned_{support}/")

    # Aggregate all metrics into list
    metrics = {
        'mse_recon_mean': [], 'mse_recon_std': [],
        'vpt_mean': [], 'vpt_std': [],
        'dst_mean': [], 'dst_std': [],
        'vpd_mean': [], 'vpd_std': [],
    }
    for query in queries:
        metric = json.load(open(f"{ckpt_path}/test_finetuned_{query}/test_{query}_metrics.json", 'r'))

        for m in metric.keys():
            if m in metrics.keys():
                metrics[m].append(metric[m])

    print(metrics)

    """ Global metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_all_finetuned_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean']):0.4f}({np.mean(metrics['mse_recon_std']):0.4f}),"
                f"{np.mean(metrics['vpt_mean']):0.3f}({np.mean(metrics['vpt_std']):0.3f}),"
                f"{np.mean(metrics['dst_mean']):0.3f}({np.mean(metrics['dst_std']):0.3f}),"
                f"{np.mean(metrics['vpd_mean']):0.3f}({np.mean(metrics['vpd_std']):0.3f})")

    """ Known grav metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_known_finetuned_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean'][:12]):0.4f}({np.mean(metrics['mse_recon_std'][:12]):0.4f}),"
                f"{np.mean(metrics['vpt_mean'][:12]):0.3f}({np.mean(metrics['vpt_std'][:12]):0.3f}),"
                f"{np.mean(metrics['dst_mean'][:12]):0.3f}({np.mean(metrics['dst_std'][:12]):0.3f}),"
                f"{np.mean(metrics['vpd_mean'][:12]):0.3f}({np.mean(metrics['vpd_std'][:12]):0.3f})")

    """ Unknown grav metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_finetuned_unknown_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean'][12:]):0.4f}({np.mean(metrics['mse_recon_std'][12:]):0.4f}),"
                f"{np.mean(metrics['vpt_mean'][12:]):0.3f}({np.mean(metrics['vpt_std'][12:]):0.3f}),"
                f"{np.mean(metrics['dst_mean'][12:]):0.3f}({np.mean(metrics['dst_std'][12:]):0.3f}),"
                f"{np.mean(metrics['vpd_mean'][12:]):0.3f}({np.mean(metrics['vpd_std'][12:]):0.3f})")
