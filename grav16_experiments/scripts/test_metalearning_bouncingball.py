"""
@file test_metalearning_mixedphysics.py

Handles iterating through all the meta-learning testing sets for Mixed Physics and
aggregating them into meaned sets across each physics type and globally.
"""
import os
import json
import numpy as np

models = [
    # "lstm_meta",
    # "vrnn_meta",
    # "rgnres_meta",
    # "node_meta",
    # "nfssm_localglobal_meta",
    # "dvbf_meta",
    # "leads_meta"
    # "dkf_meta"
    # "dkf_snp_meta"
    # "coda_meta"
    "metahyperssm"
]

ckpt_paths = [
    # "experiments/grav16_lstm/lstm_meta/version_1/",
    # "experiments/grav16_vrnn/vrnn_meta/version_1",
    # "experiments/grav16_rgnres/rgnres_meta/version_1",
    # "experiments/grav16_node/node_meta/version_1",
    "experiments/grav16_nfssm_localglobal/nfssm_localglobal_meta/version_1",
    # "experiments/grav16_dvbf/dvbf_meta/smallerODE",
    # "experiments/grav16_leads/leads_meta/smallerODE"
    # "experiments/grav16_dkf/dkf_meta/version_1"
    # "experiments/grav16_dkf_snp/dkf_snp_meta/version_1"
    # "experiments/grav16_coda/coda_meta/version_4"
    "experiments/grav16_metahyperssm/metahyperssm/version_1/"
]

checkpts = [
    "None",
    "None",
    "None",
    "None",
    "None"
]

queries = ["train", "valid"] + \
          [f"qry_{i}" for i in [0, 1, 2, 3, 5, 6, 7, 8, 11, 13, 14, 15]] + \
          [f"unknown_qry_{i}" for i in [4, 9, 10, 12]]
for model, ckpt_path, checkpt in zip(models, ckpt_paths, checkpts):
    # Get metrics on all queries
    for query in queries:
        os.system(f"python meta_main.py --train False --model {model} --ckpt_path {ckpt_path} --dataset_ver bouncing_ball/bouncingball_16 --dataset_split {query} --checkpt {checkpt}")

    # Aggregate all metrics into list
    metrics = {
        'mse_recon_mean': [], 'mse_recon_std': [],
        'vpt_mean': [], 'vpt_std': [],
        'dst_mean': [], 'dst_std': [],
        'vpd_mean': [], 'vpd_std': [],
    }
    for query in queries[2:]:
        metric = json.load(open(f"{ckpt_path}/test_{query}/test_{query}_metrics.json", 'r'))

        for m in metric.keys():
            if m in metrics.keys():
                metrics[m].append(metric[m])

    print(metrics)

    """ Global metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_all_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean']):0.4f}({np.mean(metrics['mse_recon_std']):0.4f}),"
                f"{np.mean(metrics['vpt_mean']):0.3f}({np.mean(metrics['vpt_std']):0.3f}),"
                f"{np.mean(metrics['dst_mean']):0.3f}({np.mean(metrics['dst_std']):0.3f}),"
                f"{np.mean(metrics['vpd_mean']):0.3f}({np.mean(metrics['vpd_std']):0.3f})")

    """ Known grav metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_known_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean'][:12]):0.4f}({np.mean(metrics['mse_recon_std'][:12]):0.4f}),"
                f"{np.mean(metrics['vpt_mean'][:12]):0.3f}({np.mean(metrics['vpt_std'][:12]):0.3f}),"
                f"{np.mean(metrics['dst_mean'][:12]):0.3f}({np.mean(metrics['dst_std'][:12]):0.3f}),"
                f"{np.mean(metrics['vpd_mean'][:12]):0.3f}({np.mean(metrics['vpd_std'][:12]):0.3f})")

    """ Unknown grav metrics """
    # Save metrics to an easy excel conversion style
    with open(f"{ckpt_path}/test_unknown_excel.txt", 'a') as f:
        f.write(f"\n{np.mean(metrics['mse_recon_mean'][12:]):0.4f}({np.mean(metrics['mse_recon_std'][12:]):0.4f}),"
                f"{np.mean(metrics['vpt_mean'][12:]):0.3f}({np.mean(metrics['vpt_std'][12:]):0.3f}),"
                f"{np.mean(metrics['dst_mean'][12:]):0.3f}({np.mean(metrics['dst_std'][12:]):0.3f}),"
                f"{np.mean(metrics['vpd_mean'][12:]):0.3f}({np.mean(metrics['vpd_std'][12:]):0.3f})")
