#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot Liang Index, Correlation and Transfer Entropy from text files

Updated: 01/07/2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import re


# === Funzioni di parsing ===
def parse_liang_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    data = {}
    current_key = None
    buffer = []

    for line in lines:
        line = line.strip()
        if line.startswith("===") or line == "":
            continue
        elif line.endswith(":"):
            if current_key and buffer:
                data[current_key] = "\n".join(buffer)
                buffer = []
            current_key = line[:-1].strip()
        elif current_key:
            buffer.append(line)
    if current_key and buffer:
        data[current_key] = "\n".join(buffer)

    def str_to_matrix(s):
        s_clean = s.replace('[', '').replace(']', '')
        return np.array([list(map(float, row.split())) for row in s_clean.strip().split('\n')])
    
    T = str_to_matrix(data['T matrix'])
    sig_T = str_to_matrix(data['Significance (T)'])
    tau = str_to_matrix(data['tau matrix'])
    sig_tau = str_to_matrix(data['Significance tau'])
    R = str_to_matrix(data['R matrix'])
    sig_R = str_to_matrix(data['Significance R'])

    return T, tau, R, sig_T, sig_tau, sig_R


def parse_te_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    te_dict = {}

    # Regex pattern: multiline, matches TE, p-value (even if line breaks occur)
    pattern = r"TE \(X(\d) → X(\d)\): ([\deE+.-]+) nats,.*?p-value: ([\deE+.-]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    for src, tgt, te_val, p_val in matches:
        te_dict[f'X{src}->X{tgt}'] = {
            'TE': float(te_val),
            'p': float(p_val)
        }

    return te_dict


# === Plot Functions ===
def plot_liang_matrices(liang_file, label_names, save_fig=False):
    T, tau, R, sig_T, sig_tau, sig_R = parse_liang_file(liang_file)
    nvar = T.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.4)
    sns.set(font_scale=1.4)

    # Correlation matrix R
    cmap_R = plt.cm.bwr
    R_masked = np.where(sig_R == 1, R, np.nan)
    R_annotations = np.where(sig_R == 1, np.round(R_masked, 2).astype(str), "")
    R_plot = sns.heatmap(R_masked, annot=R_annotations, fmt='', cmap=cmap_R, ax=ax[0],
                         xticklabels=label_names, yticklabels=label_names, vmin=-1, vmax=1,
                         square=True, cbar_kws={'orientation': 'horizontal', 'label': 'R'}, linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 14})
    R_plot.set_title('Correlation matrix $R$, $\Delta t = 0.01$')
    R_plot.xaxis.set_ticks_position('top')
    R_plot.set_xticklabels(R_plot.get_xmajorticklabels(), fontsize=14)
    R_plot.set_yticklabels(R_plot.get_ymajorticklabels(), fontsize=14)

    for j in range(nvar):
        for i in range(nvar):
            if sig_R[j, i] == 1:
                R_plot.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='black', linewidth=2))

    # Tau matrix
    cmap_tau = plt.cm.YlOrRd
    tau_abs = np.abs(tau)
    tau_masked = np.where(sig_tau == 1, tau_abs, np.nan)
    tau_annotations = np.where(sig_tau == 1, np.round(tau_abs, 2).astype(str), "")
    tau_plot = sns.heatmap(tau_masked, annot=tau_annotations, fmt='', cmap=cmap_tau, ax=ax[1],
                           xticklabels=label_names, yticklabels=label_names, vmin=0, vmax=60,
                           square=True, cbar_kws={'orientation': 'horizontal', 'label': r'$|\tau|$ (%)'}, linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 14})
    tau_plot.set_title('Causality strength $|\\tau|$, $\\Delta t=0.01$')
    tau_plot.xaxis.set_ticks_position('top')
    tau_plot.set_xticklabels(tau_plot.get_xmajorticklabels(), fontsize=14)
    tau_plot.set_yticklabels(tau_plot.get_ymajorticklabels(), fontsize=14)

    for j in range(nvar):
        for i in range(nvar):
            if sig_tau[j, i] == 1:
                tau_plot.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='blue', linewidth=2))

    if save_fig:
        plt.savefig('liang_causality_R_tau_bias1.png', dpi=300, bbox_inches='tight')
    
    #plt.show()


def plot_te_matrix(te_file, label_names, save_fig=False):
    match = re.search(r'embedding(\d+)', te_file)
    k = int(match.group(1)) if match else None

    te_results = parse_te_file(te_file)
    nvar = 2
    te_matrix = np.zeros((nvar, nvar))
    sig_te = np.zeros((nvar, nvar))

    te_matrix[0, 1] = te_results['X1->X2']['TE']
    te_matrix[1, 0] = te_results['X2->X1']['TE']
    sig_te[0, 1] = 1 if te_results['X1->X2']['p'] <= 0.05 else 0
    sig_te[1, 0] = 1 if te_results['X2->X1']['p'] <= 0.05 else 0

    fig_te, ax_te = plt.subplots(figsize=(6, 6))
    te_masked = np.where(sig_te == 1, te_matrix, np.nan)
    # Construct the annotation with both TE and p-value
    te_annotations = np.empty(te_matrix.shape, dtype=object)

    for i in range(nvar):
        for j in range(nvar):
            if sig_te[i, j] == 1:
                te_val = round(te_matrix[i, j], 4)
                p_val = round(te_results[f'X{i+1}->X{j+1}']['p'], 3)
                te_annotations[i, j] = f"{te_val:.4f}\np={p_val:.2e}"
            else:
                te_annotations[i, j] = ""

    cmap_te = plt.cm.PuBuGn
    sns.heatmap(te_masked, annot=te_annotations, fmt='', cmap=cmap_te, ax=ax_te,
                xticklabels=label_names, yticklabels=label_names,
                square=True, cbar_kws={'orientation': 'horizontal', 'label': 'TE'}, linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 14},
                vmin=0, vmax=max(te_matrix.max(), 1e-3))
    

    ax_te.set_title(f'Transfer Entropy (TE), k={k}, bias=1', fontsize=16)
    ax_te.xaxis.set_ticks_position('top')
    ax_te.set_xticklabels(ax_te.get_xmajorticklabels(), fontsize=14)
    ax_te.set_yticklabels(ax_te.get_ymajorticklabels(), fontsize=14)

    for j in range(nvar):
        for i in range(nvar):
            if sig_te[j, i] == 1:
                ax_te.add_patch(Rectangle((i, j), 1, 1, fill=False, edgecolor='darkgreen', linewidth=2))

    if save_fig:
        plt.savefig(f'transfer_entropy_matrix_k{k}_bias1.png', dpi=300, bbox_inches='tight')
    
    #plt.show()


# === ESEMPIO USO ===
liang_file = '/home/chiaraz/thesis/lin_oscillator/2D_system_data/liang_2D_bias1.txt'
te_file = '/home/chiaraz/thesis/lin_oscillator/2D_system_data/te_2D_embedding8_bias1.txt'
label_names = ['$x_1$', '$x_2$']

#plot_liang_matrices(liang_file, label_names, True)
plot_liang_matrices(liang_file, label_names, True)
plot_te_matrix(te_file, label_names, True)
plot_te_matrix('/home/chiaraz/thesis/lin_oscillator/2D_system_data/te_2D_embedding12_bias1.txt', label_names, True)
plot_te_matrix('/home/chiaraz/thesis/lin_oscillator/2D_system_data/te_2D_embedding20_bias1.txt', label_names, True)
plot_te_matrix('/home/chiaraz/thesis/lin_oscillator/2D_system_data/te_2D_embedding1_bias1.txt', label_names, True)
plot_te_matrix('/home/chiaraz/thesis/lin_oscillator/2D_system_data/te_2D_embedding4_bias1.txt', label_names, True)
"""
plot_te_matrix(te_4, label_names, True)
plot_te_matrix(te_8, label_names, True)
plot_te_matrix(te_12, label_names, True)
plot_te_matrix(te_14, label_names, True)
plot_te_matrix(te_20, label_names, True)
"""