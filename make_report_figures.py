"""Draw the two report figures from results.json (no model is fitted here).

Usage: python make_report_figures.py [outputs_folder]
The notebook saves full-size figures. These versions use a compact layout so the
labels stay readable at the size used in the Word report.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else 'a2_outputs')
R = json.loads((out / 'results.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                     'font.family': 'DejaVu Sans'})
BLUE, ORANGE, AQUA, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#1baf7a', '#0b0b0b', '#52514e', '#e4e3df'

# Figure 1: convergence
C = R['convergence']
train = np.array(C['sgd_train_loss'])
stop = np.array(C['sgd_stop_loss'])
epochs = np.arange(1, train.shape[1] + 1)
lb_x = [p['iterations'] for p in C['lbfgs_path']]
lb_y = [p['train_loss'] for p in C['lbfgs_path']]
stop_at = int(np.median(C['stop_epochs']))
fig, ax = plt.subplots(figsize=(6.0, 2.25))
ax.fill_between(epochs, train.min(axis=0), train.max(axis=0), color=ORANGE, alpha=0.2, linewidth=0)
ax.plot(epochs, train.mean(axis=0), color=ORANGE, linewidth=2, marker='o', markersize=3,
        label='SGD training loss (mean of 5 seeds, band = range)')
ax.plot(epochs, stop.mean(axis=0), color=AQUA, linewidth=2, linestyle='--',
        label='SGD stopping-set loss (mean of 5 seeds)')
ax.plot(lb_x, lb_y, color=BLUE, linewidth=2, marker='s', markersize=3,
        label='Baseline L-BFGS training loss')
ax.axvline(stop_at, color=MUTED, linewidth=1, linestyle=':')
ymax = max(max(lb_y[1:]), float(train.max()), float(stop.max()))
ymin = min(min(lb_y), float(train.min()), float(stop.min()))
pad = 0.1 * (ymax - ymin)
ax.set_ylim(ymin - 1.6 * pad, ymax + pad)
ax.text(stop_at * 0.93, ymin - 0.9 * pad, f'early stopping ends at epoch {stop_at} (median)',
        fontsize=8, color=MUTED, ha='right', va='center')
ax.set_xscale('log')
ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax.set_xticklabels(['1', '2', '5', '10', '20', '50', '100'])
ax.set_xlabel('Passes over the training rows (SGD epochs or L-BFGS iterations, log scale)', color=INK)
ax.set_ylabel('Log loss', color=INK)
ax.grid(axis='y', color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8, loc='upper right')
fig.tight_layout(pad=0.4)
fig.savefig(out / 'report_fig1_convergence.png', dpi=300)

# Figure 2: cross-validated performance
CV = R['cross_validation']['summary']
order = ['baseline', 'sgd', 'sgd_early_stop', 'sgd_kbest']
names = ['Baseline (L-BFGS)', 'SGD', 'SGD +\nearly stopping', 'SGD +\nfeature reduction']
fig, ax = plt.subplots(figsize=(6.0, 2.05))
x = np.arange(len(order))
width = 0.36
for offset, metric, label, color in [(-width / 2, 'f1', 'F1', BLUE), (width / 2, 'recall', 'Recall', ORANGE)]:
    means = [CV[k][metric]['mean'] for k in order]
    stds = [CV[k][metric]['std'] for k in order]
    bars = ax.bar(x + offset, means, width - 0.03, yerr=stds, color=color, ecolor=MUTED,
                  capsize=3, error_kw={'linewidth': 1}, label=label)
    for rect, mean, std in zip(bars, means, stds):
        ax.text(rect.get_x() + rect.get_width() / 2, mean + std + 0.01, f'{mean:.3f}',
                ha='center', fontsize=8, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=8.5)
ax.set_ylabel('CV mean \u00b1 SD', color=INK)
ax.set_ylim(0, 0.68)
ax.set_yticks([0, 0.2, 0.4, 0.6])
ax.grid(axis='y', color=GRID, linewidth=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=8.5, ncol=2, loc='upper right')
fig.tight_layout(pad=0.4)
fig.savefig(out / 'report_fig2_performance.png', dpi=300)
print('Saved report_fig1_convergence.png and report_fig2_performance.png to', out)
