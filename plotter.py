import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_acc(log_file, img_file):
    log = pd.read_csv(log_file, sep='\t')
    acc_cols = [col for col in log.columns if col.startswith('acc_')]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)

    for acc_col in acc_cols:
        log_subset = log[~pd.isnull(log[acc_col])]

        train_x = log_subset[~log_subset.is_validation].epoch.values
        train_y = log_subset[~log_subset.is_validation][acc_col].values
        ax.plot(train_x, train_y, c='g', label=acc_col + '_train')

        val_x = log_subset[log_subset.is_validation].epoch.values
        val_y = log_subset[log_subset.is_validation][acc_col].values
        ax.plot(val_x, val_y, c='r', label=acc_col + '_val')
    
    ax.set_xticks(np.arange(1, log.epoch.max() + 1))
    ax.set_yticks(np.linspace(0, 1, 11))
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_xlim(1, log.epoch.max() + 0.33)
    ax.set_ylim(0, 1.03)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid()
    plt.tight_layout()
    plt.savefig(img_file + '_acc.png')