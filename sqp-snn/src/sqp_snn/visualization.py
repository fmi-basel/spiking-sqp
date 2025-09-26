import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import torch


def preds_targets_plot(model, dataset, label):
    preds = model.predict(dataset)
    preds = preds.squeeze().cpu().numpy()

    labels = np.array([])
    for data in dataset:
        labels = np.append(labels, data[1])

    min_val = np.min([preds, labels])
    max_val = np.max([preds, labels])

    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal prediction')
    scatter = plt.scatter(preds, labels, c=abs(preds-labels), cmap='inferno', alpha=0.6)

    cbar = plt.colorbar(scatter)
    cbar.set_label('Absolute error')

    plt.xlabel('Predictions')
    plt.ylabel('Targets')
    plt.title(f'Predictions vs. targets ({label})')
    plt.legend()

    plt.grid(True, linestyle=':', alpha=0.7)

    pearson_corr, _ = stats.pearsonr(preds, labels)
    plt.text(0.05, 0.9, f'PCC = {pearson_corr:.3f}', transform=plt.gca().transAxes, 
         verticalalignment='top', fontsize=10)
    
    plt.tight_layout()
    return
