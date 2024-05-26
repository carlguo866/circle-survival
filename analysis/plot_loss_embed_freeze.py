#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import os
import sys
from tqdm import tqdm
import glob
import os
import pandas as pd
import seaborn as sns
from utils import format_subplot, get_final_circle_freqs
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(sys.argv)
# %%
data_path = "data/dim_freeze/"

trials = 5
mod = 59

def get_dataframe(step):
    entries = []

    for dim in tqdm(range(16, 129)):
        for seed in range(5):
            path = data_path + f"dim_{dim}_init_seed_0_vary_seed_{seed}_p_59_id_*"
            id_path = sorted(glob.glob(path))[0]
            test_loss = np.load(id_path + "/test_loss.npy")
            test_acc = np.load(id_path + "/test_acc.npy")
            entries.append([dim, test_loss[step], test_acc[step]])

    dim_freeze_df = pd.DataFrame(entries, columns=['dim', 'test loss', 'test acc'])
    return dim_freeze_df
# %%
sns.set_theme(style='whitegrid')
fig, axes = plt.subplots(2, 4, figsize=(13, 6), dpi=200)
steps = [500, 1000, 5000, 10000]

for i, step in enumerate(steps):
    dim_freeze_df = get_dataframe(step)
    converge_df = dim_freeze_df[dim_freeze_df["test loss"] < 1]

    sns.scatterplot(data=dim_freeze_df, x='dim', y='test loss', hue='test acc', ax=axes[0, i], legend=False)
    format_subplot(axes[0, i])
    sns.scatterplot(data=converge_df, x='dim', y='test loss', hue='test acc', ax=axes[1, i], palette='viridis', legend=False)
    format_subplot(axes[1, i])

    axes[0, i].axvline(x=mod, color='red', linestyle='--')
    axes[1, i].axvline(x=mod, color='red', linestyle='--')

    axes[0, i].set_ylabel("Test Loss After Training")
    axes[1, i].set_ylabel("Test Loss After Training")
    axes[1, i].set_xlabel("Dimension")
    axes[0, i].set_title(f"{step} steps", fontsize=13)
    axes[0, i].set_xlim(16, 128)
    axes[1, i].set_xlim(16, 128)
    axes[0, i].set_ylim(0, 15)
    axes[1, i].set_ylim(0, 1)

    if i > 0:
        axes[0, i].set(ylabel=None)
        axes[1, i].set(ylabel=None)
    
    axes[0, i].set(xlabel=None)

axes[0, 0].text(mod+2, 10, 'd = p', color='red', fontsize=12)

fig.suptitle(f"Test Loss After Various Steps of Training", fontsize=15)
plt.savefig(f"figs/test_loss_freeze.png")
plt.show()

# %%
