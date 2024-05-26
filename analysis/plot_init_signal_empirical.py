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
#%%
data_path = "data/more_init_signals/"

trials = 50

entries = []
for init_seed in tqdm(range(10)):
    survival_count = np.zeros(30)
    for seed in range(trials):
        path = data_path + f"dim_128_init_seed_{init_seed}_vary_seed_{seed}_p_59_id_*"
        id_path = sorted(glob.glob(path))[0]
        init_embed = np.load(id_path + "/init_embed.npy")
        final_embed = np.load(id_path + "/final_embed.npy")
        signal = np.linalg.norm(np.fft.fft(init_embed, axis=0), axis=1)
        circles = get_final_circle_freqs(final_embed)
        for freq in circles:
            survival_count[freq] += 1
    for freq in np.arange(1, 30):
        entries.append([signal[freq], survival_count[freq] / trials])

signal_df = pd.DataFrame(entries, columns=["init signal", "survival rate"])
#%%
signal_df.head()
#%%
# test significance that signal_df ["init signal"] and ["survival rate"] linearly correlated
from scipy import stats
stats.pearsonr(signal_df["init signal"], signal_df["survival rate"])
# %%
fig, ax = plt.subplots(dpi=600)

sns.scatterplot(data=signal_df, x="init signal", y="survival rate", alpha=0.7, ax=ax)
format_subplot(ax)
c = sns.color_palette("Oranges", as_cmap=True)(1)
ax.plot([86, 95], [0, 0.75], color='orange', linestyle='--', linewidth=3)

ax.set_xlabel("Initial Signal")
ax.set_ylabel("Survival Rate")
plt.title("Survival Rate Against Initial Signal", fontsize=14)
plt.savefig("figs/signal_survival_rate.png")
plt.show()