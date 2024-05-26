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
import seaborn as sns
import pandas as pd
from utils import format_subplot, get_final_circle_freqs
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print(sys.argv)
# %%
multipliers = np.exp(np.linspace(-4, 0, 16))

entries = []

for index in range(16):
    multiplier = multipliers[index]
    survival_count_1 = 0
    survival_count_2 = 0
    for seed in tqdm(range(100)):
        np.random.seed(seed)
        torch.manual_seed(seed)
        circ1, circ2 = np.random.choice(np.arange(1, 30), 2, replace=False)
        path = f"data/competition_data/index_{index}_seed_{seed}/"
        final_embed = np.load(path + "final_embed.npy")
        circle_freqs = get_final_circle_freqs(final_embed)
        survival_count_1 += circ1 in circle_freqs
        survival_count_2 += circ2 in circle_freqs

    entries.append([multiplier, survival_count_1 / 100, survival_count_2 / 100])

compete_df = pd.DataFrame(entries, columns=["multiplier", "largest", "second largest"])
compete_df.head()
# %%
sns.set_theme(style="whitegrid")

compete_df_melt = compete_df.melt(id_vars='multiplier', value_vars=['largest', 'second largest'], var_name='initial signal', value_name='survival rate')
fig, ax = plt.subplots(dpi=600)
sns.lineplot(data=compete_df_melt, x='multiplier', y='survival rate', hue='initial signal', ax=ax, linewidth=2)
ax.set_xlabel(r'Ratio $r$ of Largest to Second Largest Initial Signal')
ax.set_ylabel('Survival Rate')
format_subplot(ax)
plt.title(r'Survival Rate Against Ratio $r$')
plt.savefig("figs/init_signal_construction.png")
plt.show()
