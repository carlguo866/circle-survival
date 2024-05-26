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
def get_signal(embedding):
    return np.linalg.norm(np.fft.fft(embedding, axis=0), axis=1)
# %%
data_path = "data/more_perturbs"

entries = []

scales = np.linspace(0.85, 1.15, 50)

for init_seed in range(8):
    id_path = data_path + f"dim_128_init_seed_{init_seed}_vary_seed_0_index_0_id_*"
    path = sorted(glob.glob(id_path))[0]
    init_embed = np.load(path + "/init_embed.npy")
    init_signals = get_signal(init_embed)
    init_mean = np.mean(init_signals)
    init_signal = init_signals[25]
    for index in tqdm(range(50)):
        survive_count = 0
        for vary_seed in range(50):
            id_path = data_path + f"dim_128_init_seed_{init_seed}_vary_seed_{vary_seed}_index_{index}_id_*"
            path = sorted(glob.glob(id_path))[0]
            scale = scales[index]
            final_embed = np.load(path + "/final_embed.npy")
            circ_freqs = get_final_circle_freqs(final_embed)
            survive_count += 25 in circ_freqs
        
        entries.append([scales[index] * init_signal - init_mean, survive_count / 50, init_seed + 1])

perturb_df = pd.DataFrame(entries, columns=['diff', 'survival rate', 'trial'])
perturb_df.head()
# %%
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(dpi=600)

c = sns.color_palette("Oranges", as_cmap=True)(0.7)
sns.lineplot(data=perturb_df[perturb_df["trial"] == 3], x='diff', y='survival rate', hue="trial", ax=ax, legend=False, palette=[c])
sns.lineplot(data=perturb_df[perturb_df["trial"] != 3], x='diff', y='survival rate', hue="trial", ax=ax, legend=False, alpha=0.4)
format_subplot(ax)
ax.set_xlabel('Difference to Initial Signal Mean')
ax.set_ylabel('Survival Rate')

ax.set_ylim(-0.05, 1.05)
ax.set_xlim(-20, 5)
plt.title('Survival Rate Against Difference to Initial Signal Mean', fontsize=14)

plt.savefig('figs/perturb.png')
plt.show()