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
import math
torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
def get_signal(embedding):
    return np.linalg.norm(np.fft.fft(embedding, axis=0), axis=1)
def find_delta(frequency, mod): 
    deltas = []
    for i in range(mod): 
        if i * frequency % mod in [1, mod - 1]:
            deltas.append(i)
    return deltas

def ang(x):
    return math.cos(x) + math.sin(x) * 1j
#%%
data_path = "data/small_p/"
small_mods = [13, 19, 23, 29, 31, 37, 43, 47, 53, 59, 61, 67, 73, 79, 83, 89]

trials = 100

entries = []

for mod in small_mods:
    final_embed = np.load(data_path + f"mod_{mod}/final_embed.npy")
    for t in range(trials):
        entries.append([mod, len(get_final_circle_freqs(final_embed[t]))])
#%%
data_path = "data/big_p/"
large_mods = [97, 101, 103, 107, 109, 113, 127]

for mod in large_mods:
    for t in range(trials):
        path = data_path + f"dim_128_init_seed_0_vary_seed_{t}_p_{mod}_id_*"
        id_path = sorted(glob.glob(path))[0]
        init_embed = np.load(id_path + "/init_embed.npy")
        final_embed = np.load(id_path + "/final_embed.npy")
        entries.append([mod, len(get_final_circle_freqs(final_embed))])
    
prime_mod_df = pd.DataFrame(entries, columns=["mod", "number of circles"])
prime_mod_df.head()
# %%
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
sns.lineplot(data=prime_mod_df, x="mod", y="number of circles")
format_subplot(ax)

ax.set_xlabel(r"Modulus $p$")
ax.set_ylabel("Number of Circles")
ax.set_title(r"Number of Circles in Relation to Modulus $p$")
plt.savefig("figs/num_circles_prime_mod.png")
plt.show()
#%%
data_path = "data/diff_p/"

trials = 20
for mod in tqdm(range(16, 102)):
    for t in range(trials):
        path = data_path + f"dim_128_init_seed_0_vary_seed_{t}_p_{mod}_id_*"
        id_path = sorted(glob.glob(path))[0]
        init_embed = np.load(id_path + "/init_embed.npy")
        final_embed = np.load(id_path + "/final_embed.npy")
        entries.append([mod, len(get_final_circle_freqs(final_embed))])

mod_df = pd.DataFrame(entries, columns=["mod", "number of circles"])
mod_df.head()
# %%
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
sns.lineplot(data=mod_df[mod_df["mod"] <= 97], x="mod", y="number of circles")
format_subplot(ax)

ax.set_xlabel(r"Modulus $p$")
ax.set_ylabel("Number of Circles")
ax.set_title(r"Number of Circles in Relation to Modulus $p$")
plt.savefig("figs/num_circles_mod.png")
plt.show()
#%%
# different weight decay
data_path = "data/large_wd"

large_wds = np.linspace(0.1, 1.5, 16)

trials = 50

entries = []
for wd in tqdm(large_wds):
    weight_decay = round(wd, 1)
    init_embed = np.load(data_path + f"/wd_{weight_decay}/init_embed.npy")
    final_embed = np.load(data_path + f"/wd_{weight_decay}/final_embed.npy")
    for t in range(trials):
        entries.append([weight_decay, len(get_final_circle_freqs(final_embed[t]))])
#%%
data_path = "data/weight_decay/"

trials = 50

small_wds = 10 ** np.linspace(-4, -2, 16)
for wd in tqdm(small_wds):
    if wd < 0.002:
        continue
    wd_str = str(round(wd, 5))
    final_embed = np.load(data_path + f"final_embed_{wd_str}.npy")
    for t in range(trials):
        entries.append([float(wd_str), len(get_final_circle_freqs(final_embed[t]))])
    
wd_df = pd.DataFrame(entries, columns=["weight decay", "number of circles"])
wd_df.head()
#%%
sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=600)

sns.lineplot(data=wd_df, x="weight decay", y="number of circles", ax=axes[0])
sns.lineplot(data=wd_df[wd_df["weight decay"] < 0.01], x="weight decay", y="number of circles", ax=axes[1])
sns.lineplot(data=wd_df[wd_df["weight decay"] >= 0.01], x="weight decay", y="number of circles", ax=axes[2])
format_subplot(axes[0])
format_subplot(axes[1])
format_subplot(axes[2])

axes[0].set_xlabel("Weight Decay")
axes[1].set_xlabel("Weight Decay")
axes[2].set_xlabel("Weight Decay")
axes[0].set_ylabel("Number of Circles")
axes[0].legend(title="weight decay in range [0.003, 1.5]")
axes[1].legend(title="weight decay in range [0.003, 0.1]")
axes[2].legend(title="weight decay in range [0.1, 1.5]")
fig.suptitle("Number of Circles in Relation to Weight Decay", fontsize=16)
plt.savefig("figs/num_circles_wd.png")
plt.show()

#%% 
def calc_circularity(embedding, spectrum, frequency, mod): 
    real = spectrum[frequency].real
    imag = spectrum[frequency].imag
    real /= np.linalg.norm(real)
    imag /= np.linalg.norm(imag)
    real_proj = embedding @ real 
    real_proj = real_proj / np.sqrt(np.sum(real_proj * real_proj)) / math.sqrt(mod)
    deltas = find_delta(frequency, mod)
    real_res = 0
    for delta in deltas: 
        reciprocals =[real_proj[t * delta % mod] for t in range(mod)]
        sa=sum(reciprocals[t] * ang(2 * math.pi * t / mod) for t in range(mod))
        real_res += abs(sa) ** 2 * 2
    real_res /= len(deltas)
    imag_proj = embedding @ imag
    imag_proj = imag_proj / np.sqrt(np.sum(imag_proj * imag_proj)) / math.sqrt(mod)
    imag_res = 0
    for delta in deltas:
        reciprocals = [imag_proj[t * delta % mod] for t in range(mod)]
        sa=sum(reciprocals[t] * ang(2 * math.pi * t / mod) for t in range(mod))
        imag_res += abs(sa) ** 2 * 2
    imag_res /= len(deltas)
    results = np.array([real_res, imag_res])
    
    res = np.linalg.norm(results) / (2 ** 0.5)
    return res

# %%
data_path = "data/diff_dim_new/"

trials = 50
entries = []

mod = 59

for dim in tqdm(range(16, 129)):
    for seed in range(50):
        path = data_path + f"dim_{dim}_init_seed_0_vary_seed_{seed}_p_59_id_*"
        id_path = sorted(glob.glob(path))[0]
        init_embed = np.load(id_path + "/init_embed.npy")
        final_embed = np.load(id_path + "/final_embed.npy")
        spectrum = np.fft.fft(init_embed, axis=0)
        circularies = [calc_circularity(init_embed, spectrum, f, mod) for f in range(1, (mod + 1) // 2)]
        entries.append([dim, np.mean(circularies), np.max(circularies), len(get_final_circle_freqs(final_embed))])

dim_df = pd.DataFrame(entries, columns=['dim','mean circularity', 'max circularity', 'number of circles'])
dim_df.head()
# %%
data_path = "data/diff_dim_new/"

trials = 50
entries = []

p = 59

for dim in tqdm(range(128, 129)):
    for seed in range(50):
        path = data_path + f"dim_{dim}_init_seed_0_vary_seed_{seed}_p_59_id_*"
        id_path = sorted(glob.glob(path))[0]
        init_embed = np.load(id_path + "/init_embed.npy")
        final_embed = np.load(id_path + "/final_embed.npy")
        spectrum = np.fft.fft(init_embed, axis=0)
        circle_freqs = get_final_circle_freqs(final_embed)
        for f in range(1, 30):
            circularity = calc_circularity(init_embed, spectrum, f, p)
            entries.append([circularity, "survived" if f in circle_freqs else "dead"])

circularity_df = pd.DataFrame(entries, columns=['circularity', 'state'])
circularity_df.head()
#%%
from scipy.stats import ttest_ind
ttest_ind(circularity_df[circularity_df["state"] == "dead"]["circularity"], circularity_df[circularity_df["state"] == "survived"]["circularity"])
# %%
sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=600)
dim_df_melted = pd.melt(dim_df, id_vars=['dim'], value_vars=['mean circularity', 'max circularity'], var_name='metric', value_name='circularity')
sns.lineplot(data=dim_df_melted, x='dim', y='circularity', hue='metric', ax=axes[0])
format_subplot(axes[0])

axes[0].set_xlabel("Dimension")
axes[0].set_ylabel("Circularity")
axes[0].set_title("Initial Circularity in Relation to Dimension", fontsize=13)

sns.histplot(data=circularity_df, x="circularity", hue="state", bins=50, multiple="stack", ax=axes[1])
format_subplot(axes[1])
axes[1].set_xlabel("Circularity")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Histogram of Initial Circularity", fontsize=13)
axes[1].legend(title="", labels=["Survived", "Dead"])
plt.savefig("./../figs/circularity_dim.png")
plt.show()
# %%
sns.set_theme(style='whitegrid')

fig, ax = plt.subplots(figsize=(8, 5), dpi=600)
sns.lineplot(data=dim_df, x='dim', y='number of circles')
format_subplot(ax)

ax.set_xlabel("Dimension")
ax.set_ylabel("Number of Circles")
ax.set_title("Number of Circles in Relation to Dimension", fontsize=14)
plt.savefig("figs/num_circles_dim.png")
plt.show()