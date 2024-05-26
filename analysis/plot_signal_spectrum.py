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

# %%
trial_embeddings = [] 
data_path = "data/fix_mlp/seed_10"

embeddings = np.load(data_path + "/embeddings.npy")
embeddings.shape
# %%
spectrums = np.fft.fft(embeddings, axis=1)
signals = np.linalg.norm(spectrums, axis=-1)
signals.shape
# %%
circle_freqs = get_final_circle_freqs(embeddings)
circle_freqs
# %%
signals_df = pd.DataFrame(signals[:, 1:30], columns=[f'signal_{i}' for i in range(1, 30)])
circle_df = signals_df[[f'signal_{i}' for i in circle_freqs]]
noncircle_df = signals_df[[f'signal_{i}' for i in range(1, 30) if not i in circle_freqs]]

circle_df.head()
# %%
sns.set_theme(style="whitegrid")
fig, ax = plt.subplots(dpi=600)

sns.lineplot(data=noncircle_df, legend=None, dashes=False, palette=['grey'], alpha=0.2)
sns.lineplot(data=circle_df, legend=None, dashes=False)

grid_x=True
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
if grid_x:
    ax.grid(linestyle='--', alpha=0.4)
else:
    ax.grid(axis='y', linestyle='--', alpha=0.4)

ax.set_xscale('log')
plt.xlabel('step')
plt.ylabel('signal')
plt.title("Evolution of Frequency Signal With Time", fontsize=14)
plt.savefig("figs/signal_evolution.png")
plt.show()
#%%
fig, axes = plt.subplots(1, len(circle_freqs), figsize=(23, 4), dpi=400)

final_embedding = embeddings[-1]

# do PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2*len(circle_freqs))
pca.fit(final_embedding)
components = pca.components_

print(pca.singular_values_)

for i in range(len(circle_freqs)):
    x = components[i * 2]
    y = components[i * 2 + 1]
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    embed = np.stack([final_embedding @ x, final_embedding @ y, np.arange(59)],axis=0)

    embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
    axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
    for j in range(59):
        axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
    axes[i].set(xlabel=None)
    axes[i].set(ylabel=None)
    format_subplot(axes[i])

axes[0].text(-0.5, 0, r"k = 2, $\Delta$ = 30", fontsize=14)
axes[1].text(-0.5, 0, r"k = 8, $\Delta$ = 22", fontsize=14)
axes[2].text(-0.5, 0, r"k = 22, $\Delta$ = 8", fontsize=14)
axes[3].text(-0.5, 0, r"k = 23, $\Delta$ = 18", fontsize=14)
axes[4].text(-0.5, 0, r"k = 3, $\Delta$ = 20", fontsize=14)

fig.suptitle("Embeddings of Frequencies on PCA Planes", fontsize=20)
plt.savefig("figs/pca_embedding.png")
plt.show()

# %%
fig, axes = plt.subplots(1, len(circle_freqs), figsize=(23, 4), dpi=400)

final_embedding = embeddings[-1]

for i, freq in enumerate(circle_freqs):
    real = spectrums[-1, freq].real
    imag = spectrums[-1, freq].imag
    real /= np.linalg.norm(real)
    imag /= np.linalg.norm(imag)
    embed = np.stack([final_embedding @ real, final_embedding @ imag, np.arange(59)],axis=0)
    
    embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
    axes[i] = sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i], palette="viridis", legend=False)
    for j in range(59):
        axes[i].text(embed[0, j], embed[1, j], str(int(embed[2, j])), size=9)
    axes[i].set(xlabel=None)
    axes[i].set(ylabel=None)
    format_subplot(axes[i])

axes[0].text(-0.5, 0, r"k = 2, $\Delta$ = 30", fontsize=14)
axes[1].text(-0.5, 0, r"k = 8, $\Delta$ = 22", fontsize=14)
axes[2].text(-0.5, 0, r"k = 22, $\Delta$ = 8", fontsize=14)
axes[3].text(-0.5, 0, r"k = 23, $\Delta$ = 18", fontsize=14)
axes[4].text(-0.5, 0, r"k = 3, $\Delta$ = 20", fontsize=14)
    
fig.suptitle("Embeddings of Frequencies on FFT Planes", fontsize=20)
plt.savefig("figs/fft_embedding.png")
plt.show()
# %%
fig, axes = plt.subplots(2, 3, figsize=(9, 6), dpi=600)

steps = [0, 15000, 30000]
freqs = [1, 2]

for i, freq in enumerate(freqs):
    for j, step in enumerate(steps):
        real = spectrums[-1, freq].real
        imag = spectrums[-1, freq].imag
        real /= np.linalg.norm(real)
        imag /= np.linalg.norm(imag)
        embed = np.stack([embeddings[step] @ real, embeddings[step] @ imag, np.arange(59)],axis=0)
        
        embed_df = pd.DataFrame(embed.T, columns=['x', 'y', 'id'])
        sns.scatterplot(x='x', y='y', hue='id', data=embed_df, ax=axes[i, j], palette="viridis", legend=False)

        axes[i, j].set(xlabel=None)
        axes[i, j].set(ylabel=None)
        axes[i, j].set_xlim(-2.5, 2.5)
        axes[i, j].set_ylim(-2.5, 2.5)

        if i == 0:
            axes[i, j].set_title(f"{step} steps", fontsize=12)
        format_subplot(axes[i, j])

axes[0, 2].text(1, -2, "k = 1")
axes[1, 2].text(1, -2, "k = 2")
fig.suptitle("Evolution of Embedding on FFT Plane", fontsize=17)
plt.savefig("figs/embedding_evolution.png")
plt.show()
