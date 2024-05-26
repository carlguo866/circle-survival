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
exp_type = "mlp"
assert (exp_type in ['dataset', 'mlp',])
exp_type
#%%
path = ""
init_embed = np.load(path + f'data/gradients_{exp_type}/sample_seed_0/init_embed.npy')

paths = []
first_embeds = [] 
final_embed = []
# glob all the path in data directory
for path in tqdm(sorted(glob.glob(path + f'data/gradients_{exp_type}/*'))):
    if 'init_embed' in path:
        continue
    first_embed_path = os.path.join(path, 'first_embeds.npy')
    final_embed_path = os.path.join(path, 'final_embed.npy')
    if os.path.isfile(first_embed_path) and os.path.isfile(final_embed_path):   
        paths.append(path)      
        first_embeds.append(np.load(first_embed_path))
        final_embed.append(np.load(final_embed_path))
    

first_embeds = np.array(first_embeds)
first_embeds = first_embeds[:, :10]
final_embed = np.array(final_embed)
first_embeds.shape, final_embed.shape
#%%
def get_final_circle_freqs(embedding):
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    sorted_freq = np.argsort(signal)[::-1]
    threshold = np.mean(signal) * 2 
    num_circles = (signal > threshold).sum() // 2
    cur_freqs = [min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) for i in range(num_circles)]
    return cur_freqs

def get_signal(embedding):
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    # sorted_freq = np.argsort(signal)[::-1]
    # print(sorted_freq)
    return signal[1:30]
#%%
gradients = np.zeros((first_embeds.shape[1], first_embeds.shape[0], 29))
init_signal = get_signal(init_embed)
for i, embeds in enumerate(first_embeds):
    for j in range(first_embeds.shape[1] - 1):  
        gradients[j + 1, i] = get_signal(embeds[j + 1]) - get_signal(embeds[j])
    
final_signals = [] 
for embed in final_embed:
    final_signals.append(get_signal(embed))
    
final_signals = np.array(final_signals)

gradients.shape, final_signals.shape
#%%
survived = np.count_nonzero(final_signals > 10, axis=0) / final_signals.shape[0]
init_signal_repeated = np.repeat(np.expand_dims(init_signal, axis=0), gradients.shape[1], axis=0)
#%%
step = 4
gradients_normed = gradients[step] / init_signal_repeated
mask = final_signals > 10
gradient_df = pd.DataFrame(np.stack([gradients_normed.flatten(), mask.flatten()], axis=1), columns=["gradient", "alive"])
gradient_df.head()
#%%
# compute the stats significance that gradient_df["alive"] == 0 and gradient_df["alive"] == 1 are from different distributions
from scipy.stats import ttest_ind
ttest_ind(gradient_df[gradient_df["alive"] == 0]["gradient"], gradient_df[gradient_df["alive"] == 1]["gradient"])
#%%
fig, ax = plt.subplots(dpi=1000)
sns.set_style("whitegrid")
sns.histplot(data=gradient_df, x="gradient", hue="alive", bins=50, multiple="stack")
format_subplot(ax)
ax.set_xlabel("Gradient", fontsize=13)
ax.set_ylabel("Count", fontsize=13)
# plt.title(f"Histogram of Gradient at Step 4", fontsize=15)
ax.legend(title="", labels=["Survived", "Dead"], fontsize=11)
plt.savefig("figs/gradient_histogram.png")
plt.show()
#%%
step = 4
gradients_normed = gradients[step] / init_signal_repeated
mean_gradients = np.mean(gradients_normed, axis=0)
state = np.array([
    'alive' if final_signals[i, j] > 10 else 'dead'
    for i in range(final_signals.shape[0])
    for j in range(final_signals.shape[1]) 
])

gradient_signal_df = pd.DataFrame(np.stack([gradients_normed.flatten(), init_signal_repeated.flatten(), state], axis=1), columns = ["gradient", "signal", "state"])
gradient_signal_df.head()
#%%
gradient_signal_df["gradient"] = gradient_signal_df["gradient"].apply(lambda x: round(float(x), 5))
gradient_signal_df["signal"] = gradient_signal_df["signal"].apply(lambda x: round(float(x), 5))
gradient_signal_df.head()
#%%
# compute the stats significance that gradient_signal_df["state"] == "dead" and gradient_signal_df["state"] == "alive" are linearly separable on the plane
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = gradient_signal_df[["gradient", "signal"]].values
y = gradient_signal_df["state"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

#%%
fig, ax = plt.subplots(dpi=400)
sns.set_style("whitegrid")
# get c as lighter blue
c = sns.color_palette("Blues", as_cmap=True)(0.7) 
sns.scatterplot(data=gradient_signal_df[gradient_signal_df["state"] == "dead"], x="gradient", y="signal", c=c, s=6, alpha=0.7, ax=ax)
sns.scatterplot(data=gradient_signal_df[gradient_signal_df["state"] == "alive"], x="gradient", y="signal", c="orange", s=6, alpha=0.7, ax=ax)
format_subplot(ax)
plt.plot([-0.002, 0.001], [92.2, 82], color='r', linestyle='--')

ax.set_xlabel("Gradient", fontsize=13)
ax.set_ylabel("Initial Signal", fontsize=13)
#ax.set_title("Seperation of Dead/Survived Using Gradient and Signal", fontsize=15)
plt.legend(["Dead Frequency", "Survived Frequency"], fontsize=11)
plt.savefig("figs/gradient_signal.png")
plt.show()
#%%
fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), dpi=500)

for step in range(1, 5):
    gradients_normed = gradients[step] / init_signal_repeated
    mean_gradients = np.mean(gradients_normed, axis=0)

    pos = ((step - 1) // 2, (step - 1) % 2)

    mask = final_signals > 10
    gradients_masked = gradients_normed[mask]
    init_signal_masked = init_signal_repeated[mask]

    bins = np.linspace(min(gradients_masked.min(), gradients_normed[~mask].min()), 
                    max(gradients_masked.max(), gradients_normed[~mask].max()), 
                    21)  # 100 bins mean 101 edges

    counts_survived, _ = np.histogram(gradients_masked, bins=bins)
    counts_died, _ = np.histogram(gradients_normed[~mask], bins=bins)

    ratios = np.where(counts_died > 0, counts_survived /(counts_survived + counts_died), np.nan)
    ratios_df = pd.DataFrame(np.stack([bins[:-1], ratios], axis=1), columns=["bins", "ratios"])

    axes[pos] = sns.lineplot(x='bins', y='ratios', data=ratios_df, ax=axes[pos], legend=False, linewidth=2)
    axes[pos].set_xlabel("Gradient", fontsize=16)
    if step <= 2:
        axes[pos].set(xlabel=None)
    axes[pos].set_ylabel("Survival Rate", fontsize=16)
    axes[pos].set_title(f"Step {step}", fontsize=17)
    format_subplot(axes[pos])
    xticks = axes[pos].get_xticklabels()
    axes[pos].set_xticklabels(xticks, rotation = 45)

# add more space between top two plots and bottom two plots
plt.subplots_adjust(hspace=0.5)

# fig.suptitle("Survival Rate Against Gradient", fontsize=22)
plt.savefig("figs/gradient_survival_rate.png")
plt.show()