#%% 
from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
from sklearn.linear_model import LinearRegression, Lasso
import seaborn as sns
import itertools
import pandas as pd
import glob
from tqdm import tqdm
from utils import get_final_circle_freqs, rollout_ode_traj, format_subplot
#%% 

## Feel free to change the path to the embeddings
path = "data/fix_mlp/seed_1/embeddings.npy"
embeddings = np.load(path)
signals = np.linalg.norm(np.fft.fft(embeddings, axis=1), axis=-1)
print(signals.shape)
#%% 
circle_freq = get_final_circle_freqs(embeddings)
circle_freq
#%% 
# only first order terms and the first 1000 timesteps
n = 1000
circle_signal = signals[:n, 1:30]
y = signals[1:n + 1, 1:30] - signals[:n, 1:30]
reg = Lasso(alpha=0.01).fit(circle_signal, y)
print(reg.score(circle_signal, y))
trajs = rollout_ode_traj(reg, circle_signal)


#%% 
coef_matrix = reg.coef_

# plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(coef_matrix, fmt=".2f", cmap="coolwarm")
plt.title("Coefficient Matrix Heatmap", fontsize=16)
plt.xlabel("Coefficients 1")
plt.ylabel("Freq")
current_yticks = plt.gca().get_yticks()
new_yticks = current_yticks + 1
plt.gca().set_yticklabels(new_yticks.astype(int))
current_xticks = plt.gca().get_xticks()
new_xticks = current_xticks + 1
plt.gca().set_xticklabels(new_xticks.astype(int))
plt.show()

#%% 
import scipy

# 1e-8 is added to the diagonal to avoid singular matrix
A = reg.coef_ + 1e-8 * np.eye(len(reg.coef_))
b = reg.intercept_
t0 = 0  # Initial time
tf = 1000 # Final time
x0 = circle_signal[0]

def analytical_solution(A, b, t, x0):
    expA = scipy.linalg.expm(A * (t - t0))
    x = np.dot(expA, x0) 
    y = np.dot(expA - np.eye(len(A)), np.dot(np.linalg.inv(A), b))
    return x +y

t = np.linspace(t0, tf, 1000)

# Compute the analytical solution
analytical = np.array([analytical_solution(A, b, ti, x0) for ti in t])

#%% 
df = pd.DataFrame(trajs)
df2 = pd.DataFrame(trajs2)
df3 = pd.DataFrame(circle_signal)
df4 = pd.DataFrame(analytical)
# convert them into one dataframe with different identifier
df['identifier'] = r'Lasso, $\alpha = 0.01$'
df2['identifier'] = 'Linear Regression'
df3['identifier'] = 'Original'
df4['identifier'] = 'Lasso Analytical Solution'
df = pd.concat([df, df2, df3, df4], axis=0)
df.head()

#%%
for i in range(29):
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    sns.lineplot(data=df, x=df.index, y=i, hue='identifier')
    format_subplot(plt.gca())
    plt.title(f'Frequency {i+1}')
    plt.xlabel("Timestep")
    plt.ylabel("Signal")
    plt.legend()
    plt.savefig(f'figs/ode_freq_{i}.png')
    plt.show()