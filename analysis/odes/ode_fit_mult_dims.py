#%% 
from IPython import get_ipython
ipython = get_ipython()
if ipython: 
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
import seaborn as sns·
import itertools
import pandas as pd
import glob
from tqdm import tqdm
from utils import *
import seaborn as sns

#%% 
path = "data/diff_dim/"

n = 1000
lasso_scores = []
linear_scores = []
for dim in list(range(4, 96)): 
    files = glob.glob(path + f"dim_{dim}_seed_*/first_embeds.npy")[:10]
    lasso_score = []
    linear_score = []
    for file in tqdm(files):
        embeddings = np.load(file)

        signals = np.linalg.norm(np.fft.fft(embeddings, axis=1), axis=-1)
        circle_signal = signals[:n, 1:30]
        y = signals[1:n + 1, 1:30] - signals[:n, 1:30]
        reg = Lasso(alpha=0.01).fit(circle_signal, y)
        lasso_score.append(reg.score(circle_signal, y))
        reg = Lasso(alpha=0.00).fit(circle_signal, y)
        linear_score.append(reg.score(circle_signal, y))
    lasso_scores.append(np.array(lasso_score))
    linear_scores.append(np.array(linear_score))
#%% 
df = pd.DataFrame(lasso_scores)
df_linear = pd.DataFrame(linear_scores)
stacked_df = df.stack().reset_index()
stacked_df.columns = ['Dim', 'Seed', 'R2']
stacked_df['Dim'] = stacked_df['Dim']+4
stacked_df['identifier'] = r"Lasso, $\alpha=0.01$"

stacked_df_linear = df_linear.stack().reset_index()
stacked_df_linear.columns = ['Dim', 'Seed', 'R2']
stacked_df_linear['Dim'] = stacked_df2['Dim']+4
stacked_df_linear['identifier'] = r"Linear Regression"

total_df = pd.concat([stacked_df, stacked_df_linear], axis=0)
#%% 
fig, ax = plt.subplots()
sns.set_theme(style="whitegrid")
sns.lineplot(x='Dim', y='R2', data=total_df, hue='identifier', estimator='mean', errorbar='sd', ax=ax)
format_subplot(ax)
plt.xlabel('Dim size')
plt.ylabel(r'$R^2$')
plt.title(r'$R^2$ over trials of differently sized embeddings')
plt.legend()
plt.savefig("figs/ode_R^2.png")
plt.show()