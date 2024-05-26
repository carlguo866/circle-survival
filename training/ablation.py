#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import os
import sys
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import uuid

import argparse

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, w=2, depth=2, shp=None, embedding_size=None, embedding=None):
        super(MLP, self).__init__()
         
        if shp == None:
            shp = [in_dim] + [w]*(depth-1) + [out_dim]
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.depth = depth
                
        else:
            self.in_dim = shp[0]
            self.out_dim = shp[-1]
            self.depth = len(shp) - 1
        linear_list = []
        for i in range(self.depth):
            linear_list.append(nn.Linear(shp[i], shp[i+1]))
        self.linears = nn.ModuleList(linear_list)
        self.shp = shp
        if embedding is not None:
            self.embedding = torch.nn.Parameter(torch.from_numpy(embedding).to(device))
        else: 
            self.embedding = torch.nn.Parameter(torch.normal(0,1,size=embedding_size))

    def forward(self, x):
        shp = x.shape
        f = torch.nn.SiLU()
        acts = []
        acts.append(x.clone())
        for i in range(self.depth-1):
            x = f(self.linears[i](x))
            acts.append(x.clone())
        x = self.linears[-1](x)
        acts.append(x.clone())
        return x
    

def get_model(seed, dim, p, embedding=None, freeze_embedding=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    in_dim = 2*dim
    out_dim = p
    depth = 2
    shp = [in_dim] + [100] * depth + [out_dim]
    model = MLP(shp=shp, embedding_size=(p, dim),embedding=embedding)
    if freeze_embedding:
        model.embedding.requires_grad = False
    model = model.to(device)
    return model

def get_train_test(seed, fraction, p):
    train_num = int(p**2*fraction)
    test_num = p**2 - train_num
    np.random.seed(seed)
    train_id = np.random.choice(p**2,train_num,replace=False)
    test_id = np.array(list(set(np.arange(p**2)) - set(train_id)))

    return train_id, test_id

def get_data(model, id_, data_id, labels):
    inputs = torch.cat([model.embedding[data_id[id_][:,0]], model.embedding[data_id[id_][:,1]]], dim=1)
    return inputs, labels[id_]

def fuse_parameters(model):
    n = sum(param.numel() for param in model.parameters())
    params = torch.zeros(n).to(device).to(torch.double)
    i = 0
    for param in model.parameters():
        params_slice = params[i:i + param.numel()]
        params_slice.copy_(param.flatten())
        param.data = params_slice.view(param.shape)
        i += param.numel()
    return params

def fuse_mlp(model):
    n = sum(p.numel() for idx, p in enumerate(model.parameters()) if idx != 0)
    params = torch.zeros(n).to(device).to(torch.double)
    i = 0
    for idx, p in enumerate(model.parameters()):
        if idx == 0: 
            continue
        params_slice = params[i:i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params

def set_mlp_and_embedding(model, mlp, embedding, freeze_embedding=False):
    mlp = torch.tensor(mlp).to(device)
    i = 0
    for idx, p in enumerate(model.parameters()):
        if idx == 0: 
            p.data = torch.from_numpy(embedding).to(device)
        else: 
            p.data = mlp[i:i + p.numel()].view(p.shape)
            i += p.numel()

def intervention(embedding, freq, scale=1.0): 
    spectrum = np.fft.fft(embedding)
    spectrum[[freq, 59-freq]] *= scale
    return np.fft.ifft(spectrum)

def train(model, train_id, test_id, data_id, labels):
    log = 1000
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    l2s = []
    model_params = []
    data_log = 1000
    steps = 30001
    print_log = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.5,foreach=False)
    init_embed = model.embedding.cpu().detach().numpy()
    first_embeds = [init_embed]
    embeddings = [init_embed]
    gradients = []
    final_embed = None

    for step in tqdm(range(steps)):
        #print(step)
        
        if step % log == 0:
            model_params.append(fuse_parameters(model).cpu().detach().numpy())
        
        CEL = nn.CrossEntropyLoss()
        
        optimizer.zero_grad()
        
        inputs_train, labels_train = get_data(model, train_id, data_id, labels)
        inputs_train = inputs_train.to(device)
        labels_train = labels_train.to(device)
        pred  = model(inputs_train)
        loss = CEL(pred, labels_train)
        acc = torch.mean((torch.argmax(pred, dim=1) == labels_train).float())
        
        inputs_test, labels_test = get_data(model, test_id, data_id, labels)
        labels_test = labels_test.to(device)
        pred_test = model(inputs_test)
        loss_test = CEL(pred_test, labels_test)
        acc_test = torch.mean((torch.argmax(pred_test, dim=1) == labels_test).float())

        total_loss = loss
        total_loss.backward()
        optimizer.step()
        
        l2 = torch.norm(fuse_parameters(model))
        train_accs.append(acc.item())
        test_accs.append(acc_test.item())
        train_losses.append(loss.item())
        test_losses.append(loss_test.item())
        l2s.append(l2.item())


        if (step % log == 0):
            if print_log:
                print("step = %d | total loss: %.2e | train loss: %.2e | test loss %.2e | train acc: %.2e | test acc: %.2e "%(step, total_loss.cpu().detach().numpy(), loss.cpu().detach().numpy(), loss_test.cpu().detach().numpy(), acc.cpu().detach().numpy(), acc_test.cpu().detach().numpy()))
        
        if step < 1000: 
            first_embeds.append(model.embedding.cpu().detach().numpy())
        if step == steps -1:
            final_embed = model.embedding.cpu().detach().numpy()
            
        if step % data_log == 0:
            embeddings.append(model.embedding.cpu().detach().numpy())
            if model.embedding.requires_grad:
                gradients.append(model.embedding.grad.cpu().detach().numpy())
            
    returns = {
        'train_accs': train_accs,
        'test_accs': test_accs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'l2s': l2s,
        'init_embed': init_embed,
        'final_embed': final_embed, 
        'model_params': model_params,
        'first_embeds': first_embeds, 
        'embeddings': embeddings,
        'gradients': gradients
    }
    return returns

def set_parameters(model, params):
    params = torch.tensor(params).to(device)
    i = 0
    for p in model.parameters():
        p.data = params[i:i + p.numel()].view(p.shape)
        i += p.numel()

def get_signal(embedding):
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    return signal

def get_circle_freqs(embedding):
    spectrum = np.fft.fft(embedding, axis=0)
    signal = np.linalg.norm(spectrum, axis=1)
    sorted_freq = np.argsort(signal)[::-1]
    threshold = np.mean(signal) * 2
    num_circles = (signal > threshold).sum() // 2
    cur_freqs = [min(sorted_freq[i * 2], sorted_freq[i * 2 + 1]) for i in range(num_circles)]
    return cur_freqs

# %%
dim = 128
p = 59
    
x = np.arange(p)
y = np.arange(p)
XX, YY = np.meshgrid(x, y)
data_id = np.transpose([XX.reshape(-1,), YY.reshape(-1,)])
labels = (data_id[:,0] + data_id[:,1]) % p
labels = torch.tensor(labels, dtype=torch.long)

model = get_model(0, dim, p)
init_mlp = fuse_mlp(model)
init_embedding = model.embedding.cpu().detach().numpy()
train_id, test_id = get_train_test(0, 0.8, p)
# %%
model = get_model(0, dim, p)
set_mlp_and_embedding(model, init_mlp, init_embedding)
returns_O = train(model, train_id, test_id, data_id, labels)
returns_O["test_losses"][-1]
# %%
init_spectrum = np.fft.fft(init_embedding, axis=0)
init_spectrum_A = np.zeros_like(init_spectrum)
init_spectrum_A[[11, p - 11], :] = init_spectrum[[11, p - 11], :]
init_embedding_A = np.fft.ifft(init_spectrum_A, axis=0)
get_signal(init_embedding_A)
# %%
model = get_model(0, dim, p)
set_mlp_and_embedding(model, init_mlp, init_embedding_A)
returns_A = train(model, train_id, test_id, data_id, labels)
get_signal(returns_A['final_embed']), get_circle_freqs(returns_A['final_embed'])
# %%
init_spectrum = np.fft.fft(init_embedding, axis=0)
init_spectrum_B = np.zeros_like(init_spectrum)
init_spectrum_B[[11, p - 11, 8, p - 8], :] = init_spectrum[[11, p - 11, 8, p - 8], :]
init_embedding_B = np.fft.ifft(init_spectrum_B, axis=0)
get_signal(init_embedding_B)
# %%
model = get_model(0, dim, p)
set_mlp_and_embedding(model, init_mlp, init_embedding_B)
returns_B = train(model, train_id, test_id, data_id, labels)
get_signal(returns_B['final_embed']), get_circle_freqs(returns_B['final_embed'])
# %%
init_spectrum = np.fft.fft(init_embedding, axis=0)
init_spectrum_C = np.zeros_like(init_spectrum)
init_spectrum_C[[1, p - 1], :] = init_spectrum[[1, p - 1], :]
init_embedding_C = np.fft.ifft(init_spectrum_C, axis=0)
get_signal(init_embedding_C)
# %%
model = get_model(0, dim, p)
set_mlp_and_embedding(model, init_mlp, init_embedding_C)
returns_C = train(model, train_id, test_id, data_id, labels)
get_signal(returns_C['final_embed']), get_circle_freqs(returns_C['final_embed'])
# %%
init_spectrum = np.fft.fft(init_embedding, axis=0)
init_spectrum_D = np.zeros_like(init_spectrum)
init_spectrum_D[[1, p - 1, 2, p - 2], :] = init_spectrum[[1, p - 1, 2, p - 2], :]
init_embedding_D = np.fft.ifft(init_spectrum_D, axis=0)
get_signal(init_embedding_D)
# %%
model = get_model(0, dim, p)
set_mlp_and_embedding(model, init_mlp, init_embedding_D)
returns_D = train(model, train_id, test_id, data_id, labels)
get_signal(returns_D['final_embed']), get_circle_freqs(returns_D['final_embed'])
#%%
test_loss_df = pd.DataFrame(np.stack([np.arange(30001), returns_O["test_losses"], returns_A["test_losses"], returns_B["test_losses"], returns_C["test_losses"], returns_D["test_losses"]], axis=1), columns=["timestep", "loss_O", "loss_A", "loss_B", "loss_C", "loss_D"])
test_loss_df.head()          
# %%
def format_subplot(ax, grid_x=True):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_x:
        ax.grid(linestyle='--', alpha=0.4)
    else:
        ax.grid(axis='y', linestyle='--', alpha=0.4)
#%%
my_palette = sns.color_palette()
my_palette
# %%
sns.set_theme(style='whitegrid')

fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=600)
test_loss_df_melted = pd.melt(test_loss_df, id_vars=['timestep'], value_vars=['loss_O', 'loss_A', 'loss_B'], var_name='experiment', value_name='loss')
sns.lineplot(data=test_loss_df_melted, x="timestep", y="loss", hue="experiment", alpha=0.7, errorbar=None, ax=axes[0], palette=[my_palette[0], my_palette[1], my_palette[2]])
format_subplot(axes[0])

axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel("Timestep (Logscale)")
axes[0].set_ylabel("Test Loss (Logscale)")
axes[0].set_ylim(1e-8, 20)
axes[0].legend(title="Experiments", labels=[
    "Experiment O (No Ablation)", 
    "Experiment A (Ablating to the strongest circle)",
    "Experiment B (Ablating to two strongest circles)"], loc="lower left")

test_loss_df_melted = pd.melt(test_loss_df, id_vars=['timestep'], value_vars=['loss_O', 'loss_C', 'loss_D'], var_name='experiment', value_name='loss')
sns.lineplot(data=test_loss_df_melted, x="timestep", y="loss", hue="experiment", alpha=0.7, errorbar=None, ax=axes[1], palette=[my_palette[0], my_palette[6], my_palette[9]])
format_subplot(axes[1])

axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel("Timestep (Logscale)")
axes[1].set(ylabel=None)
axes[1].set_ylim(1e-8, 20)
axes[1].legend(title="Experiments", labels=[
    "Experiment O (No Ablation)", 
    "Experiment C (Ablating to one random circle)",
    "Experiment D (Ablating to two random circles)"], loc="lower left")

fig.suptitle("Evolution of Test Loss Over Time")
plt.savefig("./../figs/test_loss_evolution.png")
plt.show()

# %%
