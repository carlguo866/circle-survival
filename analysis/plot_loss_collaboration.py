#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import seaborn as sns
import pandas as pd
from utils import format_subplot, get_final_circle_freqs
from tqdm import tqdm

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

torch.set_default_tensor_type(torch.DoubleTensor)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
# %%
class MLP(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, w=2, depth=2, shp=None, embedding_size=None):
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
        self.embedding = torch.nn.Parameter(torch.normal(0,1,size=embedding_size))
        self.shp = shp

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
    
# %%
p = 59
d = 128

def get_model():
    
    in_dim = 2*d
    out_dim = p

    depth = 2

    shp = [in_dim] + [100] * depth + [out_dim]
    model = MLP(shp=shp, embedding_size=(p, d))
    model = model.to(device)
    return model
# %%
x = np.arange(p)
y = np.arange(p)
XX, YY = np.meshgrid(x, y)
data_id = np.transpose([XX.reshape(-1,), YY.reshape(-1,)])
labels = (data_id[:,0] + data_id[:,1]) % p
labels = torch.tensor(labels, dtype=torch.long)

def get_train_test(fraction):
    train_num = int(p**2*fraction)
    test_num = p**2 - train_num

    train_id = np.random.choice(p**2,train_num,replace=False)
    test_id = np.array(list(set(np.arange(p**2)) - set(train_id)))

    return train_id, test_id
# %%
def get_data(model, id_, cur_embedding=None):
    global labels
    if cur_embedding is not None:
        inputs = torch.cat([torch.from_numpy(cur_embedding[data_id[id_][:,0]]), torch.from_numpy(cur_embedding[data_id[id_][:,1]])], dim=1)
    else:
        inputs = torch.cat([model.embedding[data_id[id_][:,0]], model.embedding[data_id[id_][:,1]]], dim=1)
    return inputs, labels[id_]

def fuse_parameters(model):
    n = sum(p.numel() for p in model.parameters())
    params = torch.zeros(n).to(device).to(torch.double)
    i = 0
    for p in model.parameters():
        params_slice = params[i:i + p.numel()]
        params_slice.copy_(p.flatten())
        p.data = params_slice.view(p.shape)
        i += p.numel()
    return params

def set_parameters(model, params):
    params = torch.tensor(params).to(device)
    i = 0
    for p in model.parameters():
        p.data = params[i:i + p.numel()].view(p.shape)
        i += p.numel()

# %%
data_path = "data/fix_mlp/seed_10"
embeddings = np.load(data_path + "/embeddings.npy")
model_params = np.load(data_path + "/model_params.npy")
embeddings.shape, model_params.shape
# %%
circles = get_final_circle_freqs(embeddings)
circles
# %%
model = get_model()

def get_loss(params, id, t, freqs):
    embedding = embeddings[t * 100]
    spectrum = np.fft.fft(embedding, axis=0)
    for f in range(1, 30):
        if f not in freqs:
            spectrum[f, :] = spectrum[59 - f, :] = 0
    
    cur_embedding = np.fft.ifft(spectrum, axis=0).real
    
    inputs, labels = get_data(model, id, cur_embedding)
    inputs = inputs.to(device)
    labels = labels.to(device)
    
    CEL = nn.CrossEntropyLoss()
    set_parameters(model, params)
    pred = model(inputs)
    loss = CEL(pred, labels)
    acc = torch.mean((torch.argmax(pred, dim=1) == labels).float())
    return loss.cpu().detach().item(), acc.cpu().detach().item()

get_loss(model_params[-1], np.arange(p * p), -1, circles)
# %%
entries = []

for i in range(1, 30):
    ind = 0 if not i in circles else circles.index(i) + 1
    for t, param in tqdm(enumerate(model_params)):
        loss, acc = get_loss(param, np.arange(p * p), t, [i])
        entries.append([i, ind, t, loss, acc])

one_circle_df = pd.DataFrame(entries, columns = ["frequency", "index", "timestep", "loss", "accuracy"])
one_circle_df.head()
# %%
entries = []

for i in range(1, 30):
    ind = 0 if not i in circles else circles.index(i) + 1
    for t, param in tqdm(enumerate(model_params)):
        loss, acc = get_loss(param, np.arange(p * p), t, [2, i])
        entries.append([i, ind, t, loss, acc])

two_circles_df = pd.DataFrame(entries, columns = ["frequency", "index", "timestep", "loss", "accuracy"])
two_circles_df.head()
# %%
entries = []

for i in range(1, 30):
    ind = 0 if not i in circles else circles.index(i) + 1
    for t, param in tqdm(enumerate(model_params)):
        loss, acc = get_loss(param, np.arange(p * p), t, [2, 22, i])
        entries.append([i, ind, t, loss, acc])

three_circles_df = pd.DataFrame(entries, columns = ["frequency", "index", "timestep", "loss", "accuracy"])
three_circles_df.head()
# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), dpi=700)

sns.set_theme(style='whitegrid')

sns.lineplot(data=one_circle_df[one_circle_df["index"] == 0], x="timestep", y="loss", hue="frequency", alpha=0.1, legend=False, palette=["grey"], ax=axes[0])
sns.lineplot(data=one_circle_df[one_circle_df["index"] > 0], x="timestep", y="loss", hue="index", palette="viridis", ax=axes[0])

format_subplot(axes[0])

axes[0].set_xlabel("Timestep")
axes[0].set_ylabel("Test Loss")
axes[0].set_title("One Circle", fontsize=9)
axes[0].set_xticks(range(0, 300, 50), [100 * i for i in range(0, 300, 50)], rotation=45)
axes[0].set_ylim(0, 4.5)

new_title = "Frequency"
new_labels = ["2", "8", "22", "23", "3"]
leg = axes[0].get_legend()
leg.set_title(new_title)
for t, l in zip(leg.texts, new_labels):
    t.set_text(l)

sns.lineplot(data=two_circles_df[two_circles_df["index"] == 0], x="timestep", y="loss", hue="frequency", alpha=0.1, legend=False, palette=["grey"], ax=axes[1])
sns.lineplot(data=two_circles_df[two_circles_df["index"] > 1], x="timestep", y="loss", hue="index", legend=False, palette="viridis", ax=axes[1])

format_subplot(axes[1])

axes[1].set_xlabel("Timestep")
axes[1].set(ylabel=None)
axes[1].set_title("Two Circles", fontsize=9)
axes[1].set_xticks(range(0, 300, 50), [100 * i for i in range(0, 300, 50)], rotation=45)
axes[1].set_ylim(0, 4.5)

sns.lineplot(data=three_circles_df[three_circles_df["index"].isin([0, 1, 3])], x="timestep", y="loss", hue="frequency", alpha=0.1, legend=False, palette=["grey"], ax=axes[2])
sns.lineplot(data=three_circles_df[three_circles_df["index"].isin([2, 4, 5])], x="timestep", y="loss", hue="index", legend=False, palette="viridis", ax=axes[2])

format_subplot(axes[2])

axes[2].set_xlabel("Timestep")
axes[2].set(ylabel=None)
axes[2].set_title("Three Circles", fontsize=9)
axes[2].set_ylim(0, 4.5)
axes[2].set_xticks(range(0, 300, 50), [100 * i for i in range(0, 300, 50)], rotation=45)

fig.suptitle("Test Loss Over Time Using Different Circles", fontsize=11)

plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.8,
                    wspace=0.3, 
                    hspace=0.4)

plt.savefig("figs/interactions.png")
plt.show()
# %%