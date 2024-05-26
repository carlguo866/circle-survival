import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Ellipse, Circle
import os
import sys
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

def train(model, train_id, test_id, args, data_id, labels):
    log = args.log
    train_accs = []
    test_accs = []
    train_losses = []
    test_losses = []
    l2s = []
    model_params = []
    data_log = args.data_log
    steps = args.steps
    print_log = args.print_log
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=args.wd,foreach=False)
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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_seed", type=int, default=0)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--end_seed", type=int, default=1)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--p", type=int, default=59)
    parser.add_argument("--log", type=int, default=100)
    parser.add_argument("--data_log", type=int, default=1)
    parser.add_argument("--steps", type=int, default=30001)
    parser.add_argument("--print_log", type=bool, default=False)
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--fix_embedding", type=bool, default=False)
    parser.add_argument("--fix_mlp", type=bool, default=False)
    parser.add_argument("--fix_dataset", type=bool, default=False)
    parser.add_argument("--freeze_embedding", type=bool, default=False)
    parser.add_argument("--wd", type=float, default=0.5)
    
    args = parser.parse_args()
    start_seed = args.start_seed
    end_seed = args.end_seed
    print(f"doing work on seeds {start_seed} to {end_seed} exclusive")
    print(args)
    
    dim = args.dim
    p = args.p
    
    x = np.arange(p)
    y = np.arange(p)
    XX, YY = np.meshgrid(x, y)
    data_id = np.transpose([XX.reshape(-1,), YY.reshape(-1,)])
    labels = (data_id[:,0] + data_id[:,1]) % p
    labels = torch.tensor(labels, dtype=torch.long)
    
    id = uuid.uuid4().hex[:8]  
    print("id", id)

    freeze_embedding = args.freeze_embedding

    model = get_model(args.init_seed, dim, p, freeze_embedding=freeze_embedding)
    init_mlp = fuse_mlp(model)
    init_embedding = model.embedding.cpu().detach().numpy()
    train_id, test_id = get_train_test(args.init_seed, 0.8, p)


    for seed in tqdm(range(start_seed, end_seed)):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.fix_embedding: 
            model = get_model(seed, dim, p, embedding=init_embedding, freeze_embedding=freeze_embedding)
            train_id, test_id = get_train_test(seed, 0.8, p)
        elif args.fix_mlp: 
            print("fix_mlp")
            init_embed = torch.normal(0,1,size=(p, dim)).detach().numpy()
            set_mlp_and_embedding(model, init_mlp, init_embed, freeze_embedding=freeze_embedding)
        elif args.fix_dataset: 
            model = get_model(seed, dim, p, freeze_embedding=freeze_embedding)
        else: 
            model = get_model(seed, dim, p, freeze_embedding=freeze_embedding)
            train_id, test_id = get_train_test(seed, 0.8, p)
            
        data_path = f"{args.data_path}/dim_{args.dim}_init_seed_{args.init_seed}_vary_seed_{seed}_p_{p}_id_{id}"
        os.makedirs(data_path,exist_ok=True)
    
        np.save(os.path.join(data_path, 'init_mlp.npy'), init_mlp.cpu().detach().numpy())
        np.save(os.path.join(data_path, 'init_params.npy'), fuse_parameters(model).cpu().detach().numpy())
        np.save(os.path.join(data_path, 'init_embed.npy'), model.embedding.cpu().detach().numpy())
    
        returns = train(model, train_id, test_id, args, data_id, labels)

        print("init: ", returns["first_embeds"][0])
        print("final: ", returns["final_embed"])
        
        np.save(os.path.join(data_path, 'embeddings.npy'), returns['embeddings'])
        np.save(os.path.join(data_path, 'gradients.npy'), returns['gradients'])
        np.save(os.path.join(data_path, 'model_params.npy'), returns['model_params'])
        np.save(os.path.join(data_path, 'first_embeds.npy'), returns['first_embeds'])
        np.save(os.path.join(data_path, 'final_embed.npy'), returns['final_embed'])
        np.save(os.path.join(data_path, 'train_acc.npy'), returns['train_accs'])
        np.save(os.path.join(data_path, 'test_acc.npy'), returns['test_accs'])
        np.save(os.path.join(data_path, 'train_loss.npy'), returns['train_losses'])
        np.save(os.path.join(data_path, 'test_loss.npy'), returns['test_losses'])
        np.save(os.path.join(data_path, 'l2.npy'), returns['l2s'])
        np.save(os.path.join(data_path, 'train_id.npy'), train_id)
        np.save(os.path.join(data_path, 'test_id.npy'), test_id)
        torch.save(model.state_dict(), os.path.join(data_path, 'model.pt'))

        
        