import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ml.dl_datatset import get_data_loaders
from ml.dl_model import build_combined_model
import joblib
from tqdm import tqdm
from const import constants
from torchmetrics.classification import BinaryAccuracy

from pathlib import Path
import wandb
import json

from warnings import simplefilter
simplefilter('ignore')

def train(cm, dl_tr, dl_te, loss_fnc, optimizer, epochs, device, acc, output_path, is_wandb=True):
    cm = cm.to(device)
    loss_fnc = loss_fnc.to(device)
    acc = acc.to(device)
    train_loss_list = []
    train_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    n_epochs_0_learn = 0
    best_val = 100000
    
    for e in range(epochs):
        batch_iterator = tqdm(dl_tr, desc=f"Train epoch {e}: ")

        cm.train()
        train_acc = 0
        train_loss = 0
        for data in batch_iterator:
            for key, val in data.items():
                data[key] = val.to(device)
            output = cm(data).squeeze()
            y = data['label']
            loss = loss_fnc(output, data['label'].float())
            train_loss += loss.detach().item()
            ypred = torch.where(output>=0.5, 1, 0)
            acc_score = acc(ypred, data['label'])
            train_acc += acc_score
            batch_iterator.set_postfix({"train loss": f"{loss.detach().item():2.3f}", "train acc": f"{acc_score.item():2.3f}"}) 
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
        train_acc = train_acc/len(dl_tr)
        train_loss = train_loss/len(dl_tr)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        print(f'Epoch {e}: train loss {train_loss:.2f}, train acc {train_acc:.2f} ...')

        cm.eval()
        val_acc = 0
        val_loss = 0
        batch_iterator = tqdm(dl_te, desc=f"Val epoch {e}: ")
        with torch.no_grad():
            for data in batch_iterator:
                for key, val in data.items():
                    data[key] = val.to(device)
                y = data['label']
                output = cm(data).squeeze()
                loss = loss_fnc(output, y.float())
                val_loss += loss.item()
                ypred = torch.where(output>=0.5, 1, 0)
                acc_score = acc(ypred, y)
                val_acc += acc_score
                batch_iterator.set_postfix({"val loss": f"{loss.detach().item():2.3f}", "val acc": f"{acc_score.item():2.3f}"}) 

        val_acc = val_acc/len(dl_te)
        val_loss = val_loss/len(dl_te)
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)
        print(f'Epoch {e}: val loss {val_loss:.2f}, val acc {val_acc:.2f} ...')
        print('----------------------------------------------------------------------------')
        if is_wandb:
            wandb.log({'train_loss': train_loss, "train_acc":train_acc, 'val_loss': val_loss, "val_acc":val_acc})
        torch.save({
            'model_state_dict':cm.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            },
        os.path.join(output_path, f"epoch{e}_val_{str(val_loss).replace('.','f')[:4]}.pth"))
        

        if val_loss < best_val:
            best_val = val_loss
            n_epochs_0_learn = 0
        else:
            n_epochs_0_learn += 1
        if n_epochs_0_learn>=10:
            print(f'Exited at {e} epochs due to insufficient learning ...')
            break
    
    return cm, train_loss_list, val_loss_list, train_acc_list, val_acc_list

def initialize_model(model:nn.Module):
    for p in model.parameters():
        if p.dim()>1:
            nn.init.xavier_uniform_(p)


lr = 1e-3
event_idx = 99
fold_idx = 1
batch_size = 32

def train_from_main(cm, dl_tr, dl_te, lr, eps, epochs, device, acc, output_path, event_idx, fold_idx,
                    wandb_dict=None, model_args=None):
    modelfullpath = Path(output_path) / Path(f"{event_idx}_{fold_idx}") / Path("model") #/ Path("runs")
    modelfullpath.mkdir(parents=True, exist_ok=True)
    len_runs = len(os.listdir(modelfullpath))
    modelfullpath = Path(output_path) / Path(f"{event_idx}_{fold_idx}") / Path("model") / Path(f"runs_{len_runs}")
    modelfullpath.mkdir(parents=True, exist_ok=True)

    if model_args is not None:
        with open(os.path.join(modelfullpath, 'model_dict.json'), 'w') as f:
            json.dump(model_args, f)

    loss_fnc = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(cm.parameters(), lr=lr, eps=eps)
    if wandb_dict is not None:
        boardpath = Path(output_path) / Path(f"{event_idx}_{fold_idx}") / Path("board") 
        boardpath.mkdir(parents=True, exist_ok=True)
        boardpath = Path(output_path) / Path(f"{event_idx}_{fold_idx}") / Path("board") / Path(f'runs_{len_runs}')
        boardpath.mkdir(parents=True, exist_ok=True)
        wandb_dict['dir'] = boardpath
        
        wandb_dict['name'] = wandb_dict['name'].format(event_idx, fold_idx, len_runs)
        wandb.init(
            **wandb_dict
        )
        cm, train_loss_list, val_loss_list, train_acc_list, val_acc_list = train(cm, dl_tr, dl_te, loss_fnc, optimizer, epochs, device, acc, modelfullpath, True)
    else:
        cm, train_loss_list, val_loss_list, train_acc_list, val_acc_list = train(cm, dl_tr, dl_te, loss_fnc, optimizer, epochs, device, acc, modelfullpath, False)
    
    with open(os.path.join(modelfullpath, 'train_loss_list.joblib'), 'wb') as f:
        joblib.dump(train_loss_list, f)
    with open(os.path.join(modelfullpath, 'train_acc_list.joblib'), 'wb') as f:
        joblib.dump(train_acc_list, f)
    with open(os.path.join(modelfullpath, 'val_loss_list.joblib'), 'wb') as f:
        joblib.dump(val_loss_list, f)
    with open(os.path.join(modelfullpath, 'val_acc_list.joblib'), 'wb') as f:
        joblib.dump(val_acc_list, f)


if __name__ == "__main__":
    data_path = "/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats"
    dl_tr, dl_te, vocab_dict = get_data_loaders(data_path, event_idx, fold_idx, batch_size)
    sample = next(iter(dl_tr))
    avg_embed_dict = {'s_cc': (len(vocab_dict['s_cc']), 128)}
    embed_dict = {}
    for key, vd in vocab_dict.items():
        if key.startswith('s_') and 'cc' not in key:
            embed_dict[key] = (len(vd), 8)
    
    static_dict = {
        'avg_embed_dict': avg_embed_dict,
        'embed_dict':embed_dict,
        'ff_in':sample['s_num_feats'].size(-1),
        'num_layers':1,
        'hidden_size':128,
        'output_size':64
    }

    dynamic_dict = {
        'event_vocab': len(vocab_dict['d_ev']),
        'event_dims':256,
        'dx_vocab': len(vocab_dict['d_dx']),
        'dx_dims':128,
        'seq_len':99,
        'lstm_hidden':64,
        'fc_hidden':512,
        'lstm_layers':1,
        'fc_layers':1,
        'output_size':64,
        'elapsed_time_included':True
    }
    combined_dict = {
        'hidden_layer':[256, 64],
    }
    cm = build_combined_model(static_dict, dynamic_dict, combined_dict)
    initialize_model(cm)
    loss_fnc = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(cm.parameters(), lr=1e-3, eps=1e-9)

    boardfullpath = Path(os.path.join(constants.DL_OUTPUT)) / Path("runs") / Path("99_1") / Path("board")
    boardfullpath.mkdir(exist_ok=True, parents=True)
    # writer = SummaryWriter(boardfullpath)
    acc = BinaryAccuracy()

    modelfullpath = Path(os.path.join(constants.DL_OUTPUT)) / Path("99_1") / Path("99_1") / Path("model")
    modelfullpath.mkdir(exist_ok=True, parents=True)
    wandb.init(project="ED-StaticDynamic", name="Event 99 fold 1 first trial", dir=os.path.join(constants.DL_OUTPUT, "99_1", "board"), config = dict(
        lr=1e-3,
        event_idx=99,
        fold_idx=0,
        static_model=static_dict,
        dynamic_model=dynamic_dict,
        combined_model=combined_dict,
    ))

    train(cm, dl_tr, dl_te, loss_fnc, optimizer, epochs=50, device='cuda', writer=None, acc=acc, output_path=modelfullpath)