from pickletools import optimize
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
from tqdm import tqdm
from const import constants
from torchmetrics.classification import BinaryAccuracy

from pathlib import Path
import wandb
from ml.dl_model import build_combined_model
from ml.dl_datatset import get_data_loaders
from sklearn.metrics import balanced_accuracy_score

from warnings import simplefilter
simplefilter('ignore')


if __name__ == '__main__':
    model_state_dict_path = '/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_output/runs/99_0/model/epoch20.pth'    
    dl_train, dl_test, vocab_sizes = get_data_loaders(constants.DL_FEATS_DIR, 99, 1, 32)
    config_dict ={
    "lr":  0.001,
    "static_model": {
        "ff_in": 10,
        "embed_dict": {
            "s_al": [
            8,
            8
            ],
            "s_hr": [
            25,
            8
            ],
            "s_dow": [
            9,
            8
            ],
            "s_moa": [
            14,
            8
            ],
            "s_ethn": [
            8,
            8
            ],
            "s_mnth": [
            13,
            8
            ],
            "s_firstrace": [
            9,
            8
            ]
        },
        "num_layers": 2,
        "hidden_size": 256,
        "output_size": 64,
        "avg_embed_dict": {
            "s_cc": [
            391,
            128
            ]
        }
        },
    "dynamic_model": {
        "dx_dims": 256,
        "seq_len": 99,
        "dx_vocab": 1481,
        "fc_hidden": 1024,
        "fc_layers": 1,
        "event_dims": 512,
        "event_vocab": 2653,
        "lstm_hidden": 128,
        "lstm_layers": 1,
        "output_size": 64,
        "elapsed_time_included":True 
    },
    "combined_model": {
        "hidden_layer": [
            512,
            64
        ]
        }
    } 
    cm = build_combined_model(static_dict=config_dict["static_model"], dynamic_dict=config_dict['dynamic_model'], combined_dict=config_dict["combined_model"])
    data = torch.load(model_state_dict_path)
    cm.load_state_dict(data['model_state_dict'])
    cm.eval()
    total_acc = 0
    for batch in dl_train:
        output = cm(batch).squeeze()
        total_acc += balanced_accuracy_score(torch.where(torch.sigmoid(output)>=0.5, 1, 0), batch['label'])
    total_acc = total_acc/len(dl_train)
    print(total_acc)

        
