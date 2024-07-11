import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from ml.dl_datatset import get_data_loaders



'''
1. Preprocess Static files the same way as I do for regular ML
2. Get all of the dynamic sequence (at index i)
    2.1. Create an embedding layer that accomodate for all valid events
    2.2. Create an LSTM to predict 
'''

class AvgEmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embedding_size, padd_idx=0):
        super(AvgEmbeddingLayer, self).__init__()
        self.embed_layer = nn.Embedding(vocab_size, embedding_size, padding_idx=padd_idx)
    
    def forward(self, x):
        out = self.embed_layer(x)
        mask = x!=0
        mean_embed = (out * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        return mean_embed


class StationaryModel(nn.Module):
    def __init__(self, avg_embed_dict, embed_dict, ff_in, hidden_size, num_layers, output_size):
        super(StationaryModel, self).__init__()
        self.avg_embed_dict = nn.ModuleDict()
        self.embed_dict = nn.ModuleDict()
        self.total_embeddings = 0
        # Assuming that padding idx is always 0
        for feat_key, (vocab_size, embed_size) in avg_embed_dict.items():
            self.avg_embed_dict[feat_key] = AvgEmbeddingLayer(vocab_size, embed_size, 0)
            self.total_embeddings += embed_size

        for feat_key, (vocab_size, embed_size) in embed_dict.items():
            self.embed_dict[feat_key] = nn.Embedding(vocab_size, embed_size, 0)
            self.total_embeddings += embed_size

        num_feats_model = [nn.Linear(ff_in, hidden_size), nn.ReLU()]
        for nl in range(1, num_layers):
            num_feats_model.append(nn.Linear(hidden_size, hidden_size))
            if nl%2==0:
                num_feats_model.append(nn.Dropout(0.1))
            num_feats_model.append(nn.ReLU())

        self.ff_num_feats = nn.Sequential(
            *num_feats_model
        )

        self.combiner = nn.Linear(self.total_embeddings+hidden_size, output_size)
        self.output_size = output_size
        
        
    def forward(self, static_data_dict):
        # batch_size = static_data_dict['s_num_feats'].size(0)

        avg_embed = None
        for key, model in self.avg_embed_dict.items():
            out_avg_embed = model(static_data_dict[key+'_seq'])
            if avg_embed is None:
                avg_embed = out_avg_embed
            else:
                avg_embed = torch.cat([avg_embed, out_avg_embed], dim=-1)

        embed = None
        for key, model in self.embed_dict.items():
            out_embed = model(static_data_dict[key+'_seq'])
            if embed is None:
                embed = out_embed
            else:
                embed = torch.cat([embed, out_embed], dim=-1)

        out_hidden = self.ff_num_feats(static_data_dict['s_num_feats'])

        all_feats = torch.cat([avg_embed, embed, out_hidden], dim=-1)
        return self.combiner(all_feats)


class DynamicModel(nn.Module):
    def __init__(self, event_vocab, event_dims, dx_vocab, dx_dims, seq_len, lstm_hidden, fc_hidden, lstm_layers, fc_layers, output_size, elapsed_time_included=True):
        super(DynamicModel, self).__init__()
        self.elapsed_time = elapsed_time_included
        self.event_embed = nn.Embedding(event_vocab, event_dims)
        self.dx_embed = nn.Embedding(dx_vocab, dx_dims)

        total_feats = lstm_hidden*3 if elapsed_time_included else 2*lstm_hidden

        self.event_lstm = nn.LSTM(event_dims, lstm_hidden, lstm_layers, batch_first=True)
        self.dx_lstm = nn.LSTM(dx_dims, lstm_hidden, lstm_layers, batch_first=True)
        self.el_lstm = nn.LSTM(1, lstm_hidden, 1, batch_first=True)

        fc_list = [nn.Linear(total_feats, fc_hidden), nn.ReLU()]
        for l in range(1, fc_layers-1):
            fc_list.append(nn.Linear(fc_hidden, fc_hidden))
            if l%2 == 0:
                fc_list.append(nn.Dropout(0.1))
            fc_list.append(nn.ReLU())
        fc_list.append(nn.Linear(fc_hidden, output_size))
        self.fc = nn.Sequential(*fc_list)
        self.output_size = output_size

    def forward(self, data_dict):
        #(batchsize, seqlen) => (#batchsize, seqlen, dimlen)
        ev_embed = self.event_embed(data_dict['d_ev_seq']) 
        #(batchsize, seqlen) => (#batchsize, seqlen, dimlen)
        dx_embed = self.dx_embed(data_dict['d_dx_seq'])
        dx_embed = dx_embed.mean(dim=2)
        
        output, (hidden_ev, cell) = self.event_lstm(ev_embed)
        output, (hidden_dx, cell) = self.dx_lstm(dx_embed)

        output, (hidden_el, cell) = self.el_lstm(data_dict['d_elt'].unsqueeze(-1))
        
        if self.elapsed_time:
            all_feats = torch.cat([hidden_ev[-1], hidden_dx[-1], hidden_el[-1]], dim=-1)
        else:
            all_feats = torch.cat([hidden_ev[-1], hidden_dx[-1]], dim=-1)
        
        out = self.fc(all_feats)
        return out


class CombinedModel(nn.Module):
    def __init__(self, static_model: StationaryModel, dynamic_model: DynamicModel, hidden_layer=[]):
        super(CombinedModel, self).__init__()
        self.sm = static_model
        self.dm = dynamic_model
        self.feat_size = self.sm.output_size + self.dm.output_size
        
        if len(hidden_layer)==0:
            self.fc =  nn.Linear(self.feat_size, 1)
        else:
            fc_list = [nn.Linear(self.feat_size, hidden_layer[0]), nn.ReLU()]
            for lidx in range(1,len(hidden_layer)):
                fc_list.append(nn.Linear(hidden_layer[lidx-1], hidden_layer[lidx]))
                if lidx%2 == 0:
                    fc_list.append(nn.Dropout(0.1))
                fc_list.append(nn.ReLU())
            
            fc_list.append(nn.Linear(hidden_layer[-1], 1))
            self.fc = nn.Sequential(*fc_list)
    
    def forward(self, data_dict):
        sout = self.sm(data_dict)
        dout = self.dm(data_dict)
        feats = torch.cat([sout, dout], dim=-1)
        return self.fc(feats)
        

def build_combined_model(
    static_dict,
    dynamic_dict,
    combined_dict
    
):

    sm = StationaryModel(**static_dict)
    dm = DynamicModel(**dynamic_dict)
    cm = CombinedModel(sm, dm, **combined_dict)
    return cm

        
if __name__ == '__main__':
    data_path = "/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_dl_feats"
    dl_tr, dl_te, vocab_dict = get_data_loaders(data_path, 99, 0, 32)
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
        'num_layers':2,
        'hidden_size':256,
        'output_size':64
    }

    dynamic_dict = {
        'event_vocab': len(vocab_dict['d_ev']),
        'event_dims':512,
        'dx_vocab': len(vocab_dict['d_dx']),
        'dx_dims':256,
        'seq_len':99,
        'lstm_hidden':128,
        'fc_hidden':1024,
        'lstm_layers':1,
        'fc_layers':1,
        'output_size':64,
        'elapsed_time_included':True
    }
    combined_dict = {
        'hidden_layer':[512, 64],
    }
    # DynamicModel()
    # model = StationaryModel(avg_embed_dict, embed_dict, sample['s_num_feats'].size(-1), 32, 2, 64)
    # model = DynamicModel(len(vocab_dict['d_ev']), 512, len(vocab_dict['d_dx']), 256, 99, 128, 1024, 1, 2, 512, True)
    # out = model(sample)
    cm = build_combined_model(static_dict, dynamic_dict, combined_dict)
    x = 0