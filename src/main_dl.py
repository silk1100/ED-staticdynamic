import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import os
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryAccuracy

from ml.dl_datatset import get_data_loaders
from ml.dl_model import build_combined_model
from ml.dl_train import train_from_main
from preprocess.dynamicDLprocessor import preprocess_for_training, preprocess_for_testing

from tqdm import tqdm
from const import constants
from torchmetrics.classification import BinaryAccuracy

from pathlib import Path
import wandb

from warnings import simplefilter
simplefilter('ignore')

'''
1. Input to the script are the preprocessing, and split training/testing dataset which was created via main.py
2. Apply DL preprocessing via scripts in dynamicDLprocessor
3. Wrap the DL preprocessed data inside a pytorch dataloaders via methods in dl_dataset
4. Initalize a DL model via methods in dl_model
5. Train the DL model via train method in dl_train
6. Validate the trained model using method in dl_validate
'''

config = {
    'preprocess':{
        'train': dict(
            ml_dir = constants.ML_DATA_OUTPUT_ID,
            ds_dir = constants.DS_DATA_OUTPUT,
            fold_idx = 0,
            event_idx = 99,
            include_date_vocabs = True,
            static_dictionaries_params = {
                'cfcg': ('Coverage_Financial_Class_Grouper', False, True),
                'cc':   ('Chief_Complaint_All', False, True),
                'ethn':  ('Ethnicity', False, True),
                'fr':   ('FirstRace', False, True),
                'al':   ('Acuity_Level', False, True),
                'ma':   ('Means_Of_Arrival', False, True)
            },
            dynamic_dictionaries_params = {
                'ev': ('Mixed_Event_Type', 'PAT_ENC_CSN_ID', False, False),
                'dx':  ('Primary_DX_ICD10', 'PAT_ENC_CSN_ID', True, False)
            },
            output_path = constants.DL_FEATS_DIR
    ),
        'test': dict(
        ml_dir = constants.ML_DATA_OUTPUT_ID,
        ds_dir = constants.DS_DATA_OUTPUT,
        df_fold_idx = 1,
        df_event_idx = 99,
        vocab_fold_idx=0,
        vocab_event_idx=99,
        static_vocab_path=constants.DL_FEATS_DIR,
        dynamic_vocab_path=constants.DL_FEATS_DIR,
        include_date_vocabs = True,
        output_path = constants.DL_FEATS_DIR
    )
    },
    'ds':{
        'pat_id':'PAT_ENC_CSN_ID',
        'didx': 'Calculated_DateTime',
        'target_col': 'Admitted_YN',
        'event_index': 99,
        'fold_idx':0,
        'batch_size':64,
        'train': True,
        'static_transformer_dict':{"Patient_Age":'minmax', 'Number of Inpatient Admissions in the last 30 Days': 'minmax',
                                            'Number of past appointments in last 60 days':'minmax',
                                            'Number of past inpatient admissions over ED visits in last three years':'minmax',
                                            'Count_of_Chief_Complaints':'minmax'},
        'dynamic_transformer_dict':{'elapsed_time_hr': 'std'},
        'static_transformer_path':os.path.join(constants.DL_FEATS_DIR, 'static_transformer_{0}_{1}.joblib'),
        'dynamic_transformer_path':os.path.join(constants.DL_FEATS_DIR, 'dynamic_transformer_{0}_{1}.joblib'),
        'dynamic_ast_cols':['Primary_DX_ICD10_seq'], # Columns which are lists and when read_csv, they are read as string. e.g. df.loc[0, 'Primary_DX_ICD10_sq'] = "['R.10', 'R.20.x']"
        'static_ast_cols':['Chief_Complaint_All_seq'], # Columns which are lists and when read_csv, they are read as string. e.g. df.loc[0, 'Primary_DX_ICD10_sq'] = "['R.10', 'R.20.x']"
    },
    'model':dict(
        static_dict = {
            # 'avg_embed_dict': avg_embed_dict # Should be added manually in the main function
            # 'embed_dict':embed_dict, # # Should be added manually in the main function
            # 'ff_in':sample['s_num_feats'].size(-1),# Should be added manually in the main function
            'num_layers':1,
            'hidden_size':64,
            'output_size':128
        },

        dynamic_dict = {
            # 'event_vocab': len(vocab_dict['d_ev']),
            'event_dims':80,  # 1 Last updated from 256
            # 'dx_vocab': len(vocab_dict['d_dx']),
            'dx_dims':80,
            'seq_len':99,
            'lstm_hidden':128,
            'fc_hidden':160,
            'lstm_layers':1,
            'fc_layers':2,
            'output_size':180,
            'elapsed_time_included':True
        },
        combined_dict = {
            'hidden_layer':[128, 32], # 0 Last updated to [128, 32] was bad 
        }
    ),
    'train':dict(
        lr=2e-4,
        event_idx=99,
        fold_idx=0,
        eps=1e-9,
        epochs=50,
        device='cuda',
        output_path = constants.DL_OUTPUT,
        wandb_dict=dict(
            project="ED-StaticDynamic-DL",
            name="(Different static/dynamic combination low dim output fc) Event {0} fold_idx {1} trial {2}",
            dir = constants.DL_OUTPUT
        )
    )
}

if __name__ == "__main__":
    static_dynamic_df, static_vocab_dict1, dynamic_vocab_dict1 = preprocess_for_training(**config['preprocess']['train']) 
    # static_dynamic_df_test, static_vocab_dict, dynamic_vocab_dict = preprocess_for_testing(**config['preprocess']['test'])
    # dl_tr, dl_te, vocab_dict = get_data_loaders(static_dynamic_df, **config['ds'])
    dl_tr, dl_te = get_data_loaders(static_dynamic_df, **config['ds'])
    sample = next(iter(dl_tr))['s_num_feats']
    config['model']['static_dict']['avg_embed_dict'] = {'s_cc': (len(static_vocab_dict1['cc']), 64)} 
    embed_dict = {}
    for key, vd in static_vocab_dict1.items():
        if 'cc' not in key:
            embed_dict['s_'+key] = (len(vd), 5)
    
    config['model']['static_dict']['embed_dict'] = embed_dict 
    config['model']['static_dict']['ff_in'] = sample.size(-1)
    config['model']['dynamic_dict']['event_vocab'] = len(dynamic_vocab_dict1['ev'])
    config['model']['dynamic_dict']['dx_vocab'] = len(dynamic_vocab_dict1['dx'])
    model = build_combined_model(**config['model'])
    
    
    config['train']['cm'] = model
    config['train']['dl_tr'] = dl_tr 
    config['train']['dl_te'] = dl_te
    config['train']['acc'] = BinaryAccuracy()
    config['train']['model_args'] = config['model']
    cd = config['preprocess']
    cd.update(config['ds'])
    cd.update(config['model'])
    cd.update({key: val for key, val in config['train'].items() if 'wandb' not in key})
    
    config['train']['wandb_dict']['config'] = cd

    train_from_main(**config['train'])

    x = 0
    
    