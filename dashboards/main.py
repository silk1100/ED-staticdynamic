import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import os
import joblib
# st.set_page_config(
#     page_title="Ex-stream-ly Cool App",
#     page_icon="?",
#     layout="wide",
#     initial_sidebar_state="expanded",
#     menu_items={
#         'Get Help': 'https://www.extremelycoolapp.com/help',
#         'Report a bug': "https://www.extremelycoolapp.com/bug",
#         'About': "# This is a header. This is an *extremely* cool app!"
#     }
# )

main_dir = "/project/InternalMedicine/Basit_lab/s223850/ED-StaticDynamic/ml_results_240324/tr_8_te_4/run_1/"

static_dir = os.path.join(main_dir, 'static')
dynamic_dir = os.path.join(main_dir, 'dynamic')
comb_dir = os.path.join(main_dir, 'comb')
all_dir = os.path.join(main_dir, 'all')

@st.cache_data(persist="disk", show_spinner=True)
def read_data(dir='/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events - 12.21.23.csv'):
    df = pl.read_csv(dir,
                    infer_schema_length=int(1e6),
                    null_values=dict(Patient_Age='NULL'),
                    dtypes=dict(Patient_Age=pl.Float64)
                    )
    df = df.with_columns([
        pl.col("Arrived_Time").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f"),
        pl.col("Calculated_DateTime").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f"),
    ])
    df = df.filter(
        (pl.col("Calculated_DateTime").dt.year()>=2021)&(pl.col("ED_Disposition").is_not_null())&(pl.col("ED_Disposition").is_in(['Admitted', 'Discharged']))
        ).sort("Calculated_DateTime")
    
    admitted_with_order = df.filter(pl.col('ED_Disposition')=='Admitted').\
                      group_by('PAT_ENC_CSN_ID').\
                      agg([
                        pl.col('Type').is_in(['Order - Admission']).alias('has_admission_order')
                        #   pl.col('Type').apply(lambda x: 'Order - Admission' in x, return_dtype=pl.Boolean).alias('has_admission_order')
                      ])
    admi_pats_with_no_order = admitted_with_order.filter(pl.col('has_admission_order')==False).select(pl.col('PAT_ENC_CSN_ID'))
    df = df.filter(~pl.col('PAT_ENC_CSN_ID').is_in(admi_pats_with_no_order))
    dd = df.group_by("PAT_ENC_CSN_ID").map_groups(lambda x: x[:np.where(x['Type']=='Order - Admission')[0][0]+1] if 'Order - Admission' in x['Type'] else x)
    de = dd.with_columns([
    (pl.col("Calculated_DateTime")-pl.col("Arrived_Time")).dt.total_seconds().alias('elapsed_time_sec'),
    (pl.col("Calculated_DateTime")-pl.col("Arrived_Time")).dt.total_hours().alias('elapsed_time_hr'),
    ])
    dee = de.group_by(
    'PAT_ENC_CSN_ID'
    ).map_groups(
        lambda x: x.with_columns((-1*(pl.col("elapsed_time_sec")-x.get_column("elapsed_time_sec")[-1])).alias('inv_elapsed_sec'))
    )
    dee = dee.with_columns(
        (pl.col("inv_elapsed_sec")/60).alias('inv_elapsed_min'),
        (pl.col("inv_elapsed_sec")/60/60).alias('inv_elapsed_hr'),
    )
    return dee

def jobobj2featmat(f1, modelname='cat', topfeats=20):
    return pd.DataFrame(f1[modelname]['model'].feature_importances_, index=f1[modelname]['model'].feature_names_).reset_index().rename(columns={'index':'names', 0:'importance'}).sort_values(by='importance', ascending=False).iloc[:topfeats]

@st.cache_data
def get_acc_ppv_4_catboost(all_sorted_feats):
    event_index_feats = {}
    bacc_mean = []
    bacc_std = []

    ppv_mean = []
    ppv_std = []

    event_idx = []
    for idx in range(0, len(all_sorted_feats), 2):
        iidx = int(all_sorted_feats[idx].split('_')[2])
        event_idx.append(iidx)
        assert(all_sorted_feats[idx].split('_')[2] == all_sorted_feats[idx+1].split('_')[2])
        with open(os.path.join(all_dir, all_sorted_feats[idx]), 'rb') as f:
            f1 = joblib.load(f)
            feats1 = jobobj2featmat(f1)
        with open(os.path.join(all_dir, all_sorted_feats[idx+1]), 'rb') as f:
            f2 = joblib.load(f)
            feats2 = jobobj2featmat(f2)
        common_events = np.intersect1d(feats1['names'].values, feats2['names'].values)
        event_index_feats[iidx] = common_events
        bacc_mean.append(f1['cat']['val_score_avg']['bacc'])
        bacc_std.append(f1['cat']['val_score_std']['bacc'])
        ppv_mean.append(f1['cat']['val_score_avg']['ppv'])
        ppv_std.append(f1['cat']['val_score_std']['ppv'])
    return bacc_mean, bacc_std, ppv_mean, ppv_std, event_index_feats


if 'df' not in st.session_state:
    df = read_data('/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events - 12.21.23.csv')
    st.session_state['df'] = df
else:
    df = st.session_state['df']


all_sorted_feats = sorted(os.listdir(all_dir), key=lambda x: (int(x.split('_')[2]),int(x.split('_')[-1].split('.')[0])))

bacc_mean, bacc_std, ppv_mean, ppv_std, event_index_feats = get_acc_ppv_4_catboost(all_sorted_feats)

st.session_state['bacc_mean'] = bacc_mean
st.session_state['bacc_std'] = bacc_std
st.session_state['ppv_mean'] = ppv_mean
st.session_state['ppv_std'] = ppv_std
st.session_state['feats'] = event_index_feats