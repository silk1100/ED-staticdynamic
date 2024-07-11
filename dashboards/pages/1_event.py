from turtle import width
import streamlit as st
import polars as pl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import numpy as np

df = st.session_state['df']
bacc_mean = st.session_state['bacc_mean']
bacc_std = st.session_state['bacc_std']
ppv_mean = st.session_state['ppv_mean']
ppv_std = st.session_state['ppv_std']
feats = st.session_state['feats']

event_idx = np.arange(13, len(bacc_mean)+13)
feats_list = list(map(lambda kv: kv[1], feats.items()))
feats_list_str = list(map(lambda x: '<br>'.join(x.tolist()), feats_list))

def get_events_at(df, idx):
    v = df.group_by('PAT_ENC_CSN_ID').apply(
        lambda g: pl.DataFrame(
            {'EVENT_NAME': [g['EVENT_NAME'][idx]] if g.height>idx else ['NA']}
        )
    )
    v = v.with_columns(pl.col('EVENT_NAME').cast(pl.String))
    return v['EVENT_NAME'].value_counts().sort(by='count', descending=True).filter(pl.col("EVENT_NAME")!='NA')

@st.cache_data
def get_most_common_feats(top_idx=10):
    df_admit = df.filter(pl.col('ED_Disposition')=='Admitted')
    df_disch = df.filter(pl.col('ED_Disposition')!='Admitted')
    max_admitted_events =  df_admit.group_by('PAT_ENC_CSN_ID').agg(pl.col('EVENT_NAME').len()).max()['EVENT_NAME'][0]
    max_disch_events =  df_disch.group_by('PAT_ENC_CSN_ID').agg(pl.col('EVENT_NAME').len()).max()['EVENT_NAME'][0]
    admit_events = []
    disch_events = []
    for idx in range(0, max_admitted_events):
        res = get_events_at(df_admit, idx)['EVENT_NAME'][:top_idx].to_list()
        admit_events.append('<br>'.join(res))
        # s = ""
        # for rsv in res:
        #     s += str(rsv)+'<br>'
        # admit_events.append(s)
        
    for idx in range(0, max_disch_events):
        res = get_events_at(df_disch, idx)['EVENT_NAME'][:top_idx].to_list()
        disch_events.append('<br>'.join(res))
        # s = ""
        # for rsv in res:
        #     s += str(rsv)+'<br>'
        # disch_events.append(s)

    return admit_events, disch_events

admit_events, disch_events = get_most_common_feats(10)
st.write(admit_events)
st.write(disch_events)
df_admit = df.filter(pl.col('ED_Disposition') == 'Admitted')
df_disch = df.filter(pl.col('ED_Disposition') != 'Admitted')
all_pats = df['PAT_ENC_CSN_ID'].unique()
admit_pats = df_admit['PAT_ENC_CSN_ID'].unique()
disch_pats = df_disch['PAT_ENC_CSN_ID'].unique()
st.title('Event based analysis')
st.markdown('### Data summary')
st.write(f'num of rows: {len(df)} and # of cols: {len(df.columns)} ...')
st.write(f'total num of events for admitted patients {len(df_admit)} ({100*len(df_admit)/len(df):.2f}%)')
st.write(f'total num of events for discharged patients {len(df_disch)} ({100*len(df_disch)/len(df):.2f}%)')
st.write(f'total num of admitted patients {len(admit_pats)} ({100*len(admit_pats)/len(all_pats):.2f}%)')
st.write(f'total num of discharged patients {len(disch_pats)} ({100*len(disch_pats)/len(all_pats):.2f}%)')
st.divider()

# ===================================================================================================
st.markdown('### Data summary')

admitted_pat_id = df_admit['PAT_ENC_CSN_ID'].unique()
discharged_pat_id = df_disch['PAT_ENC_CSN_ID'].unique()
# adm_pat = st.selectbox("Select an admitted patient:", admitted_pat_id, 0)
# disc_pat = st.selectbox("Select an discharged patient:", discharged_pat_id, 0)
adm_pat = admitted_pat_id[0]

df_event_cnt = df.with_columns(
    [
        pl.arange(0,pl.col('EVENT_NAME').len()).over(pl.col('PAT_ENC_CSN_ID')).alias('count')
    ]
)

max_len_pat = df_event_cnt.group_by('PAT_ENC_CSN_ID').agg([
    pl.col('count').last().alias('num of events'),
    pl.col('ED_Disposition').last()
    ])

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

fig_hist = px.histogram(max_len_pat, x='num of events', color='ED_Disposition', marginal='box', opacity=0.6,
                        color_discrete_map=dict(Admitted='red', Discharged='blue'))
for trace in fig_hist.data:
    fig.add_trace(trace, row=1, col=1)
data = go.Scatter(
        x = event_idx,
        y = bacc_mean,
        # mode = 'lines+markers',,
        mode='lines',
        name='catboost/accuracy',
        text=feats_list_str,
        error_y=dict(
            type='data',
            array=bacc_std,
            visible=True,
            width=0.5
        ),
        line=dict(color='purple', width=0.5 ),
        hoverinfo='text+x+y'
        # marker=dict(color='purple', symbol='ci', width=0.5)
    )


fig.add_trace(data, row=2, col=1)

fig.update_layout(
    title=dict(text='Histogram of number of events by ED_Disposition', font_size=16),
    height=600,
    # width=600
)
st.plotly_chart(fig, use_container_width=True)

st.write(df_event_cnt.filter(pl.col('PAT_ENC_CSN_ID') == adm_pat).select([
    pl.col('Type'),
    pl.col('EVENT_NAME'),
    pl.col('count'),
    pl.col('ED_Disposition'),
]))