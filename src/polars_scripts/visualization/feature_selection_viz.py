import sys
import os

MAIN_DIR = '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src'
MAIN_DIR1 = '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src/polars_scripts'
sys.path.insert(1, MAIN_DIR)
sys.path.insert(1, MAIN_DIR1)

import plotly.graph_objects as go
import plotly.express as px
import joblib
import numpy as np
import polars as pl
from collections import defaultdict

def load_files(fullfilespaths):
    fs_files = []
    for f in fullfilespaths:
        fs_files += os.listdir(f)

    fs_files_ordered = sorted(fs_files, key=lambda k: (k.split('_')[0], k.split('_')[4].split('-')[-1], int(k.split('_')[2].split('-')[1])))
    cat_scores = defaultdict(int)
    lg_scores = defaultdict(int)

    # INitialization for pandas approach
    # cat_scores = None
    # lg_scores = None
    for idx, file in enumerate(fs_files_ordered):
        for folder in fullfilespaths:
            if os.path.exists(os.path.join(folder, file)):
                with open(os.path.join(folder, file), 'rb')  as f:
                    data = joblib.load(f)
                break
        
        if hasattr(data['fs']['comb_model'].estimator_, 'feature_importances_'):
            for imp, name in zip(data['fs']['comb_model'].estimator_.feature_importances_, np.array(data['fs']['comb_feats'])[data['fs']['comb_model'].support_]):
                cat_scores[name] += imp
        else:
            coef_list = data['fs']['comb_model'].estimator_.coef_.tolist()[0]
            for imp, name in zip(coef_list, np.array(data['fs']['comb_feats'])[data['fs']['comb_model'].support_]):
                lg_scores[name] += imp
    return cat_scores, lg_scores


if __name__ == "__main__":
    if not os.path.exists('./cat_scores.joblib') or not os.path.exists('./lg_scores.joblib'):
        fs_dir1 = '/work/InternalMedicine/s223850/ED-StaticDynamic/final_output_sep2024/240916-1426-fsml'
        fs_dir2 = '/work/InternalMedicine/s223850/ED-StaticDynamic/final_output_sep2024/240916-1427-fsml'
        cat_scores, lg_scores = load_files([fs_dir1, fs_dir2])
        with open('./cat_scores.joblib', 'wb') as f:
            joblib.dump(cat_scores, f)
        with open('./lg_scores.joblib', 'wb') as f:
            joblib.dump(lg_scores, f)
    else:
        with open('./cat_scores.joblib', 'rb') as f:
            cat_scors = joblib.load(f)
        with open('./lg_scores.joblib', 'rb') as f:
            lg_scores = joblib.load(f)
    
    ordered_lg_scores = sorted(lg_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
    x_lg_scores = list(map(lambda x: x[0], ordered_lg_scores))[:50][::-1]
    y_lg_scores = list(map(lambda x: x[1], ordered_lg_scores))[:50][::-1]
    # plt.figure(figsize=(20, 10))
    # plt.barh(x_lg_scores, np.abs(y_lg_scores))

    fig = go.Figure(go.Bar(
        y=x_lg_scores,
        x=np.abs(y_lg_scores),  # Use absolute values as in the original code
        orientation='h'
    ))

    # Update layout to match the required style
    fig.update_layout(
        title_font_size=32,
        title_font_family='Times New Roman',
        xaxis_title_font_size=32,
        xaxis_tickfont_size=16,
        xaxis_title_font_family='Times New Roman',
        yaxis_title_font_size=32,
        yaxis_tickfont_size=16,
        yaxis_title_font_family='Times New Roman',
        yaxis_range=[0.48, 0.85],
        width=1300,
        height=800,
        template='seaborn'
)
    fig.show()