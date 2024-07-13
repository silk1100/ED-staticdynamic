import os
import sys

sys.path.insert(1, os.getenv("EDStaticDynamic"))
import polars as pl
import numpy as np
from const import constants
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pandas.tseries.holiday import USFederalHolidayCalendar
import pandas as pd
from typing import List
import joblib

USCal = USFederalHolidayCalendar()

def get_holiday(x):
    n = USCal.holidays(x, x, return_name=True)
    if len(n) == 0:
        return 'null'
    else:
        return n.iloc[0]

def get_holiday_optimized(df, date_col):
    newdatecol = f'{date_col}_date'
    df = df.with_columns(
        pl.col(date_col).dt.date().alias(newdatecol)
    )
    start_date = df[newdatecol].to_pandas().min()
    end_date = df[newdatecol].to_pandas().max()
    
    df_holiday = USCal.holidays(start_date, end_date, return_name=True).to_frame().reset_index()
    df_holiday.columns = [newdatecol, 'holiday']
    df_holiday[newdatecol] = df_holiday[newdatecol].dt.date
    df_holiday = pl.DataFrame(df_holiday)
    df = df.join(df_holiday, on=newdatecol, how='inner')
    df = df.drop(newdatecol)
    if 'holiday_right' in df.columns:
        df = df.rename({'holiday_right': 'holiday'})
    return df

def read_and_clean(file, infer_length=20e6, sample_encs=0):
    df = pl.read_csv(file,
        infer_schema_length=int(infer_length),
        null_values='NULL',
        dtypes=dict(Patient_Age=pl.Float64)
    )
    # TODO: Replaced by create_sample_df
    # if sample_encs is not None and sample_encs > 0:
    #     pats = df['PAT_ENC_CSN_ID'].unique()
    #     rpats = np.random.choice(pats, sample_encs, replace=False)
    #     df = df.filter(pl.col('PAT_ENC_CSN_ID').is_in(rpats))
    
    
    df = df.filter(pl.col('ED_Disposition').is_in(['Admitted', 'Discharged']))

    df = category_mappers_static(df)

    if df['Arrived_Time'].dtype == pl.String:  
        df = df.with_columns([
            pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
            # pl.col('Calculated_DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
        ])

    if df['Calculated_DateTime'].dtype == pl.String:
        df = df.with_columns([
            # pl.col('Arrived_Time').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f"),
            pl.col('Calculated_DateTime').str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S%.f")
        ])  
    df = df.sort(by=['PAT_ENC_CSN_ID', 'Calculated_DateTime'])
    # df = df.with_columns((pl.col('elapsed_time_min').shift(-1)-pl.col('elapsed_time_min')).over('PAT_ENC_CSN_ID').alias('processing_time_min'))

    # Create new target column
    df = df.with_columns(
        (pl.col("Type") == 'Order - Admission').sum().over('PAT_ENC_CSN_ID').alias('admit-order-cnt')
    ).with_columns(
        pl.when(pl.col('admit-order-cnt')==0).
        then(0).otherwise(1).alias('has_admit_order')
    ).drop(columns=['admit-order-cnt'])

    df_trim = df.with_columns(
        (pl.col('Type')=='Order - Admission').cum_sum().over('PAT_ENC_CSN_ID').alias('n_admit_orders'),
        (pl.col('Type')=='Order - Discharge').cum_sum().over('PAT_ENC_CSN_ID').alias('n_disch_orders'),
    ).filter( (pl.col('n_admit_orders')<1) & (pl.col('n_disch_orders')<1) )

    df_trim = df_trim.with_columns([
        ((pl.col('Calculated_DateTime')-pl.col('Arrived_Time')).dt.total_seconds().cast(pl.Float32)/60.0).alias('elapsed_time_min')
    ])

    df_trim = df_trim.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32).over('PAT_ENC_CSN_ID').alias('event_idx')
    )

    df_trim = df_trim.with_columns(
        pl.col('elapsed_time_min').last().over('PAT_ENC_CSN_ID').alias('tta'),
        pl.col('event_idx').last().over('PAT_ENC_CSN_ID').alias('eidx_ed')
    )

    df_trim = df_trim.with_columns(
        [
            pl.col("Chief_Complaint_All").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').
                map_elements(lambda x: x.split(','), return_dtype=pl.List(pl.String)).alias('cc_list'),

            # pl.col("Primary_DX_Name").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').
            #     map_elements(lambda x: x.split(','), return_dtype=pl.List(pl.String)).alias('dx_list'),

            pl.col("Primary_DX_ICD10").map_elements(lambda x: x.split(','), return_dtype=pl.List(pl.String)).
                alias('dxcode_list'),
            pl.col("EVENT_NAME").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').alias('EVENT_NAME_NORM'),
            pl.col("Type").str.to_lowercase().str.replace_all(r'[^a-zA-Z0-9_,/]+','').alias('Type_NORM'),
            pl.col('Arrived_Time').dt.day().alias('arr_day'),
            pl.col('Arrived_Time').dt.weekday().alias('arr_dow'),
            pl.col('Arrived_Time').dt.year().alias('arr_year'),
            pl.col('Arrived_Time').dt.month().alias('arr_month'),
            pl.col('Arrived_Time').dt.hour().alias('arr_hour'),
            # pl.col('Arrived_Time').map_elements(lambda x: get_holiday(x.date()), return_dtype=pl.String).alias('holiday')
        ]
    ).with_columns(
        (pl.col('Type_NORM')+'_'+pl.col('EVENT_NAME_NORM')).alias('type_name')
    )
    df_trim = get_holiday_optimized(df_trim, 'Arrived_Time')
    for k, v in constants.vital_ranges_dict.items():
        df_trim = df_trim.with_columns(
            pl.when((pl.col('EVENT_NAME') == k)&((pl.col("MEAS_VALUE")>v[1])|(pl.col("MEAS_VALUE")<v[0])))
            .then(pl.lit(None)).otherwise(pl.col("MEAS_VALUE")).alias("MEAS_VALUE_UP")
        )

    # Split MEAS_VALUE based on the different vitals (11 fields)
    uniq_vitals = df_trim.filter(pl.col('Type') == 'Vitals')['EVENT_NAME'].unique()
    df_trim = df_trim.with_columns(
        [
            pl.when(
                (pl.col("EVENT_NAME")==uq)&(pl.col("MEAS_VALUE_UP").is_not_null())
            ).then(
                pl.col("MEAS_VALUE_UP")
            ).otherwise(pl.lit(None)).alias(f'MEAS_VALUE_{uq}')
            for uq in uniq_vitals
        ]
    )

    df_trim = df_trim.with_columns(
        [
            pl.when(pl.col(c).is_in(constants.NULL_LIST)).then(pl.lit('null')).otherwise(pl.col(c)).alias(c)
            for c in constants.static_singleval_cat_cols
        ]
    )

    return df_trim

def create_vocabs_for_dep_cols(df:pl.DataFrame, colnames:List[str], inc_cum_thresh: float, multival=False):
    '''
    Currently handling dependence between only two colunms 
    '''
    vocab_dict = {}
    uq1 = df[colnames[0]].unique()
    for u1 in uq1:
        vocab_dict[u1] = (0, {f'null_{colnames[1]}_{u1}': 0, f'rare_{colnames[1]}_{u1}':1})
        dd = df.filter(pl.col(colnames[0]) == u1)[colnames[1]].value_counts().sort(by='count', descending=True)
        if len(dd) > 100:
            dd = dd.with_columns(
                pl.col('count').cum_sum().alias('cumsum')
            ).with_columns(
                (pl.col('cumsum')/pl.col('cumsum').max()).alias('prob')
            ).with_columns(
                (pl.col('prob')<=inc_cum_thresh).alias('included')
            )
            inc_df = dd.filter(pl.col('included'))
            inc_vals = inc_df[colnames[1]].to_list()

            vocab_dict[u1][1].update(dict(zip(inc_vals, list(range(2, len(inc_vals) + 2)))))
        else:
            inc_df = dd     

        inc_vals = inc_df[colnames[1]].to_list()
        vocab_dict[u1][1].update(dict(zip(inc_vals, list(range(2, len(inc_vals) + 2)))))

    return vocab_dict

def create_vocabs_for_cols(df, colname, inc_cum_thresh, multival=False):
    '''
    colname is assumed to be already converted into list e.g. cc_list, dx_list 
    '''
    if multival:
        dd = df.explode(colname)[colname].value_counts().sort(by='count', descending=True)
    else:
        dd = df[colname].value_counts().sort(by='count', descending=True)
    dd = dd.with_columns(
        pl.col('count').cum_sum().alias('cumsum')
    ).with_columns(
        (pl.col('cumsum')/pl.col('cumsum').max()).alias('prob')
    ).with_columns(
        (pl.col('prob')<=inc_cum_thresh).alias('included')
    )
    inc_df = dd.filter(pl.col('included'))
    inc_vals = inc_df[colname].to_list()
    inc_dict = {}
    inc_dict[f'null_{colname}'] = 0
    if inc_cum_thresh < 1.0:
        inc_dict[f'rare_{colname}'] = 1
    for c in inc_vals:
        inc_dict[c] = len(inc_dict)
    return inc_dict

def create_ohe(df: pl.DataFrame, colname, vdict):
    ohe_mat = np.zeros((len(df), len(vdict)), dtype=np.int16)
    for idx, row in enumerate(df.iter_rows(named=True)):
        if row[colname] is None or len(row[colname]) == 0:
            ohe_mat[idx, 0] = 1
        else:
            for c in row[colname]:
                if c not in vdict:
                    ohe_mat[idx, 1] = 1
                else:
                    ohe_mat[idx, vdict[c]] = 1
    return ohe_mat

def create_le(df, colname, vdict):
    df = df.with_columns(
        pl.col(colname).map_elements(lambda x: vdict.get(x, 1) if x is not None else x, return_dtype=pl.UInt16).alias(f'{colname}_le')
    )
    return df

def category_mappers_static(df):
    binary_cols = [c for c in df.columns if 'has' in c.lower() or 'yn' in c.lower()]
    number_cols = [c for c in df.columns if 'number' in c.lower()] 

    
    eth_exp = [pl.when(pl.col('Ethnicity').is_in(['Declinded', '*Unspecified'])).\
        then(pl.lit("Unknown")).otherwise(pl.col('Ethnicity')).alias('Ethnicity')]
    # if df['Ethnicity'].isna().sum() > 0:
    #     df.loc[df['Ethnicity'].isna(), 'Ethnicity'] = 'Unknown'
    # if (df['Ethnicity'] == 'Declined').sum() > 0:
    #     df.loc[df['Ethnicity'] == 'Declined', 'Ethnicity'] = 'Unknown'

    # if (df['Ethnicity'] == '*Unspecified').sum() > 0:
    #     df.loc[df['Ethnicity'] == '*Unspecified', 'Ethnicity'] = 'Unknown'

    binary_col_expr = []
    for col in binary_cols:
        if df[col].dtype != pl.String: continue
        binary_col_expr.append(
            pl.when(pl.col(col).str.to_lowercase().is_in(["y",'yes'])).then(pl.lit(1)).\
                when(pl.col(col).str.to_lowercase().is_in(["n",'no'])).then(pl.lit(0)).\
                    otherwise(pl.lit(None)).cast(pl.UInt8).alias(f'{col}')
        )

    num_col_expr = []
    for col in number_cols:
        num_col_expr.append(
            pl.col(col).cast(pl.UInt16)
        )
    
    moa_expr = [pl.when(pl.col('Means_Of_Arrival').is_in(['NULL', 'Other'])).then(pl.lit(None)).\
        otherwise(pl.col('Means_Of_Arrival')).alias('Means_Of_Arrival')]
     
    # if df['Means_Of_Arrival'].isna().sum() > 0:
    #     df.loc[df['Means_Of_arrival']=='NULL', 'Means_Of_Arrival'] = 'Other'
    #     df.loc[df['Means_Of_Arrival'].isna(), 'Means_Of_Arrival'] = 'Other'
    

    # if df['Coverage_Financial_Class_Grouper'].isna().sum() > 0:
    #     df.loc[df['Coverage_Financial_Class_Grouper'].isna(), 'Coverage_Financial_Class_Grouper'] = 'None'

    # if df['Sex'].isna().sum() > 0:
    #     df.loc[df['Sex'].isna(), 'Sex'] = 'Unknown'
    
    # if df['Acuity_Level'].isna().sum() > 0:
    #     df.loc[df['Acuity_Level'].isna(), 'Acuity_Level'] = 'Unknown'
    
    # if df['Chief_Complaint_All'].isna().sum() > 0:
    #     df.loc[df['Chief_Complaint_All'].isna(), 'Chief_Complaint_All'] = 'Unknown'
    

    firstrace_exp = [pl.when(pl.col('FirstRace').is_in(['Unavailable/Unknown', 'Declined'])).then(pl.lit(None)).\
        otherwise(pl.col('FirstRace')).alias('FirstRace')]

    # if df['FirstRace'].isna().sum() > 0:
    #     df.loc[df['FirstRace'].isna(), 'FirstRace'] = 'Unknown'
    # if (df['FirstRace'] == 'Unavailable/Unknown').sum() > 0:
    #     df.loc[df['FirstRace'] == 'Unavailable/Unknown', 'FirstRace'] = 'Unknown'
    # if (df['FirstRace'] == 'Declined').sum() > 0:
    #     df.loc[df['FirstRace'] == 'Declined', 'FirstRace'] = 'Unknown'

    all_expr = eth_exp+binary_col_expr+num_col_expr+moa_expr+firstrace_exp
    df = df.with_columns(all_expr)
    
    return df

def create_sample_df(input_path, output_path, id_col, n_samples):
    df = pl.read_csv(input_path,
        infer_schema_length=int(20e6),
        null_values='NULL',
        dtypes=dict(Patient_Age=pl.Float64)
    )

    uids = df[id_col].unique()
    rids = np.random.choice(uids, n_samples, replace=False)
    df.filter(pl.col(id_col).is_in(rids))
    df.write_csv(output_path)
    return df

if __name__ == "__main__":
    # df = create_sample_df(constants.RAW_DATA, constants.RAW_DATA_SAMPLE, 'PAT_ENC_CSN_ID', 10000)
    # df = read_and_clean(constants.RAW_DATA_SAMPLE, infer_length=10000)
    df = read_and_clean(constants.RAW_DATA, infer_length=int(20e6))
    with open(constants.CLEAN_DATA, 'wb') as f:
        joblib.dump(df, f)
    
    # with open(constants.CLEAN_DATA, 'rb') as f:
    #     df = joblib.load(f)
    
    '''
    [] Data should be splitted into training and testing with a sliding training window
    [] Model should be split based on the number of events occurred so far

    '''
    
    

    # dx_dict = create_vocabs_for_cols(df, 'dx_list', 0.99, True)
    # cc_dict = create_vocabs_for_cols(df, 'cc_list', 0.99, True)
    # ev_dict = create_vocabs_for_cols(df, 'EVENT_NAME_NORM', 0.99, False)

    # # print(f'Time taken for sequential: {time.time()-t} seconds ...')

    # tv_dict = create_vocabs_for_dep_cols(df, colnames = ['Type_NORM', 'EVENT_NAME_NORM'],
    #                                      inc_cum_thresh=0.99, multival=False)  
    
    # dx_mat = create_ohe(df, 'dx_list', dx_dict)