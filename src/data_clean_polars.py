import os
import sys

sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import polars as pl
from const import constants

def read_and_parse_csv(csv_file):
    df = pl.read_csv(csv_file,
                 infer_schema_length=int(5e6),
                 null_values=dict(Patient_Age='NULL', MEAS_VALUE='NULL'),
                 dtypes=dict(Patient_Age=pl.Float64)
        ) 


    df = df.filter((pl.col('ED_Disposition')=='Admitted')|(pl.col('ED_Disposition')=='Discharged'))
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
    df = df.with_columns([
        (pl.col('Calculated_DateTime')-pl.col('Arrived_Time')).dt.total_minutes().alias('elapsed_time_min')
    ])
    df = df.with_columns((pl.col('elapsed_time_min').shift(-1)-pl.col('elapsed_time_min')).over('PAT_ENC_CSN_ID').alias('processing_time_min'))
    df = df.filter(pl.col('ED_Disposition').is_in(['Admitted', 'Discharged']))

    # Create new target column
    df = df.with_columns(
        (pl.col("Type") == 'Order - Admission').sum().over('PAT_ENC_CSN_ID').alias('admit-order-cnt')
    ).with_columns(
        pl.when(pl.col('admit-order-cnt')==0).
        then(0).otherwise(1).alias('has_admit_order')
    ).drop(columns=['admit-order-cnt'])

    df_trim = df.with_columns(
        (pl.col('Type')=='Order - Admission').cumsum().over('PAT_ENC_CSN_ID').alias('n_admit_orders'),
        (pl.col('Type')=='Order - Discharge').cumsum().over('PAT_ENC_CSN_ID').alias('n_disch_orders'),
    ).filter( (pl.col('n_admit_orders')<1) & (pl.col('n_disch_orders')<1) )


    return df_trim


if __name__ == "__main__":
    df = read_and_parse_csv(constants.RAW_DATA)
    
    
    
    # 840 subjects (0.8% of the total patients)
    # neg_pat_id = df.filter(pl.col('first_event_elapsed_time')<0)['PAT_ENC_CSN_ID'].unique() 
    # df = df.filter(~pl.col("PAT_ENC_CSN_ID").is_in(neg_pat_id))


    x = 0