import os
import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')
# SRC_PATH = os.getenv("EDStaticDynamic")
# sys.path.insert(1, SRC_PATH)
import polars as pl


# =============================================== DATA =======================================================
# RAW_DATA = '/work/InternalMedicine/s223850/raw_data/ED Events - 11.21.23.csv'
# RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events - 12.21.23.csv'
# RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 5.28.24.csv'
RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 6.6.24.csv'
CLEAN_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED_EVENTS_6624_clean.joblib'
CLEAN_DATA_PARQUET = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED_EVENTS_6624_clean.parquet'

# RAW_DATA = os.path.join(SRC_PATH, 'ED Events Last 2 Years - compiled 6.6.24.csv')
# CLEAN_DATA = os.path.join(SRC_PATH, 'ED_EVENTS_6624_clean.joblib')
# CLEAN_DATA_PARQUET = os.path.join(SRC_PATH, 'ED_EVENTS_6624_clean.parquet')

# RAW_DATA_SAMPLE = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 5.28.24_sample.csv'

WORKING_DIR = '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src'
OUTPUT_DIR = '/work/InternalMedicine/s223850/ED-StaticDynamic/'
OUTPUT_PROJ_DIR = '/project/InternalMedicine/Basit_lab/s223850/ED-StaticDynamic' 

# TARGET = ['Admitted_YN'] Before 12_1_23
TARGET = ['ED_Disposition']

# # Fields are calculated from 00_EDA.ipynb
# STATIONARY_FIELDS = ['PAT_ENC_CSN_ID','PAT_MRN_ID', 'PAT_ID', 'Ethnicity', 'Sex', 'MultiRacial', 'Patient_Age', 
#                      'Coverage_Financial_Class_Grouper', 'Has Completed Appt in Last Seven Days', 'Has Hospital Encounter in Last Seven Days',
#                      'Number of Inpatient Admissions in the last 30 Days', 'Number of past appointments in last 60 days',
#                      'Number of past inpatient admissions over ED visits in last three years', 'Chief_Complaint_All', 'Count_of_Chief_Complaints', 'Means_Of_Arrival',
#                     'Acuity_Level', 'FirstRace',
#                     'Arrived_Time']
#                     # 'Arrived_Time_appx'] # TODO: Substitute this line by the top line once the new data arrive
# DYNAMIC_FIELDS = ['PAT_ENC_CSN_ID', 'Type', 'EVENT_NAME', 'Order_Status', 'Result_Flag',
#                   'Primary_DX_Name', 'Primary_DX_ICD10', 'Calculated_DateTime']
# # DROPPED_FIELDS = ['PAT_MRN_ID', 'PAT_ID'] Before 12_1_23
# DROPPED_FIELDS = ['PAT_MRN_ID', 'PAT_ID', 'Admitted_YN']

id_cols = ["PAT_ENC_CSN_ID",
        #    "PAT_MRN_ID",
        #    "PAT_ID"
        ]
target_col = ["has_admit_order"]

static_cols = [
        "Ethnicity", "FirstRace", "Sex", "Acuity_Level", "Means_Of_Arrival",
        "cc_list", "Coverage_Financial_Class_Grouper", "Procedure in the Last 4 Weeks",
        "Has Completed Appt in Last Seven Days",
        "Has Hospital Encounter in Last Seven Days", "MultiRacial",
        "Patient_Age", "Dispo_Prov_Admission_Rate",
        "Number of Inpatient Admissions in the last 30 Days",
        "Number of past appointments in last 60 days",
        "Number of past inpatient admissions over ED visits in last three years",
        "ProblemList_Sixty_Admission_YN", "ProblemList_Eighty_Admission_YN",
        'arr_year', 'arr_month','arr_day', 'arr_dow', 'arr_hour', 'holiday' ,"Arrived_Time", 
]

dynamic_singleval_col =[
    "Type_NORM",
    "EVENT_NAME_NORM"
]

dynamic_multival_col =[
    "dxcode_list"
]

dynamic_num_cols = [
   'elapsed_time_min',
   'event_idx',
   'MEAS_VALUE_Temp',
   'MEAS_VALUE_Weight',
   'MEAS_VALUE_BP (MAP)',
   'MEAS_VALUE_(0-10) Pain Rating: Activity',
   'MEAS_VALUE_BMI (Calculated)',
   'MEAS_VALUE_BSA (Dubois Calc)',
   'MEAS_VALUE_(0-10) Pain Rating: Rest',
]

dynamic_num_norm_method = {
   'elapsed_time_min':'std',
   'event_idx':'std',
   'MEAS_VALUE_Temp':'std',
   'MEAS_VALUE_Weight':'std',
   'MEAS_VALUE_BP (MAP)':'std',
   'MEAS_VALUE_(0-10) Pain Rating: Activity':'minmax',
   'MEAS_VALUE_BMI (Calculated)':'std',
   'MEAS_VALUE_BSA (Dubois Calc)':'std',
   'MEAS_VALUE_(0-10) Pain Rating: Rest':'minmax'
    
}

dynamic_cols = [
    "Type_NORM",
    "EVENT_NAME_NORM",
    "elapsed_time_min",
    "event_idx",
    "ED_Location_YN",
    "dxcode_list",
    'MEAS_VALUE_Temp',
    'MEAS_VALUE_Weight',
    'MEAS_VALUE_BP (MAP)',
    'MEAS_VALUE_(0-10) Pain Rating: Activity',
    'MEAS_VALUE_BMI (Calculated)',
    'MEAS_VALUE_BSA (Dubois Calc)',
    'MEAS_VALUE_(0-10) Pain Rating: Rest',
    'MEAS_VALUE_BP',
    'MEAS_VALUE_SpO2',
    'MEAS_VALUE_Resp',
    'MEAS_VALUE_Pulse',
    'Order_Status_Completed',
    'Order_Status_Sent',
    'Order_Status_Canceled',
    'Result_Flag_Resulted',
    'Result_Flag_Low',
    'Result_Flag_Canceled',
    'Result_Flag_Low Panic',
    'Result_Flag_Ordered',
    'Result_Flag_Normal',
    'Result_Flag_High Panic',
    'Result_Flag_Abnormal',
    'Result_Flag_High'
]

# ==================================================== Clean =======================================================
spo2 = [67, 100]
temp = [96, 106]
pulse = [45, 180]
BMI = [10, 500]
Resp = [1, 100]

vital_ranges_dict = {
    'SpO2': spo2,
    'BMI (Calculated)': BMI,
    'Resp': Resp,
    'Temp':temp,
    'Pulse': pulse
}

# =============================================== Preprocessing =======================================================
NULL_LIST = [
    # pl.Null,
    None,
    'none',
    'null', 
    'unknown',
    'undefined',
    '*unspecified',
    'unspecified'
]

static_singleval_cat_cols = [
        "Ethnicity", "FirstRace", "Sex", "Acuity_Level", "Means_Of_Arrival",
        "Coverage_Financial_Class_Grouper",
        'arr_month','arr_day','arr_hour', 'holiday', 'arr_dow'
]

static_multval_cat_cols = [
    'cc_list'
]

static_num_cols = [
    'arr_year',
    "Dispo_Prov_Admission_Rate", 
    "Number of Inpatient Admissions in the last 30 Days",
    "Number of past appointments in last 60 days",
    "Number of past inpatient admissions over ED visits in last three years"
]
static_num_norm_method = {
    'arr_year': 'std',
    'Dispo_Prov_Admission_Rate': 'std',
    "Number of Inpatient Admissions in the last 30 Days": 'std',
    "Number of past appointments in last 60 days": 'std',
    "Number of past inpatient admissions over ED visits in last three years": 'minmax'
}
# =============================================== ML =======================================================
OUTPUT_DIR = '/work/InternalMedicine/s223850/ED-StaticDynamic/ml_results'
FS_OUTPUT_DIR = '/work/InternalMedicine/s223850/ED-StaticDynamic/fs_results'

# TRAINING_PERIOD = 8
# TESTING_PERIOD  = 4 

# DS_DATA_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_ds_parallel")
# ML_DATA_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_feats")
# ML_DATA_OUTPUT_ID = os.path.join(OUTPUT_DIR, "static_dynamic_feats_ID_240324")

# DL_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_dl_output")
# DL_FEATS_DIR =  os.path.join(OUTPUT_DIR, 'static_dynamic_dl_feats')
# ML_RESULTS_OUTPUT = os.path.join(OUTPUT_PROJ_DIR, "ml_results_240324")
#===========================================================================================================