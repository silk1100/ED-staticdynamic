import os
import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

# =============================================== DATA =======================================================
# RAW_DATA = '/work/InternalMedicine/s223850/raw_data/ED Events - 11.21.23.csv'
# RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events - 12.21.23.csv'
# RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 5.28.24.csv'
RAW_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 6.6.24.csv'
CLEAN_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED_EVENTS_6624_clean.joblib'

RAW_DATA_SAMPLE = '/work/InternalMedicine/s223850/ED-StaticDynamic/raw_data/ED Events Last 2 Years - compiled 5.28.24_sample.csv'

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

id_cols = ["PAT_ENC_CSN_ID", "PAT_MRN_ID", "PAT_ID"]
target_col = ["has_admit_order"]
static_cols = ["Ethnicity", "FirstRace", "Sex", "Acuity_Level", "Means_Of_Arrival",
               "Chief_Complaint_All", "Coverage_Financial_Class_Grouper", "Procedure in the Last 4 Weeks",
               "Has Completed Appt in Last Seven Days", "Has Hospital Encounter in Last Seven Days", "MultiRacial",
              "Patient_Age", "Dispo_Prov_Admission_Rate", "Number of Inpatient Admissions in the last 30",
              "Number of past appointments in last 60 days", "Number of past inpatient admissions over ED visits", "Arrived_Time",
               "ProblemList_Sixty_Admission_YN", "ProblemList_Eighty_Admission_YN",
               'arr_year', 'arr_month','arr_day','arr_hour', 'holiday']

static_clean_cols = ["Ethnicity", "FirstRace", "Sex", "Acuity_Level", "Means_Of_Arrival",
               "Chief_Complaint_All", "Coverage_Financial_Class_Grouper", "Procedure in the Last 4 Weeks",
               "Has Completed Appt in Last Seven Days", "Has Hospital Encounter in Last Seven Days", "MultiRacial",
              "Patient_Age", "Dispo_Prov_Admission_Rate", "Number of Inpatient Admissions in the last 30",
              "Number of past appointments in last 60 days", "Number of past inpatient admissions over ED visits", "Arrived_Time",
               "ProblemList_Sixty_Admission_YN", "ProblemList_Eighty_Admission_YN",
               'arr_year', 'arr_month','arr_day','arr_hour', 'holiday']

dynamic_cols = [
    "Type",
    "EVENT_NAME",
    "MEAS_VALUE",
    "elapsed_time_min",
    "Order_Status",
    "ED_Location_YN",
    "Primary_DX_ICD10",
    "Result_Flag"
]

dynamic_clean_cols = [
    'type_name',
    "elapsed_time_min",
    "ED_Location_YN",
    "Result_Flag"
    'MEAS_VALUE_UP',
    'MEAS_VALUE_Weight',
    'MEAS_VALUE_Temp',
    'MEAS_VALUE_BP',
    'MEAS_VALUE_SpO2',
    'MEAS_VALUE_BSA (Dubois Calc)',
    'MEAS_VALUE_(0-10) Pain Rating: Activity',
    'MEAS_VALUE_BP (MAP)',
    'MEAS_VALUE_BMI (Calculated)',
    'MEAS_VALUE_(0-10) Pain Rating: Rest',
    'MEAS_VALUE_Resp',
    'MEAS_VALUE_Pulse'
]

# ==================================================== Clean =======================================================
spo2 = [67, 100]
temp = [60, 115]
pulse = [27, 600]
BMI = [1, 400]
Resp = [1, 100]

vital_ranges_dict = {
    'SpO2': spo2,
    'BMI (Calculated)': BMI,
    'Resp': Resp,
    'Temp':temp,
    'Pulse': pulse
}

# =============================================== Preprocessing =======================================================
CLEAN_DATA_DIR = '/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target'
# CLEAN_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target/df_clean.csv'
# CLEAN_DATA = '/work/InternalMedicine/s223850/ED-StaticDynamic/clean_target/df_clean.csv'
# MISSING_VALS = { # Obselete (replaced with utils.category_mapper_static, and utils.category_mapper_dynamic)
#     'Means_Of_Arrival': 'Car', 
#     'Ethnicity': 'Non-Hispanic/Latino', # Contains (Declined, Unknown, *Unspecified) Need to be grouped in one group,
#     'Coverage_Financial_Class_Grouper': 'None',
# }
# DIFFERENT_ENCOUNTERS = [651902046, 659362974, 671952725, 671952746, 675563577, 681239896, 
#                         681828883, 682315360, 683950976,
#                         686439858, 692452984] Before 12_1_23
                
# =============================================== ML =======================================================
TRAINING_PERIOD = 8
TESTING_PERIOD  = 4 
DS_DATA_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_ds_parallel")
ML_DATA_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_feats")
ML_DATA_OUTPUT_ID = os.path.join(OUTPUT_DIR, "static_dynamic_feats_ID_240324")

DL_OUTPUT = os.path.join(OUTPUT_DIR, "static_dynamic_dl_output")
DL_FEATS_DIR =  os.path.join(OUTPUT_DIR, 'static_dynamic_dl_feats')
ML_RESULTS_OUTPUT = os.path.join(OUTPUT_PROJ_DIR, "ml_results_240324")