import os
import sys
sys.path.insert(1, '/home2/s223850/ED/UTSW_ED_EVENTBASED_staticDynamic/src')

import pandas as pd

class CustomCrossValidator1:
    def __init__(self, training_period, testing_period, date_col, step=1):
        self.tr_per = training_period
        self.te_per = testing_period
        self.date_col = date_col
        self.step=step

    def split2(self, X, silent=True):
        Xc = X.sort_values(self.date_col)
        
        start_date = Xc[self.date_col].iloc[0]
        end_date = Xc[self.date_col].iloc[-1]
        total_months = (end_date-start_date).total_seconds()/(60*60*24*30)

        if not silent: 
            print(f'Total Duration is {total_months} months ...')
        current_month = start_date-pd.Timedelta(days=30*self.step)

        while current_month+pd.Timedelta(days=30*self.tr_per)+pd.Timedelta(days=30*self.te_per) <= end_date:
            current_month = current_month + pd.Timedelta(days=30*self.step)
            ed_tr = current_month+pd.Timedelta(days=30*self.tr_per)
            ed_te = ed_tr+pd.Timedelta(days=30*self.te_per)

            Xtrain = Xc[(Xc[self.date_col]>=current_month)&(Xc[self.date_col]<=ed_tr)]
            Xtest = Xc[(Xc[self.date_col]>ed_tr)&(Xc[self.date_col]<=ed_te)]
            
            yield Xtrain, Xtest
            

    def split(self, X, silent=True):
        Xc = X.sort_values(self.date_col)
        if 'arr_mnth' not in Xc.columns:
            Xc['arr_mnth'] = Xc[self.date_col].dt.month
        if 'arr_year' not in Xc.columns:
            Xc['arr_year'] = Xc[self.date_col].dt.year
        
        start_date = Xc[self.date_col].iloc[0]
        end_date = Xc[self.date_col].iloc[-1]
        total_months = (end_date-start_date).total_seconds()/(60*60*24*30)
        if not silent: 
            print(f'Total Duration is {total_months} months ...')
        
        group_training = []
        group_testing = []
        for idx, df_grp in Xc.groupby(['arr_year', 'arr_mnth']):
            if len(group_training) < self.tr_per:
                if not silent:
                    print(f'stacking {idx} to training group ...')
                group_training.append(df_grp)
            elif len(group_testing) < self.te_per:
                if not silent:
                    print(f'stacking {idx} to testing group ...')
                group_testing.append(df_grp)
            else:
                df_train_fold = pd.concat(group_training)
                df_test_fold = pd.concat(group_testing)
                if not silent:
                    print(f"Between {group_training[0]['Arrived_Time_appx'].min()} to {group_training[-1]['Arrived_Time_appx'].max()}, the training data contains {df_train_fold['PAT_ENC_CSN_ID'].nunique()} encounters with total of {len(df_train_fold)} rows ...")
                    print(f"Between {group_testing[0]['Arrived_Time_appx'].min()} to {group_testing[-1]['Arrived_Time_appx'].max()}, the testing data contains {df_test_fold['PAT_ENC_CSN_ID'].nunique()} encounters with total of {len(df_test_fold)} rows ...")
                    print('-----------------------------')
                group_training = []
                group_testing = [] 
                yield df_train_fold, df_test_fold


# Driving code
if __name__ == "__main__":
    ROOT_PATH = '/work/InternalMedicine/s223850/ED-StaticDynamic/static_dynamic_ds_parallel'
    file_2_s = 'static_100.csv'
    file_2_d = 'dynamic_100.csv'
    df_static = pd.read_csv(os.path.join(ROOT_PATH, file_2_s))
    df_static = df_static[df_static['Arrived_Time_appx']!='-1']
    df_static['Arrived_Time_appx'] = pd.to_datetime(df_static['Arrived_Time_appx'])

    df_dynamic = pd.read_csv(os.path.join(ROOT_PATH, file_2_d))
    df_dynamic['Calculated_DateTime'] = pd.to_datetime(df_dynamic['Calculated_DateTime'])

    df_static = df_static[df_static['Arrived_Time_appx'].dt.year>=2021]
    df_dynamic = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(df_static['PAT_ENC_CSN_ID'].unique())]

    cc = CustomCrossValidator1(6, 2, 'Arrived_Time_appx')
    # for df_train_static, df_test_static in cc.split(df_static):
    for df_train_static, df_test_static in cc.split2(df_static, True):
        tr_enc = df_train_static['PAT_ENC_CSN_ID'].unique()
        te_enc = df_test_static['PAT_ENC_CSN_ID'].unique()
        df_train_dynamic = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(tr_enc)]
        df_test_dynamic = df_dynamic[df_dynamic['PAT_ENC_CSN_ID'].isin(te_enc)]
        x = 0