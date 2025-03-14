import numpy as np
from pathlib import Path
import os
import pandas as pd
import pickle
import multiprocessing as mp
import warnings
#Ignore FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

imputation_strategy = {
    #Vitals [default_impute, impute_method, min_bound_normal, max_bound_normal]
    'temperature' : [37, 'FeedForward', 36, 38],  
    'daily_weight_kg': [None, 'FeedForward', 60, 90],
    'sbp_selected': [90, 'FeedForward', 90, 130], 
    'dbp_selected': [60, 'FeedForward', 65, 75],
    'pulse': [75, 'FeedForward', 60, 90], 
    'unassisted_resp_rate': [17, 'FeedForward', 10, 24],
    'spo2': [98, 'FeedForward', 95, 100], 
    'end_tidal_co2': [None, None, 35, 45],
    'Oxygen_Flow_Rate': [None, 'FeedForward6', 0, 70, 0, 70],  
    'best_map': [70, 'FeedForward', 65, 75],
    'pf_sp': [None, 'FeedForward', 300, 500], 
    'pf_pa': [None, 'FeedForward', 400, 500],

    
    #Labs
    'anion_gap' : [None, 'FeedForward', 8, 12], 
    'base_excess': [None, 'FeedForward', -2, 2],
    'bicarb_(hco3)': [None, 'FeedForward', 22, 27], 
    'blood_urea_nitrogen_(bun)': [None, 'FeedForward', 6, 20],
    'calcium' : [None, 'FeedForward', 8.5, 10.5], #'calcium_adjusted',
    'calcium_ionized': [None, 'FeedForward', 1, 1.3], 
    'chloride': [None, 'FeedForward', 96, 106],
    'creatinine': [None, 'FeedForward', 0.5, 1.3],
    'glucose': [None, 'FeedForward', 60, 200],
    'magnesium': [None, 'FeedForward', 1.5, 2.5],
    'phosphorus': [None, 'FeedForward', 2.5, 4.5],
    'potassium': [None, 'FeedForward', 3.5, 4.5],
    'sodium': [None, 'FeedForward', 135, 145],
    'hematocrit': [None, 'FeedForward', 35, 45],
    'hemoglobin': [None, 'FeedForward', 12, 17],         
    'platelets': [None, 'FeedForward', 150, 450],
    'white_blood_cell_count': [None, 'FeedForward', 4, 11],
    'alanine_aminotransferase_(alt)': [None, 'FeedForward', 4, 36],
    'albumin': [None, 'FeedForward', 3.4, 5.4],
    'alkaline_phosphatase': [None, 'FeedForward', 44, 147],
    'ammonia': [None, 'FeedForward', 15, 45],
    'aspartate_aminotransferase_(ast)': [None, 'FeedForward', 8, 33],
    'bilirubin_direct': [None, 'FeedForward', 0.1, 0.4],
    'bilirubin_total': [None, 'FeedForward', 0.2, 1.2],
    'fibrinogen': [None, 'FeedForward', 200, 400],
    'inr': [None, 'FeedForward', 0.8, 1.3],
    'lactate_dehydrogenase': [None, 'FeedForward', 105, 350],
    'partial_prothrombin_time_(ptt)': [None, 'FeedForward', 25, 35],
    'protein': [None, 'FeedForward', 6, 8.3],
    'lipase': [None, 'FeedForward', 1, 160],
    'troponin': [None, 'FeedForward', 0, 0.03],
    'fio2': [None, 'FeedForward', 21, 40],
    #'partial_pressure_of_carbon_dioxide_(paco2)',
    #'partial_pressure_of_oxygen_(pao2)', Removing these two because correlated with pf_pa, pf_sp
    'ph': [None, 'FeedForward', 7.35, 7.45],
    'saturation_of_oxygen_(sao2)': [None, 'FeedForward', 95, 100],  
    'n_to_l' : [None, 'FeedForward', 0, 100],
    
    'gcs_total_score': [15, 'FeedForward', 1, 15],

    #Vasopressors -- how do we determine duraation of adminsitration
    #'norepinephrine': ,
    #'epinephrine',
    #'dobutamine',
    #'dopamine',
    #'phenylephrine',
    #'vasopressin',

    #Other data  
    'covid':[0, 0, 0, 1],  #Removing vent_status because same as on_vent
    'icu':[0, 0, 0, 1], #Flag
    'procedure': [0, 0, 0, 1], #Flag 
    'on_pressors':[0, 0, 0, 1],
    'on_dobutamine': [0, 0, 0, 1],
    'on_dialysis': [0, 0, 0, 1],
    'elapsed_icu': [0, None, None, None],
    'elapsed_hosp': [0, None, None, None],
    'on_vent': [0, 0, 0, 1],
    'age': [None, None, 18, 100],
    'gender': [None, None, 'Onehot'],
    'race': [None, None, 'Onehot'],
    ##'ethnicity': [None, None, 'Onehot'],
    'cci9': [None, None, 0, 100],
    'cci10': [None, None, 0, 100],
    'infection': [0,0, 0, 1],
    'sepsis': [0,0, 0, 1],
    
    'bed_type': [None, None, 'Onehot'], #Ward/ICU string
    'icu_type': [None, None, 'Onehot'], #ICU type string
    

    ##'vent_mode': [None, 'FeedForwardVO', 'Onehot'],
    'vent_rate_set': [None, 'FeedforwardVO',0, 100, 0, 600],
    'vent_tidal_rate_exhaled': [None, 'FeedForwardVO', 0, 500],
    'peep': [None, 'FeedForwardVO', 5, 15], #'vent_fio2' is important
    'vent_fio2': [None, 'FeedForwardVO', 21, 100],
    'cxr_timing': [None, 'FeedForward', None],
   # 'cxr_timing_approx_flag'
    
}



def bp_selector(row):
    """
    Selects "line" or "cuff" measurements for blood pressure.  
    """
    if row[['sbp_line','dbp_line']].notnull().all() and (row['sbp_line'] - row['dbp_line']) > 15:
        return row['sbp_line'], row['dbp_line']
    elif row[['sbp_cuff','dbp_cuff']].notnull().all() and (row['sbp_cuff'] - row['dbp_cuff']) > 15 :
        return row['sbp_cuff'], row['dbp_cuff']
    else:
        return np.nan, np.nan 
    

def process_icu_type(df):
    icu_types_in_df = df.icu_type.unique()
    
    if 'sicu BEFORE 1/18/2018; cticu ON OR AFTER 1/18/2018' in icu_types_in_df:
        indices = df.loc[df.icu_type == 'sicu BEFORE 1/18/2018; cticu ON OR AFTER 1/18/2018'].index
        if indices[0] > pd.to_datetime('1/18/2018'): 
            df.loc[indices, 'icu_type'] = 'cticu'
        else:
            df.loc[indices, 'icu_type'] = 'sicu'
    if 'cticu BEFORE 1/18/2018; micu ON OR AFTER 1/18/2018' in icu_types_in_df:
        indices = df.loc[df.icu_type == 'cticu BEFORE 1/18/2018; micu ON OR AFTER 1/18/2018'].index 
        if indices[0] > pd.to_datetime('1/18/2018'): 
            df.loc[indices, 'icu_type'] = 'micu'
        else:
            df.loc[indices, 'icu_type'] = 'cticu'
    
    if 'sicu BEFORE 1/18/2018' in icu_types_in_df:
        indices = df.loc[df.icu_type == 'sicu BEFORE 1/18/2018'].index 
        if indices[0] > pd.to_datetime('1/18/2018'): 
            df.loc[indices, 'icu_type'] = np.nan
        else:
            df.loc[indices, 'icu_type'] = 'sicu'
    return df['icu_type'].values

def process_race(df):
    if df.race.values[0] in ['Multiple', 'Unknown, Unavailable or Unreported']:
        return [np.nan]*len(df)

def process_bedtype(df):
    other_indices = df.loc[df.bed_type == 'other'].index
    if len(other_indices) > 0:
        df.loc[other_indices, 'bed_type'] = np.nan
    return df['bed_type'].values


def clean_impute_dataframe(df, imputation_strategy, stats_dict=None):
    """
    Process dataframes according to the imputation strategy.
    
    Args:
        df: pandas DataFrame to process
        imputation_strategy: dictionary containing processing rules
        stats_dict: dictionary containing stats for one-hot encoding (optional)
    """
    
    def onehot_encode(df, column_name, final_stats):
        nan_mask = df[column_name].isna()
        df[column_name] = pd.Categorical(df[column_name], categories = list(final_stats[column_name].keys()))
        dummies = pd.get_dummies(df[column_name], prefix=column_name, columns=list(final_stats[column_name].keys()))
        if nan_mask.any():
            dummies.loc[nan_mask] = 0
        return pd.concat([df, dummies], axis=1)

    # First, select only columns that are in imputation_strategy
    columns_to_process = [col for col in df.columns if col in imputation_strategy]
    df_processed = df[columns_to_process].copy()
    
    # First pass: Handle initial missing values and value bounds
    for col in columns_to_process:
        strategy = imputation_strategy[col]
        # Handle 5th and 6th elements if they exist (value bounds)
        if len(strategy) >= 6:
            min_bound, max_bound = strategy[4], strategy[5]
            mask = (df_processed[col] < min_bound) | (df_processed[col] > max_bound)
            df_processed.loc[mask, col] = np.nan
        
        # Handle initial missing values
        initial_value = strategy[0]
        if initial_value is not None:
            # If all values are missing, fill entire column
            if df_processed[col].isna().all():
                df_processed[col] = initial_value
            else:
                # Find first non-null value and fill everything before it
                first_valid = df_processed[col].first_valid_index()
                if first_valid is not None:
                    df_processed.loc[:first_valid, col] = initial_value
    
    # Second pass: Handle imputation strategies
    for col in columns_to_process:
        strategy = imputation_strategy[col]
        impute_method = strategy[1]
        
        if impute_method == 'FeedForward':
            df_processed[col] = df_processed[col].ffill()
        
        elif impute_method == 'FeedForward24':
            df_processed[col] = df_processed[col].ffill(limit=24)
        
        elif impute_method == 'FeedForward6':
            df_processed[col] = df_processed[col].ffill(limit=6)
        
        elif impute_method == 'FeedForwardVO':
            temp_series = df_processed[col].copy()
    
            # Get indices where we have valid values
            valid_indices = temp_series.dropna().index

            for valid_idx in valid_indices:
                # Only process if we're in a vent-on period
                if df_processed.loc[valid_idx, 'on_vent'] == 1:
                    # Find next vent off or next valid value after this point
                    next_indices = df_processed.index[df_processed.index > valid_idx]
                    if len(next_indices) > 0:
                        next_off = next_indices[
                            (df_processed.loc[next_indices, 'on_vent'] == 0) |
                            (~temp_series.loc[next_indices].isna())
                        ].min() if len(next_indices) > 0 else len(df_processed)

                        # Forward fill from this valid value until next off/valid
                        temp_series.loc[valid_idx:next_off] = temp_series.loc[valid_idx:next_off].ffill()

            df_processed[col] = temp_series
        
        elif impute_method == 0:
            df_processed[col] = df_processed[col].fillna(0)
    
    # Third pass: Handle normalization and one-hot encoding
    for col in columns_to_process:
        strategy = imputation_strategy[col]
        
        if strategy[2] == 'Onehot':
            if stats_dict is not None:
                df_processed = onehot_encode(df_processed, col, stats_dict)
        elif strategy[2] is not None and strategy[3] is not None:
            # Min-max normalization
            min_val, max_val = strategy[2], strategy[3]
            try:
                df_processed[col] = (df_processed[col] - min_val) / (max_val - min_val)
            except:
                print(col, min_val, max_val)
    return df_processed

def process_df(file_name):
    try:
        df = pd.read_pickle(file_name)
        df['sbp_selected'], df['dbp_selected'] = zip(*df.apply(bp_selector, axis=1))
        df['icu_type'] = process_icu_type(df)
        df['race'] = process_race(df)
        df['bed_type'] = process_bedtype(df)
        df['vent_rate_set'] = pd.to_numeric(df['vent_rate_set'], errors = 'coerce')
        df['Oxygen_Flow_Rate'] = pd.to_numeric(df['Oxygen_Flow_Rate'], errors = 'coerce')
        df['vent_tidal_rate_exhaled'] = pd.to_numeric(df['vent_tidal_rate_exhaled'], errors = 'coerce')
        df['peep'] = pd.to_numeric(df['peep'], errors = 'coerce')
        df_processed = clean_impute_dataframe(df, imputation_strategy, stats)
        df_processed.drop(columns = ['icu_type_sicu BEFORE 1/18/2018; cticu ON OR AFTER 1/18/2018',
                                'icu_type_sicu BEFORE 1/18/2018', 'icu_type_cticu BEFORE 1/18/2018; micu ON OR AFTER 1/18/2018', 
                                'bed_type_other', 'race_Unknown, Unavailable or Unreported',
                                'race_Multiple', 'bed_type', 'icu_type', 'gender', 'race' ], inplace = True)
        df_processed['cxr_timing'] = df['cxr_timing']
        df_processed['cxr_timing_approx_flag'] = df['cxr_timing_approx_flag']    
        df_processed['encounter_id'] = [file_name.stem.split('_')[0]]*len(df_processed)
        
        object_columns = [column for column in df_processed.columns if not pd.api.types.is_numeric_dtype(df[column])]
        object_columns = [column for column in object_columns if column not in ['cxr_timing', 'encounter_id']]

        for column in object_columns:
            df_processed[column] = df_processed[column].astype('float64')

        new_file_name = str(file_name).replace('.pickle', '_processed.pickle')
        df_processed.to_pickle(new_file_name)
        print("Processed", file_name, "to", new_file_name)
    except Exception as e:
        print("Error in : ", file_name)
    




root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
embedding_path = root / 'BioMedCLIP_embeddings'
supertable_path = root / 'matched_supertables_with_images'
supertable_template = "_timing_corrected.pickle"

supertables = list(supertable_path.glob("*" + supertable_template))


with open(root / 'supertable_stats.pickle', 'rb') as f:
    stats = pickle.load(f)

#for supertable in supertables[:5]:
#    process_df(supertable)

cpu_count = mp.cpu_count()
    
with mp.Pool(cpu_count) as pool:
    pool.map(process_df, supertables)
