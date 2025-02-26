import numpy as np
from pathlib import Path
import os
import pandas as pd
import pickle
import multiprocessing as mp
import warnings


def process_df(supertable_path):
    try:
        df = pd.read_pickle(supertable_path)
        df['cxr_trajectory_endpoints'] = [0]*len(df)
        df.loc[df['cxr_timing'].first_valid_index(): df['cxr_timing'].last_valid_index(), 'cxr_trajectory_endpoints'] = 1
        df['cxr_timing_ffill'] = df['cxr_timing'].ffill()
        
        recorded_cxr_index = np.where(df.cxr_timing.values != None)[0]
        interpolated_embeddings = np.zeros((len(df), 512))
        ffill_embeddings = np.zeros((len(df), 512))
        
        for idx in recorded_cxr_index:
            img_embedding_path = embedding_path / (df.iloc[idx]['cxr_timing'] + '.npy')
            if img_embedding_path.exists():
                embedding = np.load(img_embedding_path)
                interpolated_embeddings[idx, :] = embedding
                ffill_embeddings[idx, :] = embedding
            else:
                print("Error for ", supertable_path, " at ", idx)
                print("No such file: ", img_embedding_path)
                return None
        

        for i in range(len(recorded_cxr_index)-1):
            start_idx = recorded_cxr_index[i]
            end_idx = recorded_cxr_index[i+1]
            v1 = interpolated_embeddings[start_idx]
            v2 = interpolated_embeddings[end_idx]
            n_steps = end_idx - start_idx
            weights = np.arange(1, n_steps) / n_steps
            interpolated_embeddings[start_idx+1:end_idx] = v1 + weights.reshape(-1, 1) * (v2 - v1)

            # Fill in the ffill embeddings
            ffill_embeddings[start_idx+1:end_idx] = v1
        
        # Fill in the last part
        start_idx = recorded_cxr_index[-1]
        end_idx = len(df)
        ffill_embeddings[start_idx+1:end_idx] = interpolated_embeddings[start_idx]
        
        np.save(embedding_path / (str(supertable_path.stem).split('_')[0] + "_interpolated_embeddings.npy"), interpolated_embeddings)
        np.save(embedding_path / (str(supertable_path.stem).split('_')[0] + "_ffill_embeddings.npy"), ffill_embeddings)
        
        new_supertable_path = supertable_path.parent / (str(supertable_path.stem).split('_')[0] + "_image_interpolated.pickle")
        df.to_pickle(new_supertable_path)
        print("Processed ", supertable_path)
    except Exception as e:
        print("Error for ", supertable_path)
        print(e)
    
    return None 


def get_ehr_matrix(file):
    df = pd.read_pickle(file)
    df.drop(columns = [ 'cxr_timing',
                        'cxr_timing_approx_flag',
                        'encounter_id', 'cxr_timing_ffill'], inplace = True)
    #Convert all columns to float 
    for col in df.columns:
        df[col] = df[col].astype(float)
    df_values = df.values 
    np.save(file.with_suffix('.npy'), df_values)
    print(f"Saved {file.with_suffix('.npy')}")
    return None



root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
embedding_path = root / 'BioMedCLIP_embeddings'
supertable_path = root / 'matched_supertables_with_images'
supertable_template = "_image_interpolated.pickle"

supertables = list(supertable_path.glob("*" + supertable_template))
print(f"Found {len(supertables)} supertables")

cpu_count = mp.cpu_count()

with mp.Pool(cpu_count) as pool:
    pool.map(get_ehr_matrix, supertables)

print("Done!")