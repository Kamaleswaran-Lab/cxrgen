from pathlib import Path
import numpy as np
import os
from multiprocessing import Pool, cpu_count

def process_encounter(encounter_path):
    ehr = np.load(encounter_path)
    encounter_id = encounter_path.stem.split('_')[0]
    prev_cxr = np.load(embedding_path / f"{encounter_id}_ffill_embeddings.npy")
    target = np.load(embedding_path / f"{encounter_id}_interpolated_embeddings.npy")

    if encounter_id in train_encounters:
        split = 'train'
    elif encounter_id in test_encounters:
        split = 'test'
    else:
        print(f"Skipping {encounter_id}")
        return
    
    mask = ehr[:, -1]
    for idx in range(len(ehr)):
        if mask[idx] == 0:
            continue
        ehr_row = ehr[idx, :]
        prev_cxr_row = prev_cxr[idx, :]
        target_row = target[idx, :]

        np.save(mlp_data_ehr / f"{encounter_id}_{idx}_{split}_ehr.npy", ehr_row)
        np.save(mlp_data_prev_cxr / f"{encounter_id}_{idx}_{split}_prev_cxr.npy", prev_cxr_row)
        np.save(mlp_data_target / f"{encounter_id}_{idx}_{split}_target.npy", target_row)
    
    print(f"Processed {encounter_id}")
    return None 


root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
dataset = root / 'longitudinal_data'
embedding_path = dataset / 'image_embeddings'
ehr_path = dataset / 'ehr_matrices'

train_encounters = np.load(root  / 'train_15_16_19_20_21.npy', allow_pickle=True)
test_encounters = np.load(root / 'test_17_18.npy', allow_pickle=True)

train_encounters = [x.stem.split('_')[0] for x in train_encounters]
test_encounters = [x.stem.split('_')[0] for x in test_encounters]

encounter_paths = list(ehr_path.glob("*.npy"))
prev_cxr_paths = list(embedding_path.glob("*_ffill_embeddings.npy"))
target_paths = list(embedding_path.glob("*_interpolated_embeddings.npy"))

mlp_data = dataset / 'mlp_data_masked'
os.makedirs(mlp_data, exist_ok=True)


mlp_data_ehr = mlp_data / 'ehr'
mlp_data_prev_cxr = mlp_data / 'prev_cxr'
mlp_data_target = mlp_data / 'target'
os.makedirs(mlp_data_ehr, exist_ok=True)
os.makedirs(mlp_data_prev_cxr, exist_ok=True)
os.makedirs(mlp_data_target, exist_ok=True)



with Pool(cpu_count()) as p:
    p.map(process_encounter, encounter_paths)

#for encounter_path in encounter_paths:
#    process_encounter(encounter_path)
#    break 

print("Done!")