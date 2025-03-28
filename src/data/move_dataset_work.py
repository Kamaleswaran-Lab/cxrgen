from pathlib import Path
import numpy as np
import shutil
import tqdm
from multiprocessing import Pool, cpu_count

def move_ehr_matrix(supertable):
    ehr_path = ehr_dir / (supertable.stem + '.npy')
    image_path_ffill = image_dir / (supertable.stem.split('_')[0] + '_ffill_embeddings.npy')
    image_path_interpolated = image_dir / (supertable.stem.split('_')[0] + '_interpolated_embeddings.npy')

    shutil.copy(supertable, ehr_dir / ehr_path.name)
    shutil.copy(embedding_path / (supertable.stem.split('_')[0] + '_ffill_embeddings.npy') , image_path_ffill)
    shutil.copy(embedding_path / (supertable.stem.split('_')[0] + '_interpolated_embeddings.npy'), image_path_interpolated)
    return None

root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
embedding_path = root / 'BioMedCLIP_embeddings'
supertable_path = root / 'matched_supertables_with_images'
supertable_template = "_image_interpolated.npy"

supertables = list(supertable_path.glob("*" + supertable_template))
print(f"Found {len(supertables)} supertables")

output_dir = Path('/work/ma618/') / 'cxr'
output_dir.mkdir(exist_ok=True)

ehr_dir = output_dir / 'ehr_matrices'
ehr_dir.mkdir(exist_ok=True)

image_dir = output_dir / 'image_embeddings'
image_dir.mkdir(exist_ok=True)

patient_cohorts_dir = output_dir / 'patient_cohorts'
patient_cohorts_dir.mkdir(exist_ok=True)

#Has all npy files like not_longitudinal, mech_vent, etc.
shutil.copy(root / 'patient_cohorts/not_longitudinal.npy', output_dir / 'patient_cohorts/')
shutil.copy(root / 'patient_cohorts/all_mech_vent.npy', output_dir / 'patient_cohorts/')
shutil.copy(root / 'patient_cohorts/pre_vent.npy', output_dir / 'patient_cohorts/')
shutil.copy(root / 'patient_cohorts/post_vent.npy', output_dir / 'patient_cohorts/')

print("Copied patient cohorts")

n = cpu_count()
with Pool(n) as p:
    p.map(move_ehr_matrix, supertables)
    

print("Done")