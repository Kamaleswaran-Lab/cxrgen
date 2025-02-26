import shutil
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import ParameterGrid
import pandas as pd
from multiprocessing import Pool, cpu_count

import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256]):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.BatchNorm1d(size)) 
            prev_size = size
            
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)  

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten (batch_size, 512, 1) â†’ (batch_size, 512)

        for i in range(0, len(self.layers), 2):  
            x = self.layers[i](x)  # Linear layer
            x = torch.relu(x)  
            x = self.layers[i + 1](x)  # BatchNorm layer

        final_hidden = x  
        output = self.output_layer(final_hidden)  

        return output, final_hidden  # Return both the final output and last hidden layer


def process_encounter(supertable_file_name):
    df = pd.read_pickle(supertable_file_name)
    cxrs = df['cxr_timing'].loc[df['cxr_timing'].notna()].values
    for cxr in cxrs:
        embeddingp = embedding_path / (cxr + '.npy')
        if not embeddingp.exists():
            print(f"Error for {supertable_file_name.stem}")
        else:
            shutil.copy(embeddingp, selected_embeddings / (embeddingp.stem.split('_')[0] + '.npy'))
            embedding = np.load(embeddingp)
            embedding = torch.tensor(embedding)
            embedding = embedding.unsqueeze(0)
            out, hidden = model(embedding)
            hidden = hidden.detach().numpy().reshape((-1,1))[:,0]
            np.save(selected_embeddings / (embeddingp.stem.split('_')[0] + '_postmlp.npy'), hidden)
    return None



root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
embedding_path = root / 'BioMedCLIP_embeddings'
dimreduce = root / 'dimReduce'
mimic_classifier_path = root / 'MIMIC_Classifier'
long_data = root / 'longitudinal_data_corrected'
embeddings_dir = long_data / 'image_embeddings'
ehr_dir = long_data / 'ehr_matrices'
notes_dir = long_data / 'notes'
selected_embeddings = embedding_path / 'selected_embeddings'
selected_embeddings.mkdir(exist_ok=True)

INPUT_SIZE = 512

LABEL_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]


OUTPUT_SIZE = len(LABEL_COLUMNS)  

model = MLPModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE)
model = torch.load(mimic_classifier_path / 'mlp_model_trained.pth', weights_only = False)

supertable_path = root / 'matched_supertables_with_images'
sups = list(supertable_path.glob("*_image_*.pickle"))

with Pool(cpu_count()) as p:
    p.map(process_encounter, sups)

print("All embeddings processed successfully")
