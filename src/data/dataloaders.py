import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

class MLPEncounterDataset(Dataset):
    def __init__(self, 
                 encounter_paths: List[str],
                 prev_cxr_paths: List[str],
                 target_paths: List[str]):
        """
        Memory-efficient dataset using memory-mapped numpy arrays.
        
        Args:
            encounter_paths: List of paths to encounter .npy files
            prev_cxr_paths: List of paths to previous CXR embedding .npy files
            target_paths: List of paths to target embedding .npy files
        """
        self.encounter_paths = encounter_paths
        self.prev_cxr_paths = prev_cxr_paths
        self.target_paths = target_paths


        # Create index mapping without loading data
        #self.index_map = []  # (encounter_idx, hour_idx)
        
        # Use memmap to get array shapes without loading
        #for enc_idx, enc_path in enumerate(encounter_paths):
        #    ehr_array = np.load(enc_path)
        #    prev_cxr_array = np.load(prev_cxr_paths[enc_idx])
        #    target_array = np.load(target_paths[enc_idx])
        #    
        #    num_hours = len(ehr_array)
        #    assert len(prev_cxr_array) == len(target_array) == num_hours, \
        #        f"Mismatch in hours for encounter {enc_idx}"
            
        #    for hour_idx in range(num_hours):
        #        self.index_map.append((enc_idx, hour_idx))
    
    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        path = self.encounter_paths[idx]
        
        # Extract encounter and hour indices
        path_split = path.stem.split('_')
        enc_idx = int(path_split[0])
        #hour_idx = int(path_split[1])

        prev_cxr_path = self.prev_cxr_paths[enc_idx]
        target_path = self.target_paths[enc_idx]

        prev_cxr_path_split = prev_cxr_path.stem.split('_')
        target_path_split = target_path.stem.split('_')

        assert enc_idx == int(prev_cxr_path_split[0]) == int(target_path_split[0]), \
            f"Encounter index mismatch: {enc_idx} vs {prev_cxr_path_split[0]} vs {target_path_split[0]}"
        

        # Memory map the arrays and get specific rows
        ehr_array = np.load(path)
        prev_cxr_array = np.load(prev_cxr_path)
        target_array = np.load(target_path)

        mask_cxr_endpoints = np.int32(ehr_array[:, -1]).copy()  # Mask for CXR endpoints
        ehr_features = ehr_array[mask_cxr_endpoints][:, :-1]
        prev_cxr_array = prev_cxr_array[mask_cxr_endpoints]
        target_array = target_array[mask_cxr_endpoints]

        n = len(ehr_features) 
        random_idx = np.random.randint(0, n)
        ehr_features = ehr_features[random_idx]
        prev_cxr_array = prev_cxr_array[random_idx]
        target_array = target_array[random_idx]

        # Handle missing values
        ehr_features = np.nan_to_num(ehr_features, nan=0.0)
        
        return {
            'ehr': torch.tensor(ehr_features, dtype=torch.float32),
            'prev_cxr': torch.tensor(prev_cxr_array, dtype=torch.float32),
            'target': torch.tensor(target_array, dtype=torch.float32)
        }

class TransformerEncounterDataset(Dataset):
    def __init__(self,
                 encounter_paths: List[str],
                 prev_cxr_paths: List[str],
                 target_paths: List[str],
                 max_seq_length: int = 500):
        """
        Memory-efficient dataset for transformer using memory-mapped numpy arrays.
        """
        self.encounter_paths = encounter_paths
        self.prev_cxr_paths = prev_cxr_paths
        self.target_paths = target_paths
        self.max_seq_length = max_seq_length
    
        # Store sequence lengths using memmap
        #self.sequence_lengths = []
        #for enc_idx, enc_path in enumerate(encounter_paths):
        #     ehr_array = np.load(enc_path)
        #     mask_cxr_endpoints = ehr_array[:, -1]  # Mask for CXR endpoints       
        #     num_hours = sum(mask_cxr_endpoints)
        #     self.sequence_lengths.append(min(num_hours, max_seq_length))
    
    def pad_sequence(self, sequence: np.ndarray, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
        """Pad sequence and create attention mask."""
        seq_len = len(sequence)
        if seq_len > max_len:
            return sequence[seq_len - max_len:], np.ones(max_len)
        
        padding_len = max_len - seq_len
        padded_seq = np.pad(sequence, ((0, padding_len), (0, 0)), mode='constant', constant_values=0)
        attention_mask = np.concatenate([np.ones(seq_len), np.zeros(padding_len)])
        return padded_seq, attention_mask
    
    def __len__(self):
        return len(self.encounter_paths)
    
    def __getitem__(self, idx):
        encounter_path = self.encounter_paths[idx]
        prev_cxr_path = self.prev_cxr_paths[idx]
        target_path = self.target_paths[idx]

        #Check encounter name 
        assert Path(encounter_path).stem.split('_')[0] == Path(prev_cxr_path).stem.split('_')[0] == Path(target_path).stem.split('_')[0], \
            f"Encounter name mismatch: {Path(encounter_path).stem} vs {Path(prev_cxr_path).stem} vs {Path(target_path).stem}"
                
        # Memory map arrays and get sequences
        ehr_array = np.load(encounter_path)
        prev_cxr_array = np.load(prev_cxr_path)
        target_array = np.load(target_path)
        
        mask_cxr_endpoints = ehr_array[:, -1].astype(bool)  # Mask for CXR endpoints
        
        # Get sequences up to max_length (this only loads needed rows)
        ehr_features = ehr_array[mask_cxr_endpoints][:, :-1]
        prev_cxr = prev_cxr_array[mask_cxr_endpoints]
        target = target_array[mask_cxr_endpoints]
        
        # Pad sequences and create attention mask
        ehr_features, attention_mask = self.pad_sequence(ehr_features, self.max_seq_length)
        prev_cxr, _ = self.pad_sequence(prev_cxr, self.max_seq_length)
        target, _ = self.pad_sequence(target, self.max_seq_length)
        
        # Handle missing values
        ehr_features = np.nan_to_num(ehr_features, nan=0.0)
        
        return {
            'ehr': torch.tensor(ehr_features, dtype=torch.float32),
            'prev_cxr': torch.tensor(prev_cxr, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
            'encounter_name': Path(encounter_path).stem
        }
    
def create_encounter_dataloaders(
    encounter_paths : List[str],
    prev_cxr_paths : List[str],
    target_paths : List[str],
    batch_size: int,
    model_type: str = 'mlp',
    max_seq_length: int = 500,
    num_workers: int = 4,
    shuffle: bool = True
) -> DataLoader:
    """
    Create dataloader from List of paths to encounter, previous CXR, and target embeddings numpy files.
    """
    
    # Convert paths to strings and sort them

    encounter_paths = [str(p) for p in encounter_paths]
    prev_cxr_paths = [str(p) for p in prev_cxr_paths]
    target_paths = [str(p) for p in target_paths]
    
    encounter_paths.sort()
    prev_cxr_paths.sort()
    target_paths.sort()

    if model_type.lower() == 'mlp':
        dataset = MLPEncounterDataset(
            encounter_paths=encounter_paths,
            prev_cxr_paths=prev_cxr_paths,
            target_paths=target_paths
        )
    elif model_type.lower().startswith('transformer'):
        dataset = TransformerEncounterDataset(
            encounter_paths=encounter_paths,
            prev_cxr_paths=prev_cxr_paths,
            target_paths=target_paths,
            max_seq_length=max_seq_length
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
