import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from typing import Callable, List, Dict, Tuple
import os


import src.utils

class cxrDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        notesdf: pd.DataFrame,
        root_dir: str,
        image_dir: str,
        acc_to_path: Callable,
        transform=None,
        max_sequence_length: int = None
    ):
        """
        Args:
            df: DataFrame with meta data about cxr images
            notesdf: DataFrame with notes data, matched by accession number
            root_dir: Root directory for images
            image_dir: Directory containing images under root_dir
            acc_to_path: Function that maps (accession_number, image_dir, series_number, root_dir) to image path
            transform: Optional transform to be applied on images
            max_sequence_length: Maximum number of images in sequence (pad/trim if needed)
        """
        self.root_dir = root_dir
        self.image_dir = image_dir
        self.acc_to_path = acc_to_path
        self.transform = transform
        self.max_sequence_length = max_sequence_length
        self.notesdf = notesdf
        
        # Process the dataframe to get sequences
        self.sequences = self._process_dataframe(df)
        
    def _process_dataframe(self, df: pd.DataFrame) -> List[Dict]:
        # Sort by study date
        df['StudyDate'] = pd.to_datetime(df['StudyDate'])
        df = df.sort_values('StudyDate')
        
        # Group by encounter and get the lowest series number for each accession
        sequences = []
        
        for encounter_id, group in df.groupby('ENCOUNTER_NBR'):
            # Get unique accessions and their earliest series
            unique_accessions = (
                group.sort_values('SeriesNumber')
                .groupby('AccessionNumber')
                .first()
                .reset_index()
            )
            
            # Sort by study date within encounter
            unique_accessions = unique_accessions.sort_values('StudyDate')
            
            if len(unique_accessions) < 2:
                continue
            
            # Add notes to the sequence, matching the order of accession number
            unique_accession_list = unique_accessions['AccessionNumber'].tolist()
            notes = [self.notesdf.loc[self.notesdf.ACC_NBR == unique_accession_list[x]]['DOC_TEXT'].values[0] \
                     for x in range(len(unique_accession_list))]

            sequence = {
                'ENCOUNTER_NBR': encounter_id,
                'AccessionNumber': unique_accession_list,
                'SeriesNumber': unique_accessions['SeriesNumber'].tolist(),
                'StudyDate': unique_accessions['StudyDate'].tolist(),
                'Year': unique_accessions['year'].astype(str).tolist(),
                'Notes': notes
            }
            
            sequences.append(sequence)
            
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def _load_and_process_image(self, year: str, acc_num: str, series_num: int) -> torch.Tensor:
        """Load and process a single image."""
        img_path = self.acc_to_path(self.root_dir, year, self.image_dir, acc_num, series_num)
        image = Image.open(img_path).convert('RGB')  # Convert to grayscale


        if self.transform:
            image = self.transform(image)
        else:
            # Basic transform if none provided
            image = torch.from_numpy(np.array(image)).float() / 255.0
            
        return image
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        sequence = self.sequences[idx]
        
        # Load all images in the sequence
        images = []
        for acc_num, series_num, year in zip(
            sequence['AccessionNumber'],
            sequence['SeriesNumber'],
            sequence['Year']
        ):
            img = self._load_and_process_image(year, acc_num, series_num)
            images.append(img)
            
        # Handle sequence length
        if self.max_sequence_length is not None:
            if len(images) > self.max_sequence_length:
                # Trim sequence
                images = images[:self.max_sequence_length]
                for key in ['AccessionNumber', 'SeriesNumber', 'StudyDate', 'Year', 'Notes']:
                    sequence[key] = sequence[key][:self.max_sequence_length]
            elif len(images) < self.max_sequence_length:
                # Pad sequence with zeros
                pad_length = self.max_sequence_length - len(images)
                pad_shape = list(images[0].shape)
                padding = [torch.zeros(pad_shape) for _ in range(pad_length)]
                images.extend(padding)
                
                # Add padding indicators to metadata
                for key in ['AccessionNumber', 'SeriesNumber', 'StudyDate', 'Year']:
                    sequence[key].extend([None] * pad_length)
                
                if sequence['Notes'] is None:
                    sequence['Notes'] = [''] * self.max_sequence_length
                else:
                    sequence['Notes'].extend([''] * pad_length)
        
        # Stack images into a single tensor
        image_sequence = torch.stack(images)
        
        # Create mask for valid (non-padded) positions
        valid_mask = torch.tensor([acc is not None for acc in sequence['AccessionNumber']])
        
        # Add mask to metadata
        sequence['valid_mask'] = valid_mask
        notes = sequence['Notes'] if sequence['Notes'] is not None else [''] * len(images)
        notes = [note if note is not None else '' for note in notes]

        return image_sequence, sequence['valid_mask'], notes 

def get_dataloader(
    df: pd.DataFrame,
    notesdf: pd.DataFrame,
    root_dir: str,
    image_dir: str,
    acc_to_path: Callable,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    max_sequence_length: int = None
) -> DataLoader:
    """
    Creates a DataLoader for the sequential image dataset.
    
    Args:
        df: DataFrame with required columns
        notesdf: DataFrame with notes data, matched by accession number
        root_dir: Root directory for images
        image_dir: Directory containing images under root_dir
        acc_to_path: Function to map accession numbers to image paths
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        transform: Optional transforms to apply to images
        max_sequence_length: Maximum sequence length (pad/trim if needed)
    
    Returns:
        DataLoader object
    """

    def custom_collate(batch):
    # Unzip the batch into separate lists
        images, masks, notes = zip(*batch)
        
        # Stack images and masks
        images = torch.stack(images)
        masks = torch.stack(masks)
        
        # Keep notes as list of lists
        notes = list(notes)
        
        return images, masks, notes

    dataset = cxrDataset(
        df=df,
        notesdf=notesdf,
        root_dir=root_dir,
        image_dir=image_dir,
        acc_to_path=acc_to_path,
        transform=transform,
        max_sequence_length=max_sequence_length
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate  # Using default collate as we handle padding in dataset
    )

# Example usage:
"""
# Create transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Sample DataFrame
df = pd.DataFrame({
    'encounter_id': [1, 1, 1, 2, 2],
    'accession_number': ['ACC1', 'ACC2', 'ACC3', 'ACC4', 'ACC5'],
    'study_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-02-01', '2023-02-02'],
    'series_number': [1, 1, 2, 1, 1]
})

# Create dataloader
dataloader = get_dataloader(
    df=df,
    root_dir='/path/to/images',
    acc_to_path=acc_to_path,
    transform=transform,
    max_sequence_length=5
)

# Iterate through batches
for batch_images, batch_metadata in dataloader:
    # batch_images shape: [batch_size, seq_length, channels, height, width]
    # batch_metadata contains sequence information and valid_mask
    pass
"""