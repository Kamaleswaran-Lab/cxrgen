import torch
from urllib.request import urlopen
from PIL import Image
from open_clip import create_model_from_pretrained
import pandas as pd
from pathlib import Path
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["HF_HOME"] = "/hpc/dctrl/ma618"

sys.path.append('..')
import src.utils as utils

root = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays')
save_path = Path('/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/BioMedCLIP_embeddings')
os.makedirs(save_path, exist_ok=True)

YEARS = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
image_dir = "extracted-images"
metadata = "metadata_with_supertables_filtered_notes_filtered_with_img_paths.csv"
notes = "all_notes.csv"

#Pytorch dataset and dataloader for reading images from the 'img_path' column in the metadata dataframe 
class cxrStaticDataset(torch.utils.data.Dataset):
    def __init__(self, metadatadf, root, transform=None):
        self.metadatadf = metadatadf
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.metadatadf)

    def __getitem__(self, idx):
        img_path = self.metadatadf.iloc[idx]['img_paths']
        try:
            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
        except:
            img = Image.open(self.metadatadf.iloc[0]['img_paths'])
            if self.transform:
                img = self.transform(img)

        return img, Path(img_path).stem 

def main():
    #Read metadata and notes 
    metadatadf = pd.read_csv(root / metadata)
    print("Metadata read successfully")

    # Load the model and config files from the Hugging Face Hub
    model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    model = model.cuda()
    print("Model loaded successfully")

    #Dataloader
    cxrdataset = cxrStaticDataset(metadatadf, root, transform=preprocess)
    dataloader = torch.utils.data.DataLoader(cxrdataset, batch_size=64, shuffle=False, num_workers=4)

    #Get image embeddings
    for images, ids in tqdm(dataloader):
        inputs = images.cuda()
        with torch.no_grad():
            outputs = model.encode_image(inputs)
            outputs = outputs.cpu().numpy()
        for i, img_id in enumerate(ids):
            np.save(save_path / f"{img_id}.npy", outputs[i])
            print(f"Saved {img_id}.npy")

    print("All embeddings saved successfully")

if __name__ == "__main__":
    main()



