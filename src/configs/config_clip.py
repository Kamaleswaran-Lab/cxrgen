import sys
import os
from pathlib import Path 

def set_paths():
    os.environ["DATA_DIR"] = "/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/"
    os.environ["CHECKPOINT_DIR"] = "/hpc/dctrl/ma618/checkpoints"
    os.environ["WANDB_API_KEY"] = "2f4fe79231d455c54d712d879e3b5333f7f21ed1"
    os.environ["WANDB_LOCAL_SAVE"] = "/hpc/dctrl/ma618"
    os.environ["MIMIC_CLASSIFIER"] = "/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/MICCAI/MIMIC_Classifier"
    os.environ["INTERMEDIATE_DIR"] = "/cwork/ma618/" 
    return None

def run_configs():
    config = {
        # CLIP specific configurations will go here
    }
    return config 