import sys
import os
from pathlib import Path 

def set_paths():
    os.environ["DATA_DIR"] = "/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/"
    os.environ["CHECKPOINT_DIR"] = "/hpc/dctrl/ma618/checkpoints"
    os.environ["WANDB_API_KEY"] = "2f4fe79231d455c54d712d879e3b5333f7f21ed1"
    os.environ["WANDB_LOCAL_SAVE"] = "/hpc/dctrl/ma618"
    os.environ["MIMIC_CLASSIFIER"] = "/hpc/group/kamaleswaranlab/EmoryDataset/Images/chest_xrays/MIMIC_Classifier"
    return None

def run_configs():
    config = {
        'ehr_dim': 82,
        'cxr_dim': 512,
        'd_model': 512,
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'dropout': 0,
        'max_seq_length': 100,
        'batch_size': 32,
        'lr': 1e-4,
        'num_epochs': 50,
        'save_every': 5,
        'eval_every': 1,
        'patience': 15,
        'weight_decay':0.01,
        'lr_scheduler':"cosine",
        'warmup_ratio':0.1,
        'grad_norm_clip':1.0,
        'log_every':10,
        'teacher_forcing_ratio':0.0,
        'teacher_forcing_decay':0.98,
        'mixed_precision':True,
        'log_wandb': True,
        'model_type': 'transformer_concat',
        'num_workers': 4,
        'shuffle': True,
        'loss_fn': 'mse_bce',
        'alpha': 0.5,
        'optimizer': 'adam',
        'scheduler': 'none',
        'wandb_project': 'cxr-predictor-transformer',
        'wandb_run_name': 'transformer_concat_causal_eval_mlploss',
        'wandb_notes': 'Training simple encoder-decoder transformer model on EHR and CXR data, after ADDING projected prev cxr embeddings to ehr before positional encoding. debugged inf vals, concat the inputs. Making Target input None',
    }
    return config
