import wandb 
import torch
import torch.nn as nn
from argparse import ArgumentParser
from pathlib import Path 
import os 
import sys
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import shutil


from models import mlp, transformernn
from data import dataloaders
from training.trainer import Trainer, TrainerMLP, TimeSeriesTransformerTrainer, TrainerConfig
from src.configs.config_def import set_paths, run_configs

class MLPModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256]):
        super(MLPModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        
        prev_size = input_size
        for size in hidden_sizes:
            self.layers.append(torch.nn.Linear(prev_size, size))
            self.layers.append(torch.nn.BatchNorm1d(size)) 
            prev_size = size
            
        self.output_layer = torch.nn.Linear(hidden_sizes[-1], output_size)  

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten (batch_size, 512, 1) â†’ (batch_size, 512)

        for i in range(0, len(self.layers), 2):  
            x = self.layers[i](x)  # Linear layer
            x = torch.relu(x)  
            x = self.layers[i + 1](x)  # BatchNorm layer

        final_hidden = x  
        output = self.output_layer(final_hidden)  

        return output, final_hidden 


set_paths()
config = run_configs()

def main():
    # Initialize wandb
    wandb.login(key=os.environ['WANDB_API_KEY'])
    wandb.init(project=config['wandb_project'], notes=config['wandb_notes'], dir=os.environ['WANDB_LOCAL_SAVE'], name=config['wandb_run_name'])
    wandb.config.update(config)
    
    root = Path(os.environ['DATA_DIR']) 
    datadir = root / 'longitudinal_data_corrected'
    embeddings_dir = datadir / 'image_embeddings'
    ehr_dir = datadir / 'ehr_matrices'
    checkpoint_dir = Path(os.environ['CHECKPOINT_DIR']) / config['wandb_project']

    mimic_classifier_path = Path(os.environ['MIMIC_CLASSIFIER']) / 'mlp_model_trained.pth'

    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy('/hpc/home/ma618/cxrgen/src/configs/config_def.py', checkpoint_dir)
    shutil.copy('/hpc/home/ma618/cxrgen/src/train.py', checkpoint_dir)
    
    encounter_paths = list(ehr_dir.glob("*.npy"))
    prev_cxr_paths = list(embeddings_dir.glob("*_ffill_embeddings.npy"))
    target_paths = list(embeddings_dir.glob("*_interpolated_embeddings.npy"))
    
    #Train and test 
    #train_encounters = np.load(root / 'train_15_16_19_20_21.npy', allow_pickle=True)
    #test_encounters = np.load(root / 'test_17_18.npy', allow_pickle=True)
    #train_encounters_ = [e.stem.split('_')[0] for e in train_encounters]
    #test_encounters_ = [e.stem.split('_')[0] for e in test_encounters]

    #Train, Test, Val split
    encounter_paths.sort()
    prev_cxr_paths.sort()
    target_paths.sort()

    #train test split random
    train_encounter_paths = encounter_paths[:int(0.8 * len(encounter_paths))]
    test_encounter_paths = encounter_paths[int(0.8 * len(encounter_paths)):]

    train_prev_cxr_paths = prev_cxr_paths[:int(0.8 * len(prev_cxr_paths))]
    test_prev_cxr_paths = prev_cxr_paths[int(0.8 * len(prev_cxr_paths)):]

    train_target_paths = target_paths[:int(0.8 * len(target_paths))]
    test_target_paths = target_paths[int(0.8 * len(target_paths)):]

    #Create val split from train split
    np.random.seed(42)
    train_encounter_paths = train_encounter_paths[:int(0.8 * len(train_encounter_paths))]
    val_encounter_paths = train_encounter_paths[int(0.8 * len(train_encounter_paths)):]
    train_prev_cxr_paths = train_prev_cxr_paths[:int(0.8 * len(train_prev_cxr_paths))]
    val_prev_cxr_paths = train_prev_cxr_paths[int(0.8 * len(train_prev_cxr_paths)):]
    train_target_paths = train_target_paths[:int(0.8 * len(train_target_paths))]
    val_target_paths = train_target_paths[int(0.8 * len(train_target_paths)):]
    
    #train_encounter_paths = [e for e in encounter_paths if e.stem.split('_')[0] in train_encounters_]
    #test_encounter_paths = [e for e in encounter_paths if e.stem.split('_')[0] in test_encounters_]

    #train_prev_cxr_paths = [e for e in prev_cxr_paths if e.stem.split('_')[0] in train_encounters]
    #test_prev_cxr_paths = [e for e in prev_cxr_paths if e.stem.split('_')[0] in test_encounters]

    #train_target_paths = [e for e in target_paths if e.stem.split('_')[0] in train_encounters]
    #test_target_paths = [e for e in target_paths if e.stem.split('_')[0] in test_encounters]

    #Create val split from train split 
    #np.random.seed(42)
    #np.random.shuffle(train_encounter_paths)
    #val_split = int(0.2 * len(train_encounter_paths))
    #val_encounter_paths = train_encounter_paths[:val_split]
    #train_encounter_paths = train_encounter_paths[val_split:]

    #val_encounters = [e.stem.split('_')[0] for e in val_encounter_paths]
    #train_encounters = [e.stem.split('_')[0] for e in train_encounter_paths]

    #val_prev_cxr_paths = [e for e in prev_cxr_paths if e.stem.split('_')[0] in val_encounters]
    #train_prev_cxr_paths = [e for e in prev_cxr_paths if e.stem.split('_')[0] in train_encounters]

    #val_target_paths = [e for e in target_paths if e.stem.split('_')[0] in val_encounters]
    #train_target_paths = [e for e in target_paths if e.stem.split('_')[0] in train_encounters]

    # Load data
    train_dataloader = dataloaders.create_encounter_dataloaders(encounter_paths=train_encounter_paths, 
                                        prev_cxr_paths=train_prev_cxr_paths,
                                        target_paths=train_target_paths,
                                        batch_size=config['batch_size'],
                                        model_type=config['model_type'],
                                        max_seq_length=config['max_seq_length'],
                                        num_workers=config['num_workers'],
                                        shuffle=config['shuffle'])
    
    val_dataloader = dataloaders.create_encounter_dataloaders(encounter_paths=val_encounter_paths,
                                        prev_cxr_paths=val_prev_cxr_paths,
                                        target_paths=val_target_paths,
                                        batch_size=config['batch_size'],
                                        model_type=config['model_type'],
                                        max_seq_length=config['max_seq_length'],
                                        num_workers=config['num_workers'],
                                        shuffle=config['shuffle'])
    
    test_dataloader = dataloaders.create_encounter_dataloaders(encounter_paths=test_encounter_paths,
                                        prev_cxr_paths=test_prev_cxr_paths,
                                        target_paths=test_target_paths,
                                        batch_size=config['batch_size'],
                                        model_type=config['model_type'],
                                        max_seq_length=config['max_seq_length'],
                                        num_workers=config['num_workers'],
                                        shuffle=config['shuffle'])
    

    
    # Initialize model
    if config['model_type'] == 'mlp':
        model = mlp.MLPPredictor(config['ehr_dim'], config['cxr_dim'])
    elif config['model_type'] == 'transformer':
        model_config = {'ehr_dim' : config['ehr_dim'],
                        'cxr_dim' : config['cxr_dim'],
                        'd_model' : config['d_model'],
                        'num_encoder_layers' : config['num_encoder_layers'],
                        'num_decoder_layers' : config['num_decoder_layers'],
                        'num_heads' : config['num_heads'],
                        'mlp_ratio' : config['mlp_ratio'],
                        'dropout' : config['dropout'],
                        'max_seq_length' : config['max_seq_length']}
        model = transformernn.create_transformer_model(model_config)
    elif config['model_type'] == 'transformer_concat':
        model_config = {'ehr_dim' : config['ehr_dim'],
                        'cxr_dim' : config['cxr_dim'],
                        'd_model' : config['d_model'],
                        'num_encoder_layers' : config['num_encoder_layers'],
                        'num_decoder_layers' : config['num_decoder_layers'],
                        'num_heads' : config['num_heads'],
                        'mlp_ratio' : config['mlp_ratio'],
                        'dropout' : config['dropout'],
                        'max_seq_length' : config['max_seq_length']}
        model = transformernn.create_transformerconcat_model(model_config)
    

    
    # Train model
    trainer_config = TrainerConfig(
        max_epochs=config['num_epochs'],
        batch_size=config['batch_size'],
        learning_rate=config['lr'],
        loss_fn=config['loss_fn'],
        weight_decay=config['weight_decay'],
        lr_scheduler=config['lr_scheduler'],
        warmup_ratio=config['warmup_ratio'],
        checkpoint_dir= checkpoint_dir,
        save_every=config['save_every'],
        patience=config['patience'],
        grad_norm_clip=config['grad_norm_clip'],
        log_every=config['log_every'],
        eval_every=config['eval_every'],
        mixed_precision=config['mixed_precision'],
        teacher_forcing_ratio=config['teacher_forcing_ratio'],
        teacher_forcing_decay=config['teacher_forcing_decay'],  
        classifier_path=mimic_classifier_path,
        alpha=config['alpha']
    )

    trainer = Trainer(
        model=model,
        config=trainer_config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader
    )
    
    # Train model
    results = trainer.train()
    
    # Print best results
    print(f"Best epoch: {results['best_epoch']} with validation loss: {results['best_val_loss']:.4f}")

if __name__ == '__main__':
    main()  
