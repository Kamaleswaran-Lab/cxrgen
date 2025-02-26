import torch
from torch.utils.data import DataLoader
import wandb
import os
from tqdm import tqdm
import sys 
sys.path.append('../')
from models.transformer import EncoderDecoderTransformer
from typing import Optional, Tuple
from pathlib import Path


class TrainerMLP:
    def __init__(self, model, optimizer, loss_fn, device='cuda', 
                 checkpoint_dir='checkpoints', log_wandb=True):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.log_wandb = log_wandb
        self.log_file = Path(self.checkpoint_dir) / 'train.log'
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        
        with tqdm(dataloader, desc=f'Epoch {epoch}') as pbar:
            for batch in pbar:
                # Move data to device
                ehr = batch['ehr'].to(self.device)
                prev_cxr = batch['prev_cxr'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                predictions = self.model(ehr, prev_cxr)
                loss = self.loss_fn(predictions, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                if self.log_wandb:
                    wandb.log({'batch_loss': loss.item()})
                

        
        epoch_loss = total_loss / len(dataloader)
        
        return epoch_loss
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_dir, epoch):
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    


class TimeSeriesTransformerTrainer:
    def __init__(self,
                 model: EncoderDecoderTransformer,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: callable,
                 device: str = 'cuda', checkpoint_dir: str = 'checkpoints',
                 teacher_forcing_ratio: float = 0.5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.checkpoint_dir = checkpoint_dir,
        self.model.to(device)
    
    def train_step(self, batch: dict) -> dict:
        """Train for a single batch."""
        self.model.train()
        
        # Move data to device
        ehr = batch['ehr'].to(self.device)
        prev_cxr = batch['prev_cxr'].to(self.device)
        target = batch['target'].to(self.device)
        mask = batch.get('attention_mask', None)
        if mask is not None:
            mask = mask.to(self.device)
        
        # Use teacher forcing based on ratio
        use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
        
        # Forward pass
        if use_teacher_forcing:
            # Shift targets for teacher forcing (exclude last step)
            target_input = torch.cat([
                torch.zeros_like(target[:, :1]),  # Start token
                target[:, :-1]  # Remaining tokens shifted by 1
            ], dim=1)
            
            # Use teacher forcing
            outputs = self.model(ehr, prev_cxr, target_input, mask, mask)
        else:
            # No teacher forcing
            outputs = self.model(ehr, prev_cxr, None, mask, mask)
        
        # Calculate loss
        loss = self.loss_fn(outputs, target)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {'loss': loss.item()}
    
    def save_checkpoint(self, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }
        path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_dir, epoch):
        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']
    


class TrainerConfig:
    """Configuration for the Trainer."""
    def __init__(
        self, 
        max_epochs: int = 100,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        loss_fn: str = "mse",
        weight_decay: float = 0.01,
        lr_scheduler: str = "cosine",
        warmup_ratio: float = 0.1,
        checkpoint_dir: str = "checkpoints",
        save_every: int = 1,
        patience: int = 10,
        grad_norm_clip: float = 1.0,
        log_every: int = 10,
        eval_every: int = 1,
        mixed_precision: bool = True,
        teacher_forcing_ratio: float = 0.5,
        teacher_forcing_decay: float = 0.9,
        classifier_path: str = 'classifier.pt',
        alpha: float = 0.5
    ):
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler = lr_scheduler
        self.warmup_ratio = warmup_ratio
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.patience = patience
        self.grad_norm_clip = grad_norm_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.mixed_precision = mixed_precision
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.teacher_forcing_decay = teacher_forcing_decay
        self.loss_fn = loss_fn
        self.classifier_path = classifier_path
        self.alpha = alpha

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
    
LABEL_COLUMNS = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly',
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia',
    'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other',
    'Fracture', 'Support Devices'
]
    
class MSE_BCE(torch.nn.Module):
    def __init__(self, classifer_path: str, alpha: float = 0.5):
        super(MSE_BCE, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.classifier_path = classifer_path
        self.mlp = MLPModel(512, len(LABEL_COLUMNS))
        self.mlp = torch.load(self.classifier_path, weights_only = False).to(self.device)
        self.alpha = alpha 
    def forward(self, y_pred, y_true):
        mse_loss = self.mse(y_pred, y_true)
        y_pred = y_pred.reshape(-1, 512)
        y_true = y_true.reshape(-1, 512)
        y_pred_labels, _ = self.mlp(y_pred)
        y_true_labels, _ = self.mlp(y_true)
        #y_pred_labels = torch.sigmoid(y_pred_labels)
        #y_true_labels = torch.sigmoid(y_true_labels)
        bce_loss = self.bce(y_pred_labels, y_true_labels)
        return (1-self.alpha)*mse_loss + self.alpha*bce_loss

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainerConfig,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        test_dataloader: Optional[torch.utils.data.DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.device = device
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize teacher forcing ratio
        self.teacher_forcing_ratio = config.teacher_forcing_ratio
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Setup learning rate scheduler
        self.total_steps = len(train_dataloader) * config.max_epochs
        self.warmup_steps = int(self.total_steps * config.warmup_ratio)
        
        if config.lr_scheduler == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_steps - self.warmup_steps
            )
        elif config.lr_scheduler == "linear":
            self.scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, 
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.total_steps - self.warmup_steps
            )
        elif config.lr_scheduler == "constant":
            self.scheduler = torch.optim.lr_scheduler.ConstantLR(
                self.optimizer, factor=1.0, total_iters=self.total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler: {config.lr_scheduler}")
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=self.warmup_steps
        )
        
        # Setup loss function
        if config.loss_fn == 'mse':
            self.criterion = torch.nn.MSELoss()
        elif config.loss_fn == 'mse_bce':
            self.criterion= MSE_BCE(config.classifier_path, config.alpha)
        else:
            raise ValueError(f"Loss function {config['loss_fn']} not supported")
        
        # Setup mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.mixed_precision)
        
        # Setup model
        self.model.to(device)
        
        # Tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
    def train(self):
        """Main training loop with validation."""
        wandb.init(project="medical-prediction", config=vars(self.config))
        
        # Track model and batch size
        model_size = sum(p.numel() for p in self.model.parameters())
        wandb.run.summary["model_parameters"] = model_size
        wandb.run.summary["batch_size"] = self.config.batch_size
        
        # Initialize metrics tracking
        best_epoch = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(self.config.max_epochs):
            # Decay teacher forcing ratio
            self.teacher_forcing_ratio *= self.config.teacher_forcing_decay
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch)
            train_losses.append(train_loss)
            
            # Evaluate on validation set
            if epoch % self.config.eval_every == 0:
                val_loss = self.evaluate(self.val_dataloader, "val")
                val_losses.append(val_loss)
                
                # Log epoch metrics
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "teacher_forcing_ratio": self.teacher_forcing_ratio,
                })
                
                # Save checkpoint if improved
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(f"best_model.pt", epoch, val_loss)
                    wandb.run.summary["best_val_loss"] = val_loss
                    wandb.run.summary["best_epoch"] = epoch
                else:
                    self.epochs_without_improvement += 1
                
                # Save regular checkpoint
                if epoch % self.config.save_every == 0:
                    self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt", epoch, val_loss)
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.patience:
                    print(f"Early stopping at epoch {epoch} as validation loss didn't improve for {self.config.patience} epochs")
                    break
        
        # Log learning curves
        epochs_range = list(range(len(train_losses)))
        data = [[x, y] for (x, y) in zip(epochs_range, train_losses)]
        table = wandb.Table(data=data, columns=["epoch", "train_loss"])
        wandb.log({"train_curve": wandb.plot.line(table, "epoch", "train_loss", title="Training Loss")})
        
        val_epochs = [i * self.config.eval_every for i in range(len(val_losses))]
        data = [[x, y] for (x, y) in zip(val_epochs, val_losses)]
        table = wandb.Table(data=data, columns=["epoch", "val_loss"])
        wandb.log({"val_curve": wandb.plot.line(table, "epoch", "val_loss", title="Validation Loss")})
        
        # Final evaluation on test set if available
        if self.test_dataloader is not None:
            # Load best model
            self.load_checkpoint(f"best_model.pt")
            test_loss = self.evaluate(self.test_dataloader, "test")
            wandb.run.summary["test_loss"] = test_loss
            
        wandb.finish()
        
        return {
            "best_epoch": best_epoch,
            "best_val_loss": self.best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch and return average loss."""
        self.model.train()
        total_loss = 0.0
        
        with tqdm(self.train_dataloader, unit="batch", desc=f"Epoch {epoch}") as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                # Move batch to device
                batch_encounters = batch['encounter_name']
                del batch['encounter_name']
                batch = {k: v.to(self.device) for k, v in batch.items()}
            
                batch['ehr'][torch.isinf(batch['ehr'])] = 0
                
                # Decide on teacher forcing
                use_teacher_forcing = torch.rand(1).item() < self.teacher_forcing_ratio
                target_input = batch["target"][:, :-1] if use_teacher_forcing else None
                
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(
                        ehr=batch["ehr"],
                        prev_cxr=batch["prev_cxr"],
                        target_input=None,
                        encoder_attention_mask=batch.get("attention_mask"),
                        decoder_attention_mask=batch.get("attention_mask"),
                        causal_mask=True
                    )
                    loss = self.criterion(outputs, batch["target"])
                
                # Backward pass with gradient scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                
                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Update learning rate
                if self.global_step < self.warmup_steps:
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step()
                
                # Update metrics
                total_loss += loss.item()
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]["lr"])
                
                # Log metrics
                if batch_idx % self.config.log_every == 0:
                    wandb.log({
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                        "train/global_step": self.global_step,
                    })
        
        avg_loss = total_loss / len(self.train_dataloader)
        return avg_loss
    
    @torch.no_grad()
    def evaluate(self, dataloader: torch.utils.data.DataLoader, split: str = "val") -> float:
        """Evaluate model on dataloader and return average loss."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating on {split}", leave=False):
                # Move batch to device
                batch_encounters = batch['encounter_name']
                del batch['encounter_name']

                batch = {k: v.to(self.device) for k, v in batch.items()}

                batch['ehr'][torch.isinf(batch['ehr'])] = 0
                
                # Forward pass
                outputs = self.model(
                    ehr=batch["ehr"],
                    prev_cxr=batch["prev_cxr"],
                    target_input=None,  # No teacher forcing during evaluation
                    encoder_attention_mask=batch.get("attention_mask"),
                    decoder_attention_mask=batch.get("attention_mask"),
                    causal_mask=True
                )
                loss = self.criterion(outputs, batch["target"])
                
                # Update metrics
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "warmup_scheduler_state_dict": self.warmup_scheduler.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "global_step": self.global_step,
            "val_loss": val_loss,
            "best_val_loss": self.best_val_loss,
            "epochs_without_improvement": self.epochs_without_improvement,
            "teacher_forcing_ratio": self.teacher_forcing_ratio,
        }, checkpoint_path)
    
    def load_checkpoint(self, filename: str) -> Tuple[int, float]:
        """Load model checkpoint and return epoch and validation loss."""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.warmup_scheduler.load_state_dict(checkpoint["warmup_scheduler_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.epochs_without_improvement = checkpoint["epochs_without_improvement"]
        self.teacher_forcing_ratio = checkpoint["teacher_forcing_ratio"]
        
        return checkpoint["epoch"], checkpoint["val_loss"]