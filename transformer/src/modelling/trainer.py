import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Callable
from tqdm import tqdm
import wandb



class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any] = None,
        device: str = 'cpu',
        grad_clip: Optional[float] = 1.0,
        use_wandb: bool = False,
        wandb_project: str = "ImplementingTransformers",
        wandb_config: Optional[Dict] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.grad_clip = grad_clip
        self.use_wandb = use_wandb
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.step = 0
        
        if self.use_wandb:
            wandb.init(project=wandb_project, config=wandb_config)
            wandb.watch(model, log="all", log_freq=100)
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> float:
    
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            src = batch['src'].to(self.device)
            tgt_input = batch['tgt_input'].to(self.device)
            tgt_output = batch['tgt_output'].to(self.device)
            
            src_mask = batch.get('src_mask', None)
            tgt_mask = batch.get('tgt_mask', None)
            if src_mask is not None:
                src_mask = src_mask.to(self.device)
            if tgt_mask is not None:
                tgt_mask = tgt_mask.to(self.device)
            
            output = self.model(src, tgt_input, src_mask, tgt_mask)
            
 
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.scheduler.get_lr()[0] if hasattr(self.scheduler, 'get_lr') else self.optimizer.param_groups[0]['lr']
                self.learning_rates.append(current_lr)
            
            total_loss += loss.item()
            num_batches += 1
            self.step += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            if self.use_wandb:
                log_dict = {"train/loss": loss.item(), "train/step": self.step}
                if self.scheduler is not None:
                    current_lr = self.scheduler.get_lr()[0] if hasattr(self.scheduler, 'get_lr') else self.optimizer.param_groups[0]['lr']
                    log_dict["train/learning_rate"] = current_lr
                wandb.log(log_dict)
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(
        self,
        dataloader: DataLoader
    ) -> float:

        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                src = batch['src'].to(self.device)
                tgt_input = batch['tgt_input'].to(self.device)
                tgt_output = batch['tgt_output'].to(self.device)
                
                src_mask = batch.get('src_mask', None)
                tgt_mask = batch.get('tgt_mask', None)
                if src_mask is not None:
                    src_mask = src_mask.to(self.device)
                if tgt_mask is not None:
                    tgt_mask = tgt_mask.to(self.device)
                
                output = self.model(src, tgt_input, src_mask, tgt_mask)
                
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        if self.use_wandb:
            wandb.log({"val/loss": avg_loss, "val/epoch": len(self.val_losses)})
        
        return avg_loss
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        save_path: Optional[str] = None,
        save_extra: Optional[Dict[str, Any]] = None,
        on_epoch_end: Optional[Callable[[int, float, float], None]] = None
    ):
        best_val_loss = float('inf')
        
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
                print(f"Created checkpoint directory: {save_dir}")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch(train_dataloader, epoch)
            print(f"Train Loss: {train_loss:.4f}")
            
            val_loss = None
            if val_dataloader is not None:
                val_loss = self.validate(val_dataloader)
                print(f"Val Loss: {val_loss:.4f}")
                
                if save_path is not None and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }
                    if save_extra:
                        checkpoint.update(save_extra)
                    torch.save(checkpoint, save_path)
                    print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            if self.scheduler is not None:
                current_lr = self.scheduler.get_lr()[0] if hasattr(self.scheduler, 'get_lr') else self.optimizer.param_groups[0]['lr']
                print(f"Learning Rate: {current_lr:.6f}")
            
            if self.use_wandb:
                epoch_log = {"epoch": epoch, "epoch/train_loss": train_loss}
                if val_loss is not None:
                    epoch_log["epoch/val_loss"] = val_loss
                wandb.log(epoch_log)
            
            if on_epoch_end is not None:
                on_epoch_end(epoch, train_loss, val_loss if val_loss else 0.0)
        
        if self.use_wandb:
            wandb.finish()
        
        return best_val_loss
    
    def get_metrics(self) -> Dict[str, list]:
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates
        }

