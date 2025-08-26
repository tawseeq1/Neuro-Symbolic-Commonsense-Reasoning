"""
Main trainer class for neuro-symbolic models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
import wandb
import os
from tqdm import tqdm
import json

from ..models.base import BaseModel
from ..reasoning.fusion import NeuroSymbolicFusion
from ..evaluation.metrics import compute_metrics
from .optimizer import get_optimizer, get_scheduler


class Trainer:
    """Main trainer for neuro-symbolic models."""
    
    def __init__(self, model: BaseModel, fusion_model: Optional[NeuroSymbolicFusion] = None,
                 train_dataloader: Optional[DataLoader] = None,
                 val_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None,
                 config: Dict[str, Any] = None):
        self.model = model
        self.fusion_model = fusion_model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.config = config or {}
        
        # Training setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.fusion_model:
            self.fusion_model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = get_optimizer(
            model=self.model if not self.fusion_model else self.fusion_model,
            **self.config.get("optimizer", {})
        )
        
        self.scheduler = get_scheduler(
            optimizer=self.optimizer,
            **self.config.get("scheduler", {})
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0.0
        self.training_history = []
        
        # Setup wandb if configured
        if self.config.get("use_wandb", False):
            wandb.init(
                project=self.config.get("wandb_project", "neuro-symbolic"),
                config=self.config
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        if self.fusion_model:
            self.fusion_model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.fusion_model:
                outputs = self.fusion_model(**batch)
                loss = self.fusion_model.compute_loss(
                    neural_outputs=outputs,
                    symbolic_outputs=None,
                    labels=batch["labels"]
                )
            else:
                outputs = self.model(**batch)
                loss = outputs.get("loss", 0.0)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config["max_grad_norm"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / num_batches:.4f}"
            })
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        if self.fusion_model:
            self.fusion_model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.fusion_model:
                    outputs = self.fusion_model(**batch)
                    loss = self.fusion_model.compute_loss(
                        neural_outputs=outputs,
                        symbolic_outputs=None,
                        labels=batch["labels"]
                    )
                    predictions = outputs["predictions"]
                else:
                    outputs = self.model(**batch)
                    loss = outputs.get("loss", 0.0)
                    predictions = outputs["predictions"]
                
                # Collect predictions and labels
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                
                total_loss += loss.item()
                num_batches += 1
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_labels)
        metrics["val_loss"] = total_loss / num_batches
        
        return metrics
    
    def train(self, num_epochs: int, save_dir: str = "checkpoints") -> Dict[str, List[float]]:
        """Train the model for multiple epochs."""
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step(val_metrics["val_loss"])
            
            # Log metrics
            metrics = {**train_metrics, **val_metrics}
            self.training_history.append(metrics)
            
            if self.config.get("use_wandb", False):
                wandb.log(metrics, step=epoch)
            
            # Save best model
            val_metric = val_metrics.get("accuracy", val_metrics.get("f1", 0.0))
            if val_metric > self.best_val_metric:
                self.best_val_metric = val_metric
                self.save_checkpoint(os.path.join(save_dir, "best_model.pt"))
            
            # Save checkpoint every few epochs
            if (epoch + 1) % self.config.get("save_every", 5) == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            
            # Print metrics
            print(f"Epoch {epoch+1}/{num_epochs}")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            print()
        
        return self.training_history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "current_epoch": self.current_epoch,
            "best_val_metric": self.best_val_metric,
            "training_history": self.training_history
        }
        
        if self.fusion_model:
            checkpoint["fusion_model_state_dict"] = self.fusion_model.state_dict()
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "fusion_model_state_dict" in checkpoint and self.fusion_model:
            self.fusion_model.load_state_dict(checkpoint["fusion_model_state_dict"])
        
        self.current_epoch = checkpoint["current_epoch"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.training_history = checkpoint["training_history"]
        
        print(f"Checkpoint loaded from {path}")
    
    def evaluate(self, dataloader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if dataloader is None:
            dataloader = self.test_dataloader
        
        if dataloader is None:
            raise ValueError("No dataloader provided for evaluation")
        
        self.model.eval()
        if self.fusion_model:
            self.fusion_model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.fusion_model:
                    outputs = self.fusion_model(**batch)
                    predictions = outputs["predictions"]
                    probabilities = outputs["combined_probs"]
                else:
                    outputs = self.model(**batch)
                    predictions = outputs["predictions"]
                    probabilities = outputs.get("probabilities", torch.softmax(outputs["logits"], dim=-1))
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_labels, all_probabilities)
        
        return metrics 