"""
Base model class for all neural models.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModel


class BaseModel(nn.Module):
    """Base class for all neural models."""
    
    def __init__(self, model_name: str, num_labels: int = 2, **kwargs):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        self.model = self._create_model(**kwargs)
    
    def _create_model(self, **kwargs):
        """Create the underlying transformer model."""
        raise NotImplementedError
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        raise NotImplementedError
    
    def predict(self, **inputs) -> Dict[str, torch.Tensor]:
        """Make predictions (same as forward for most models)."""
        return self.forward(**inputs)
    
    def get_logits(self, **inputs) -> torch.Tensor:
        """Get raw logits from the model."""
        outputs = self.forward(**inputs)
        return outputs["logits"]
    
    def get_probabilities(self, **inputs) -> torch.Tensor:
        """Get probabilities from the model."""
        logits = self.get_logits(**inputs)
        return torch.softmax(logits, dim=-1)
    
    def get_predictions(self, **inputs) -> torch.Tensor:
        """Get predicted class labels."""
        logits = self.get_logits(**inputs)
        return torch.argmax(logits, dim=-1)
    
    def save_pretrained(self, path: str):
        """Save the model and tokenizer."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load a pretrained model."""
        # This is a simplified version - in practice, you'd want to save/load more state
        model_name = kwargs.get("model_name", "roberta-base")
        num_labels = kwargs.get("num_labels", 2)
        return cls(model_name=model_name, num_labels=num_labels, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_name": self.model_name,
            "num_labels": self.num_labels,
            "model_type": self.__class__.__name__,
            "parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 