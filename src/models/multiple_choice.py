"""
Multiple choice model for question answering.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoModelForMultipleChoice

from .base import BaseModel


class MultipleChoiceModel(BaseModel):
    """Multiple choice model using AutoModelForMultipleChoice."""
    
    def __init__(self, model_name: str, num_choices: int = 4, **kwargs):
        super().__init__(model_name, num_labels=num_choices, **kwargs)
        self.num_choices = num_choices
    
    def _create_model(self, **kwargs):
        """Create the multiple choice model."""
        return AutoModelForMultipleChoice.from_pretrained(
            self.model_name,
            **kwargs
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the multiple choice model."""
        # Reshape inputs for multiple choice
        batch_size, num_choices, seq_len = input_ids.shape
        
        # Flatten for the model
        flat_input_ids = input_ids.view(-1, seq_len)
        flat_attention_mask = attention_mask.view(-1, seq_len)
        
        outputs = self.model(
            input_ids=flat_input_ids,
            attention_mask=flat_attention_mask,
            labels=labels,
            **kwargs
        )
        
        # Reshape logits back to (batch_size, num_choices)
        logits = outputs.logits.view(batch_size, num_choices)
        
        result = {"logits": logits}
        
        if labels is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def get_choice_scores(self, input_ids: torch.Tensor, 
                         attention_mask: torch.Tensor) -> torch.Tensor:
        """Get scores for each choice."""
        logits = self.get_logits(input_ids=input_ids, attention_mask=attention_mask)
        return torch.softmax(logits, dim=-1)
    
    def get_best_choice(self, input_ids: torch.Tensor, 
                       attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the index of the best choice."""
        logits = self.get_logits(input_ids=input_ids, attention_mask=attention_mask)
        return torch.argmax(logits, dim=-1)
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for multiple choice."""
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits, labels) 