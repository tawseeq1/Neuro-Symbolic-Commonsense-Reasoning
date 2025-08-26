"""
Entailment model for sequence classification.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from transformers import AutoModelForSequenceClassification

from .base import BaseModel


class EntailmentModel(BaseModel):
    """Entailment model using AutoModelForSequenceClassification."""
    
    def __init__(self, model_name: str, num_labels: int = 2, **kwargs):
        super().__init__(model_name, num_labels, **kwargs)
    
    def _create_model(self, **kwargs):
        """Create the sequence classification model."""
        return AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
            **kwargs
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the entailment model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        result = {"logits": outputs.logits}
        
        if labels is not None:
            result["loss"] = outputs.loss
        
        return result
    
    def get_entailment_scores(self, input_ids: torch.Tensor, 
                             attention_mask: torch.Tensor) -> torch.Tensor:
        """Get entailment scores (probability of entailment)."""
        logits = self.get_logits(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)
        # Return probability of positive class (entailment)
        return probs[:, 1] if self.num_labels == 2 else probs
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss."""
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.num_labels), labels.view(-1)) 