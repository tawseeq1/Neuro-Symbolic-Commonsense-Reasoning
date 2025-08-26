"""
Bi-encoder model for retrieval-based question answering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from transformers import AutoModel, AutoTokenizer

from .base import BaseModel


class BiEncoderModel(BaseModel):
    """Bi-encoder model for retrieval-based QA."""
    
    def __init__(self, model_name: str, embedding_dim: int = 768, **kwargs):
        super().__init__(model_name, num_labels=1, **kwargs)  # num_labels not used for bi-encoder
        self.embedding_dim = embedding_dim
        
        # Use base model instead of sequence classification
        self.model = AutoModel.from_pretrained(model_name)
        
        # Projection layer for embeddings
        self.projection = nn.Linear(self.model.config.hidden_size, embedding_dim)
        
        # Temperature parameter for similarity computation
        self.temperature = nn.Parameter(torch.ones([]) * 0.1)
    
    def _create_model(self, **kwargs):
        """Create the base transformer model."""
        return AutoModel.from_pretrained(self.model_name, **kwargs)
    
    def encode_query(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode query text to embeddings."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        query_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(query_embeddings)
    
    def encode_candidates(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode candidate texts to embeddings."""
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        candidate_embeddings = outputs.last_hidden_state[:, 0, :]
        return self.projection(candidate_embeddings)
    
    def forward(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                candidate_input_ids: torch.Tensor, candidate_attention_mask: torch.Tensor,
                labels: Optional[torch.Tensor] = None, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the bi-encoder."""
        # Encode queries and candidates
        query_embeddings = self.encode_query(query_input_ids, query_attention_mask)
        candidate_embeddings = self.encode_candidates(candidate_input_ids, candidate_attention_mask)
        
        # Normalize embeddings
        query_embeddings = F.normalize(query_embeddings, p=2, dim=-1)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=-1)
        
        # Compute similarity scores
        similarity_scores = torch.matmul(query_embeddings, candidate_embeddings.t()) / self.temperature
        
        result = {"similarity_scores": similarity_scores}
        
        if labels is not None:
            # Compute contrastive loss
            loss = self.compute_contrastive_loss(similarity_scores, labels)
            result["loss"] = loss
        
        return result
    
    def compute_contrastive_loss(self, similarity_scores: torch.Tensor, 
                                labels: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss for retrieval."""
        # Labels should indicate which candidate is correct for each query
        batch_size = similarity_scores.size(0)
        
        # Create positive pairs (query, correct_candidate)
        positive_scores = similarity_scores[torch.arange(batch_size), labels]
        
        # Create negative pairs (query, incorrect_candidates)
        negative_scores = similarity_scores.clone()
        negative_scores[torch.arange(batch_size), labels] = float('-inf')
        
        # Compute contrastive loss
        logits = torch.cat([positive_scores.unsqueeze(1), negative_scores], dim=1)
        labels_contrastive = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        loss_fct = nn.CrossEntropyLoss()
        return loss_fct(logits, labels_contrastive)
    
    def get_retrieval_scores(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                            candidate_input_ids: torch.Tensor, candidate_attention_mask: torch.Tensor) -> torch.Tensor:
        """Get retrieval scores for query-candidate pairs."""
        outputs = self.forward(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            candidate_input_ids=candidate_input_ids,
            candidate_attention_mask=candidate_attention_mask
        )
        return outputs["similarity_scores"]
    
    def get_best_candidate(self, query_input_ids: torch.Tensor, query_attention_mask: torch.Tensor,
                          candidate_input_ids: torch.Tensor, candidate_attention_mask: torch.Tensor) -> torch.Tensor:
        """Get the index of the best candidate."""
        scores = self.get_retrieval_scores(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            candidate_input_ids=candidate_input_ids,
            candidate_attention_mask=candidate_attention_mask
        )
        return torch.argmax(scores, dim=-1) 