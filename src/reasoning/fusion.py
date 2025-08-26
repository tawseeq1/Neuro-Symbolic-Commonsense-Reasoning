"""
Neuro-symbolic fusion strategies.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from ..models.base import BaseModel
from ..rules.psl import PSLReasoner, PSLRule
from ..rules.knowledge import KnowledgeRetriever


class NeuroSymbolicFusion(ABC, nn.Module):
    """Abstract base class for neuro-symbolic fusion strategies."""
    
    def __init__(self, neural_model: BaseModel, symbolic_reasoner: PSLReasoner, 
                 knowledge_retriever: Optional[KnowledgeRetriever] = None):
        super().__init__()
        self.neural_model = neural_model
        self.symbolic_reasoner = symbolic_reasoner
        self.knowledge_retriever = knowledge_retriever
    
    @abstractmethod
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Forward pass combining neural and symbolic reasoning."""
        pass
    
    @abstractmethod
    def compute_loss(self, neural_outputs: Dict[str, torch.Tensor], 
                    symbolic_outputs: Dict[str, torch.Tensor],
                    labels: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        pass


class PipelineFusion(NeuroSymbolicFusion):
    """Pipeline fusion: neural → symbolic."""
    
    def __init__(self, neural_model: BaseModel, symbolic_reasoner: PSLReasoner,
                 knowledge_retriever: Optional[KnowledgeRetriever] = None,
                 alpha: float = 0.5):
        super().__init__(neural_model, symbolic_reasoner, knowledge_retriever)
        self.alpha = alpha
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Pipeline fusion forward pass."""
        # Get neural predictions
        neural_outputs = self.neural_model(**inputs)
        neural_logits = neural_outputs["logits"]
        neural_probs = torch.softmax(neural_logits, dim=-1)
        
        # Get symbolic reasoning if knowledge retriever is available
        symbolic_scores = None
        if self.knowledge_retriever is not None:
            # Extract text for knowledge retrieval
            text_inputs = self._extract_text_inputs(inputs)
            facts = self._retrieve_knowledge(text_inputs)
            symbolic_outputs = self._apply_symbolic_reasoning(facts, neural_probs)
            symbolic_scores = symbolic_outputs["symbolic_scores"]
        
        # Combine neural and symbolic scores
        if symbolic_scores is not None:
            # Log-linear interpolation: logit' = logit + α * symbolic_score
            combined_logits = neural_logits + self.alpha * symbolic_scores.unsqueeze(1)
            combined_probs = torch.softmax(combined_logits, dim=-1)
        else:
            combined_logits = neural_logits
            combined_probs = neural_probs
        
        return {
            "neural_logits": neural_logits,
            "neural_probs": neural_probs,
            "symbolic_scores": symbolic_scores,
            "combined_logits": combined_logits,
            "combined_probs": combined_probs,
            "predictions": torch.argmax(combined_logits, dim=-1)
        }
    
    def compute_loss(self, neural_outputs: Dict[str, torch.Tensor], 
                    symbolic_outputs: Optional[Dict[str, torch.Tensor]],
                    labels: torch.Tensor) -> torch.Tensor:
        """Compute loss for pipeline fusion."""
        # Use neural loss only in pipeline mode
        if "loss" in neural_outputs:
            return neural_outputs["loss"]
        else:
            # Compute cross-entropy loss manually
            logits = neural_outputs["logits"]
            loss_fct = nn.CrossEntropyLoss()
            return loss_fct(logits, labels)
    
    def _extract_text_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text inputs for knowledge retrieval."""
        # This is a simplified version - in practice you'd need to handle different input formats
        text_inputs = {}
        
        if "input_ids" in inputs:
            # For entailment models
            text_inputs["text"] = self.neural_model.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
        elif "query_input_ids" in inputs:
            # For bi-encoder models
            text_inputs["query"] = self.neural_model.tokenizer.batch_decode(
                inputs["query_input_ids"], skip_special_tokens=True
            )
            text_inputs["candidates"] = self.neural_model.tokenizer.batch_decode(
                inputs["candidate_input_ids"], skip_special_tokens=True
            )
        
        return text_inputs
    
    def _retrieve_knowledge(self, text_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge."""
        facts = {}
        
        if "text" in text_inputs:
            # For entailment models
            for i, text in enumerate(text_inputs["text"]):
                facts[f"example_{i}"] = self.knowledge_retriever.retrieve(text, top_k=5)
        elif "query" in text_inputs and "candidates" in text_inputs:
            # For bi-encoder models
            for i, (query, candidates) in enumerate(zip(text_inputs["query"], text_inputs["candidates"])):
                facts[f"example_{i}"] = self.knowledge_retriever.get_facts_for_question(
                    query, candidates, top_k=5
                )
        
        return facts
    
    def _apply_symbolic_reasoning(self, facts: Dict[str, Any], 
                                 neural_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply symbolic reasoning to retrieved facts."""
        # Convert facts to symbolic scores
        batch_size = neural_probs.size(0)
        device = neural_probs.device
        symbolic_scores = torch.zeros(batch_size, device=device)
        
        # This is a simplified version - in practice you'd need more sophisticated fact-to-score mapping
        for i in range(batch_size):
            example_facts = facts.get(f"example_{i}", [])
            if example_facts:
                # Simple heuristic: higher score if more relevant facts found
                score = min(len(example_facts) * 0.1, 1.0)
                symbolic_scores[i] = score
        
        return {"symbolic_scores": symbolic_scores}


class JointFusion(NeuroSymbolicFusion):
    """Joint fusion: neural + symbolic with differentiable logic regularizer."""
    
    def __init__(self, neural_model: BaseModel, symbolic_reasoner: PSLReasoner,
                 knowledge_retriever: Optional[KnowledgeRetriever] = None,
                 lambda_logic: float = 0.1, alpha: float = 0.5):
        super().__init__(neural_model, symbolic_reasoner, knowledge_retriever)
        self.lambda_logic = lambda_logic
        self.alpha = alpha
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """Joint fusion forward pass."""
        # Get neural predictions
        neural_outputs = self.neural_model(**inputs)
        neural_logits = neural_outputs["logits"]
        neural_probs = torch.softmax(neural_logits, dim=-1)
        
        # Get symbolic reasoning
        symbolic_outputs = None
        logic_loss = torch.tensor(0.0, device=neural_logits.device)
        
        if self.knowledge_retriever is not None:
            text_inputs = self._extract_text_inputs(inputs)
            facts = self._retrieve_knowledge(text_inputs)
            symbolic_outputs = self._apply_symbolic_reasoning(facts, neural_probs)
            
            # Compute logic regularization loss
            if symbolic_outputs:
                logic_loss = self.symbolic_reasoner.compute_logic_loss(
                    facts=symbolic_outputs.get("facts", {}),
                    rule_instances=symbolic_outputs.get("rule_instances", [])
                )
        
        # Combine neural and symbolic scores
        symbolic_scores = symbolic_outputs["symbolic_scores"] if symbolic_outputs else None
        
        if symbolic_scores is not None:
            combined_logits = neural_logits + self.alpha * symbolic_scores.unsqueeze(1)
            combined_probs = torch.softmax(combined_logits, dim=-1)
        else:
            combined_logits = neural_logits
            combined_probs = neural_probs
        
        return {
            "neural_logits": neural_logits,
            "neural_probs": neural_probs,
            "symbolic_scores": symbolic_scores,
            "combined_logits": combined_logits,
            "combined_probs": combined_probs,
            "predictions": torch.argmax(combined_logits, dim=-1),
            "logic_loss": logic_loss
        }
    
    def compute_loss(self, neural_outputs: Dict[str, torch.Tensor], 
                    symbolic_outputs: Optional[Dict[str, torch.Tensor]],
                    labels: torch.Tensor) -> torch.Tensor:
        """Compute combined loss for joint fusion."""
        # Neural loss
        if "loss" in neural_outputs:
            neural_loss = neural_outputs["loss"]
        else:
            logits = neural_outputs["logits"]
            loss_fct = nn.CrossEntropyLoss()
            neural_loss = loss_fct(logits, labels)
        
        # Logic regularization loss
        logic_loss = neural_outputs.get("logic_loss", torch.tensor(0.0, device=neural_loss.device))
        
        # Combined loss: L_total = L_neural + λ * L_logic
        total_loss = neural_loss + self.lambda_logic * logic_loss
        
        return total_loss
    
    def _extract_text_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text inputs for knowledge retrieval."""
        # Same as PipelineFusion
        text_inputs = {}
        
        if "input_ids" in inputs:
            text_inputs["text"] = self.neural_model.tokenizer.batch_decode(
                inputs["input_ids"], skip_special_tokens=True
            )
        elif "query_input_ids" in inputs:
            text_inputs["query"] = self.neural_model.tokenizer.batch_decode(
                inputs["query_input_ids"], skip_special_tokens=True
            )
            text_inputs["candidates"] = self.neural_model.tokenizer.batch_decode(
                inputs["candidate_input_ids"], skip_special_tokens=True
            )
        
        return text_inputs
    
    def _retrieve_knowledge(self, text_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant knowledge."""
        # Same as PipelineFusion
        facts = {}
        
        if "text" in text_inputs:
            for i, text in enumerate(text_inputs["text"]):
                facts[f"example_{i}"] = self.knowledge_retriever.retrieve(text, top_k=5)
        elif "query" in text_inputs and "candidates" in text_inputs:
            for i, (query, candidates) in enumerate(zip(text_inputs["query"], text_inputs["candidates"])):
                facts[f"example_{i}"] = self.knowledge_retriever.get_facts_for_question(
                    query, candidates, top_k=5
                )
        
        return facts
    
    def _apply_symbolic_reasoning(self, facts: Dict[str, Any], 
                                 neural_probs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply symbolic reasoning to retrieved facts."""
        # Convert facts to symbolic scores and create fact tensors for PSL
        batch_size = neural_probs.size(0)
        device = neural_probs.device
        symbolic_scores = torch.zeros(batch_size, device=device)
        
        # Create fact tensors for PSL reasoning
        fact_tensors = {}
        rule_instances = []
        
        for i in range(batch_size):
            example_facts = facts.get(f"example_{i}", [])
            if example_facts:
                # Simple heuristic for symbolic scores
                score = min(len(example_facts) * 0.1, 1.0)
                symbolic_scores[i] = score
                
                # Create fact tensors for PSL (simplified)
                for j, fact in enumerate(example_facts[:3]):  # Limit to 3 facts per example
                    fact_name = f"fact_{i}_{j}"
                    fact_tensors[fact_name] = torch.tensor(fact["score"], device=device)
                    
                    # Create rule instances (simplified)
                    if j < len(self.symbolic_reasoner.rules):
                        rule = self.symbolic_reasoner.rules[j]
                        rule_instances.append((rule.name, fact_name, f"consequent_{i}_{j}"))
        
        return {
            "symbolic_scores": symbolic_scores,
            "facts": fact_tensors,
            "rule_instances": rule_instances
        } 