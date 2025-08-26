"""
Probabilistic Soft Logic (PSL) implementation for symbolic reasoning.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class PSLRule:
    """A PSL rule with weighted logical implications."""
    
    name: str
    antecedent: str  # Left side of implication
    consequent: str  # Right side of implication
    weight: float = 1.0
    rule_type: str = "implication"  # implication, exception, default
    
    def __post_init__(self):
        if self.weight < 0:
            raise ValueError(f"Rule weight must be non-negative: {self.weight}")
    
    def to_string(self) -> str:
        """Convert rule to string representation."""
        return f"{self.antecedent} -> {self.consequent} [w={self.weight}]"
    
    def __str__(self) -> str:
        return self.to_string()


class PSLReasoner(nn.Module):
    """Probabilistic Soft Logic reasoner."""
    
    def __init__(self, rules: List[PSLRule], t_norm: str = "lukasiewicz"):
        super().__init__()
        self.rules = rules
        self.t_norm = t_norm
        
        # Convert rule weights to learnable parameters
        self.rule_weights = nn.Parameter(torch.tensor([rule.weight for rule in rules]))
        
        # Initialize rule satisfaction functions
        self.satisfaction_fns = {
            "lukasiewicz": self._lukasiewicz_satisfaction,
            "godel": self._godel_satisfaction,
            "product": self._product_satisfaction
        }
    
    def _lukasiewicz_satisfaction(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Compute satisfaction using Łukasiewicz t-norm."""
        # Łukasiewicz: min(1, 1 - A + B)
        return torch.clamp(1 - antecedent + consequent, min=0, max=1)
    
    def _godel_satisfaction(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Compute satisfaction using Gödel t-norm."""
        # Gödel: 1 if A <= B, B otherwise
        return torch.where(antecedent <= consequent, torch.ones_like(antecedent), consequent)
    
    def _product_satisfaction(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Compute satisfaction using product t-norm."""
        # Product: B / A if A > 0, 1 otherwise
        return torch.where(antecedent > 0, consequent / antecedent, torch.ones_like(antecedent))
    
    def compute_rule_satisfaction(self, antecedent: torch.Tensor, consequent: torch.Tensor) -> torch.Tensor:
        """Compute rule satisfaction using the specified t-norm."""
        if self.t_norm not in self.satisfaction_fns:
            raise ValueError(f"Unknown t-norm: {self.t_norm}")
        
        return self.satisfaction_fns[self.t_norm](antecedent, consequent)
    
    def forward(self, facts: Dict[str, torch.Tensor], 
                rule_instances: List[Tuple[str, str, str]]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PSL reasoner.
        
        Args:
            facts: Dictionary mapping fact names to their truth values
            rule_instances: List of (rule_name, antecedent_fact, consequent_fact) tuples
        
        Returns:
            Dictionary with rule satisfaction scores and symbolic scores
        """
        batch_size = next(iter(facts.values())).size(0)
        device = next(iter(facts.values())).device
        
        # Initialize symbolic scores
        symbolic_scores = torch.zeros(batch_size, device=device)
        rule_satisfactions = []
        
        for rule_name, antecedent_fact, consequent_fact in rule_instances:
            # Get rule weight
            rule_idx = next(i for i, rule in enumerate(self.rules) if rule.name == rule_name)
            weight = self.rule_weights[rule_idx]
            
            # Get fact values
            antecedent_val = facts[antecedent_fact]
            consequent_val = facts[consequent_fact]
            
            # Compute rule satisfaction
            satisfaction = self.compute_rule_satisfaction(antecedent_val, consequent_val)
            rule_satisfactions.append(satisfaction)
            
            # Add weighted satisfaction to symbolic scores
            symbolic_scores += weight * satisfaction
        
        # Normalize symbolic scores
        if len(rule_instances) > 0:
            symbolic_scores = symbolic_scores / len(rule_instances)
        
        return {
            "symbolic_scores": symbolic_scores,
            "rule_satisfactions": rule_satisfactions,
            "mean_satisfaction": torch.stack(rule_satisfactions).mean(dim=0) if rule_satisfactions else torch.zeros(batch_size, device=device)
        }
    
    def compute_logic_loss(self, facts: Dict[str, torch.Tensor], 
                          rule_instances: List[Tuple[str, str, str]]) -> torch.Tensor:
        """Compute logic regularization loss."""
        outputs = self.forward(facts, rule_instances)
        # Loss is 1 - mean rule satisfaction (higher satisfaction = lower loss)
        return 1 - outputs["mean_satisfaction"].mean()
    
    def get_rule_weights(self) -> Dict[str, float]:
        """Get current rule weights."""
        return {rule.name: float(self.rule_weights[i]) for i, rule in enumerate(self.rules)}
    
    def update_rule_weights(self, new_weights: Dict[str, float]):
        """Update rule weights."""
        for rule_name, weight in new_weights.items():
            rule_idx = next(i for i, rule in enumerate(self.rules) if rule.name == rule_name)
            self.rule_weights.data[rule_idx] = weight


def create_default_rules() -> List[PSLRule]:
    """Create default commonsense rules."""
    rules = [
        # Physical commonsense
        PSLRule("bird_can_fly", "Bird(x)", "CanFly(x)", weight=0.8, rule_type="default"),
        PSLRule("penguin_cannot_fly", "Penguin(x)", "¬CanFly(x)", weight=1.0, rule_type="exception"),
        PSLRule("fish_in_water", "Fish(x)", "AtLocation(x, water)", weight=0.9, rule_type="default"),
        
        # Capability rules
        PSLRule("capable_of_action", "CapableOf(x, y)", "CanDo(x, y)", weight=0.9, rule_type="default"),
        PSLRule("requires_tool", "Requires(x, y) & Has(y)", "Easier(x)", weight=0.8, rule_type="default"),
        
        # Social commonsense
        PSLRule("help_gratitude", "PersonHelps(x, y)", "GratefulTo(y, x)", weight=0.7, rule_type="default"),
        PSLRule("hurt_anger", "PersonHurts(x, y)", "AngryAt(y, x)", weight=0.8, rule_type="default"),
        
        # Temporal commonsense
        PSLRule("cause_effect", "Causes(x, y)", "HappensAfter(y, x)", weight=0.9, rule_type="default"),
        PSLRule("prerequisite", "Prerequisite(x, y)", "HappensBefore(x, y)", weight=0.9, rule_type="default"),
    ]
    return rules 