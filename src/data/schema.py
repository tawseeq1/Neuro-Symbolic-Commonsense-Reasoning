"""
Unified data schema for commonsense reasoning datasets.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


@dataclass
class Example:
    """Unified example format for all datasets."""
    id: str
    context: Optional[str] = None
    question: str
    choices: List[str]
    label: int
    dataset: str
    split: str
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.label >= len(self.choices):
            raise ValueError(f"Label {self.label} is out of range for {len(self.choices)} choices")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "id": self.id,
            "context": self.context,
            "question": self.question,
            "choices": self.choices,
            "label": self.label,
            "dataset": self.dataset,
            "split": self.split,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Example":
        """Create from dictionary format."""
        return cls(**data)


class UnifiedSchema(BaseModel):
    """Unified schema configuration for dataset preprocessing."""
    
    # Dataset-specific configurations
    commonsenseqa: Dict[str, Any] = {
        "context_field": None,
        "question_field": "question",
        "choices_field": "choices",
        "label_field": "label",
        "num_choices": 5
    }
    
    winogrande: Dict[str, Any] = {
        "context_field": "sentence",
        "question_field": None,
        "choices_field": ["option1", "option2"],
        "label_field": "answer",
        "num_choices": 2
    }
    
    piqa: Dict[str, Any] = {
        "context_field": None,
        "question_field": "goal",
        "choices_field": ["sol1", "sol2"],
        "label_field": "label",
        "num_choices": 2
    }
    
    socialiqa: Dict[str, Any] = {
        "context_field": "context",
        "question_field": "question",
        "choices_field": ["answerA", "answerB", "answerC"],
        "label_field": "correct",
        "num_choices": 3
    }
    
    # Entailment templates for converting to entailment format
    entailment_templates: Dict[str, str] = {
        "commonsenseqa": "Question: {question} Answer: {choice}",
        "winogrande": "In the sentence '{context}', the pronoun refers to {choice}",
        "piqa": "To achieve the goal '{question}', the solution is {choice}",
        "socialiqa": "Given the context '{context}', the answer to '{question}' is {choice}"
    }
    
    # Multiple choice templates
    multiple_choice_templates: Dict[str, str] = {
        "commonsenseqa": "Question: {question}\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}\nD) {choice_d}\nE) {choice_e}",
        "winogrande": "Sentence: {context}\nOption 1: {choice_1}\nOption 2: {choice_2}",
        "piqa": "Goal: {question}\nSolution 1: {choice_1}\nSolution 2: {choice_2}",
        "socialiqa": "Context: {context}\nQuestion: {question}\nA) {choice_a}\nB) {choice_b}\nC) {choice_c}"
    }
    
    def get_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset."""
        if dataset_name not in self.__dict__:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        return self.__dict__[dataset_name]
    
    def get_entailment_template(self, dataset_name: str) -> str:
        """Get entailment template for a dataset."""
        return self.entailment_templates.get(dataset_name, "{question} {choice}")
    
    def get_multiple_choice_template(self, dataset_name: str) -> str:
        """Get multiple choice template for a dataset."""
        return self.multiple_choice_templates.get(dataset_name, "{question}\n{choices}") 