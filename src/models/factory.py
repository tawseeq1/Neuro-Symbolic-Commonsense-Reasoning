"""
Model factory for creating different types of neural models.
"""

from typing import Dict, Any, Type
from .base import BaseModel
from .entailment import EntailmentModel
from .multiple_choice import MultipleChoiceModel
from .bi_encoder import BiEncoderModel


class ModelFactory:
    """Factory for creating neural models."""
    
    _models: Dict[str, Type[BaseModel]] = {
        "entailment": EntailmentModel,
        "multiple_choice": MultipleChoiceModel,
        "bi_encoder": BiEncoderModel
    }
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseModel]):
        """Register a new model type."""
        cls._models[name] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, model_name: str, **kwargs) -> BaseModel:
        """Create a model instance."""
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(cls._models.keys())}")
        
        model_class = cls._models[model_type]
        return model_class(model_name=model_name, **kwargs)
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """Get all available model types."""
        return cls._models.copy()
    
    @classmethod
    def create_from_config(cls, config: Dict[str, Any]) -> BaseModel:
        """Create model from configuration dictionary."""
        model_type = config.get("type", "entailment")
        model_name = config.get("model_name", "roberta-base")
        
        # Remove type from kwargs
        kwargs = {k: v for k, v in config.items() if k != "type"}
        
        return cls.create_model(model_type, model_name, **kwargs) 