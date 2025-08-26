"""
Dataset loaders for commonsense reasoning benchmarks.
"""

import os
from typing import List, Dict, Any, Optional
from datasets import load_dataset, Dataset
from .schema import Example, UnifiedSchema


class DatasetLoader:
    """Base class for dataset loaders."""
    
    def __init__(self, schema: UnifiedSchema):
        self.schema = schema
    
    def load(self, split: str = "train") -> List[Example]:
        """Load dataset split and convert to unified format."""
        raise NotImplementedError
    
    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        raise NotImplementedError


class CommonsenseQALoader(DatasetLoader):
    """Loader for CommonsenseQA dataset."""
    
    def __init__(self, schema: UnifiedSchema):
        super().__init__(schema)
        self.config = schema.get_config("commonsenseqa")
    
    def load(self, split: str = "train") -> List[Example]:
        """Load CommonsenseQA dataset."""
        dataset = load_dataset("commonsense_qa", split=split)
        examples = []
        
        for item in dataset:
            # Convert label from 1-indexed to 0-indexed
            label = item[self.config["label_field"]] - 1
            
            example = Example(
                id=item["id"],
                context=None,
                question=item[self.config["question_field"]],
                choices=item[self.config["choices_field"]],
                label=label,
                dataset="commonsenseqa",
                split=split,
                metadata={"question_concept": item.get("question_concept", "")}
            )
            examples.append(example)
        
        return examples
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


class WinograndeLoader(DatasetLoader):
    """Loader for Winogrande dataset."""
    
    def __init__(self, schema: UnifiedSchema):
        super().__init__(schema)
        self.config = schema.get_config("winogrande")
    
    def load(self, split: str = "train") -> List[Example]:
        """Load Winogrande dataset."""
        dataset = load_dataset("winogrande", "winogrande_xl", split=split)
        examples = []
        
        for item in dataset:
            # Convert label from 1-indexed to 0-indexed
            label = item[self.config["label_field"]] - 1
            
            example = Example(
                id=item["id"],
                context=item[self.config["context_field"]],
                question=None,
                choices=[item[self.config["choices_field"][0]], 
                        item[self.config["choices_field"][1]]],
                label=label,
                dataset="winogrande",
                split=split,
                metadata={"pronoun": item.get("pronoun", "")}
            )
            examples.append(example)
        
        return examples
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


class PIQALoader(DatasetLoader):
    """Loader for PIQA dataset."""
    
    def __init__(self, schema: UnifiedSchema):
        super().__init__(schema)
        self.config = schema.get_config("piqa")
    
    def load(self, split: str = "train") -> List[Example]:
        """Load PIQA dataset."""
        dataset = load_dataset("piqa", split=split)
        examples = []
        
        for item in dataset:
            example = Example(
                id=item["id"],
                context=None,
                question=item[self.config["question_field"]],
                choices=[item[self.config["choices_field"][0]], 
                        item[self.config["choices_field"][1]]],
                label=item[self.config["label_field"]],
                dataset="piqa",
                split=split,
                metadata={"goal": item.get("goal", "")}
            )
            examples.append(example)
        
        return examples
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


class SocialIQALoader(DatasetLoader):
    """Loader for SocialIQA dataset."""
    
    def __init__(self, schema: UnifiedSchema):
        super().__init__(schema)
        self.config = schema.get_config("socialiqa")
    
    def load(self, split: str = "train") -> List[Example]:
        """Load SocialIQA dataset."""
        dataset = load_dataset("social_i_qa", split=split)
        examples = []
        
        for item in dataset:
            # Convert label from 1-indexed to 0-indexed
            label = item[self.config["label_field"]] - 1
            
            example = Example(
                id=item["id"],
                context=item[self.config["context_field"]],
                question=item[self.config["question_field"]],
                choices=[item[self.config["choices_field"][0]], 
                        item[self.config["choices_field"][1]],
                        item[self.config["choices_field"][2]]],
                label=label,
                dataset="socialiqa",
                split=split,
                metadata={"context": item.get("context", "")}
            )
            examples.append(example)
        
        return examples
    
    def get_config(self) -> Dict[str, Any]:
        return self.config


def get_dataset_loader(dataset_name: str, schema: UnifiedSchema) -> DatasetLoader:
    """Factory function to get appropriate dataset loader."""
    loaders = {
        "commonsenseqa": CommonsenseQALoader,
        "winogrande": WinograndeLoader,
        "piqa": PIQALoader,
        "socialiqa": SocialIQALoader
    }
    
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return loaders[dataset_name](schema) 