"""
Data preprocessing for different model types.
"""

from typing import List, Dict, Any, Tuple
from .schema import Example, UnifiedSchema


class Preprocessor:
    """Base class for data preprocessing."""
    
    def __init__(self, schema: UnifiedSchema):
        self.schema = schema
    
    def preprocess(self, examples: List[Example]) -> List[Dict[str, Any]]:
        """Preprocess examples for model input."""
        raise NotImplementedError


class EntailmentPreprocessor(Preprocessor):
    """Preprocessor for entailment-style models."""
    
    def __init__(self, schema: UnifiedSchema, tokenizer=None):
        super().__init__(schema)
        self.tokenizer = tokenizer
    
    def preprocess(self, examples: List[Example]) -> List[Dict[str, Any]]:
        """Convert examples to entailment format."""
        processed = []
        
        for example in examples:
            template = self.schema.get_entailment_template(example.dataset)
            
            for i, choice in enumerate(example.choices):
                # Create hypothesis from choice
                if example.context:
                    hypothesis = template.format(
                        context=example.context,
                        question=example.question,
                        choice=choice
                    )
                else:
                    hypothesis = template.format(
                        question=example.question,
                        choice=choice
                    )
                
                # Create premise (can be empty or use question as premise)
                premise = example.question if example.question else ""
                
                processed.append({
                    "id": f"{example.id}_{i}",
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": 1 if i == example.label else 0,  # 1 for correct, 0 for incorrect
                    "original_example": example,
                    "choice_index": i
                })
        
        return processed
    
    def tokenize(self, processed_examples: List[Dict[str, Any]], 
                 max_length: int = 256) -> Dict[str, Any]:
        """Tokenize processed examples."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")
        
        premises = [ex["premise"] for ex in processed_examples]
        hypotheses = [ex["hypothesis"] for ex in processed_examples]
        
        # Tokenize premise-hypothesis pairs
        tokenized = self.tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Add labels
        tokenized["labels"] = [ex["label"] for ex in processed_examples]
        tokenized["example_ids"] = [ex["id"] for ex in processed_examples]
        
        return tokenized


class MultipleChoicePreprocessor(Preprocessor):
    """Preprocessor for multiple choice models."""
    
    def __init__(self, schema: UnifiedSchema, tokenizer=None):
        super().__init__(schema)
        self.tokenizer = tokenizer
    
    def preprocess(self, examples: List[Example]) -> List[Dict[str, Any]]:
        """Convert examples to multiple choice format."""
        processed = []
        
        for example in examples:
            template = self.schema.get_multiple_choice_template(example.dataset)
            
            # Format choices for template
            choice_vars = {}
            for i, choice in enumerate(example.choices):
                if len(example.choices) == 5:  # CommonsenseQA
                    choice_vars[f"choice_{chr(ord('a') + i)}"] = choice
                elif len(example.choices) == 2:  # Winogrande, PIQA
                    choice_vars[f"choice_{i + 1}"] = choice
                elif len(example.choices) == 3:  # SocialIQA
                    choice_vars[f"choice_{chr(ord('a') + i)}"] = choice
            
            # Create formatted text
            if example.context:
                formatted_text = template.format(
                    context=example.context,
                    question=example.question,
                    **choice_vars
                )
            else:
                formatted_text = template.format(
                    question=example.question,
                    **choice_vars
                )
            
            processed.append({
                "id": example.id,
                "text": formatted_text,
                "choices": example.choices,
                "label": example.label,
                "original_example": example
            })
        
        return processed
    
    def tokenize(self, processed_examples: List[Dict[str, Any]], 
                 max_length: int = 256) -> Dict[str, Any]:
        """Tokenize processed examples for multiple choice."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")
        
        texts = [ex["text"] for ex in processed_examples]
        
        # Tokenize all texts
        tokenized = self.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Add labels and choice information
        tokenized["labels"] = [ex["label"] for ex in processed_examples]
        tokenized["example_ids"] = [ex["id"] for ex in processed_examples]
        tokenized["choices"] = [ex["choices"] for ex in processed_examples]
        
        return tokenized


class BiEncoderPreprocessor(Preprocessor):
    """Preprocessor for bi-encoder retrieval models."""
    
    def __init__(self, schema: UnifiedSchema, tokenizer=None):
        super().__init__(schema)
        self.tokenizer = tokenizer
    
    def preprocess(self, examples: List[Example]) -> List[Dict[str, Any]]:
        """Convert examples to bi-encoder format."""
        processed = []
        
        for example in examples:
            # Create query from question and context
            if example.context:
                query = f"{example.context} {example.question}"
            else:
                query = example.question
            
            # Create candidate texts from choices
            candidates = example.choices
            
            processed.append({
                "id": example.id,
                "query": query,
                "candidates": candidates,
                "label": example.label,
                "original_example": example
            })
        
        return processed
    
    def tokenize(self, processed_examples: List[Dict[str, Any]], 
                 max_length: int = 128) -> Dict[str, Any]:
        """Tokenize for bi-encoder."""
        if not self.tokenizer:
            raise ValueError("Tokenizer not provided")
        
        # Tokenize queries
        queries = [ex["query"] for ex in processed_examples]
        query_tokens = self.tokenizer(
            text=queries,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        # Tokenize all candidates
        all_candidates = []
        candidate_mapping = []
        
        for i, ex in enumerate(processed_examples):
            for j, candidate in enumerate(ex["candidates"]):
                all_candidates.append(candidate)
                candidate_mapping.append((i, j))
        
        candidate_tokens = self.tokenizer(
            text=all_candidates,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "query_input_ids": query_tokens["input_ids"],
            "query_attention_mask": query_tokens["attention_mask"],
            "candidate_input_ids": candidate_tokens["input_ids"],
            "candidate_attention_mask": candidate_tokens["attention_mask"],
            "labels": [ex["label"] for ex in processed_examples],
            "example_ids": [ex["id"] for ex in processed_examples],
            "candidate_mapping": candidate_mapping
        } 