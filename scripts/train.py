#!/usr/bin/env python3
"""
Main training script for neuro-symbolic commonsense reasoning.
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.schema import UnifiedSchema
from data.datasets import get_dataset_loader
from data.preprocessing import EntailmentPreprocessor, MultipleChoicePreprocessor
from models.factory import ModelFactory
from rules.psl import PSLReasoner, create_default_rules
from rules.knowledge import ConceptNetKnowledge, KnowledgeRetriever
from reasoning.fusion import PipelineFusion, JointFusion
from training.trainer import Trainer
from torch.utils.data import DataLoader


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_data(config: dict, dataset_name: str):
    """Setup data loaders."""
    # Load schema
    schema = UnifiedSchema()
    
    # Get dataset loader
    dataset_loader = get_dataset_loader(dataset_name, schema)
    
    # Load data
    train_examples = dataset_loader.load("train")
    val_examples = dataset_loader.load("validation")
    test_examples = dataset_loader.load("test")
    
    # Setup preprocessor
    model_type = config.get("type", "entailment")
    if model_type == "entailment":
        preprocessor = EntailmentPreprocessor(schema)
    elif model_type == "multiple_choice":
        preprocessor = MultipleChoicePreprocessor(schema)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Preprocess data
    train_processed = preprocessor.preprocess(train_examples)
    val_processed = preprocessor.preprocess(val_examples)
    test_processed = preprocessor.preprocess(test_examples)
    
    # Create datasets (simplified - in practice you'd want a proper Dataset class)
    train_dataset = train_processed
    val_dataset = val_processed
    test_dataset = test_processed
    
    # Create dataloaders
    batch_size = config.get("training", {}).get("batch_size", 16)
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: x  # Simplified
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: x  # Simplified
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: x  # Simplified
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def setup_model(config: dict):
    """Setup neural model."""
    return ModelFactory.create_from_config(config)


def setup_symbolic(config: dict):
    """Setup symbolic reasoning components."""
    # Create PSL rules
    rules = create_default_rules()
    psl_reasoner = PSLReasoner(rules, t_norm="lukasiewicz")
    
    # Setup knowledge graph (optional)
    knowledge_retriever = None
    if config.get("use_knowledge", False):
        kg = ConceptNetKnowledge()
        knowledge_retriever = KnowledgeRetriever(kg)
    
    return psl_reasoner, knowledge_retriever


def setup_fusion(config: dict, neural_model, symbolic_reasoner, knowledge_retriever):
    """Setup neuro-symbolic fusion."""
    fusion_type = config.get("fusion_type", "pipeline")
    
    if fusion_type == "pipeline":
        return PipelineFusion(
            neural_model=neural_model,
            symbolic_reasoner=symbolic_reasoner,
            knowledge_retriever=knowledge_retriever,
            alpha=config.get("alpha", 0.5)
        )
    elif fusion_type == "joint":
        return JointFusion(
            neural_model=neural_model,
            symbolic_reasoner=symbolic_reasoner,
            knowledge_retriever=knowledge_retriever,
            lambda_logic=config.get("lambda_logic", 0.1),
            alpha=config.get("alpha", 0.5)
        )
    else:
        return None


def main():
    parser = argparse.ArgumentParser(description="Train neuro-symbolic model")
    parser.add_argument("--dataset", type=str, required=True, 
                       choices=["commonsenseqa", "winogrande", "piqa", "socialiqa"],
                       help="Dataset to train on")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to model configuration file")
    parser.add_argument("--symbolic", type=str, choices=["psl", "none"], default="none",
                       help="Symbolic reasoning method")
    parser.add_argument("--fusion", type=str, choices=["pipeline", "joint", "none"], default="none",
                       help="Fusion strategy")
    parser.add_argument("--lambda_logic", type=float, default=0.1,
                       help="Logic regularization weight")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Symbolic score interpolation weight")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config["fusion_type"] = args.fusion
    config["lambda_logic"] = args.lambda_logic
    config["alpha"] = args.alpha
    config["use_knowledge"] = args.symbolic != "none"
    
    # Setup data
    print(f"Setting up data for {args.dataset}...")
    train_dataloader, val_dataloader, test_dataloader = setup_data(config, args.dataset)
    
    # Setup model
    print("Setting up neural model...")
    neural_model = setup_model(config)
    
    # Setup symbolic components
    fusion_model = None
    if args.symbolic != "none":
        print("Setting up symbolic reasoning...")
        symbolic_reasoner, knowledge_retriever = setup_symbolic(config)
        
        if args.fusion != "none":
            print(f"Setting up {args.fusion} fusion...")
            fusion_model = setup_fusion(config, neural_model, symbolic_reasoner, knowledge_retriever)
    
    # Setup trainer
    print("Setting up trainer...")
    trainer = Trainer(
        model=neural_model,
        fusion_model=fusion_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        config=config
    )
    
    # Train
    print("Starting training...")
    num_epochs = config.get("training", {}).get("num_epochs", 10)
    history = trainer.train(num_epochs, save_dir=args.output_dir)
    
    # Evaluate
    print("Evaluating...")
    metrics = trainer.evaluate()
    print("Final metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("Training completed!")


if __name__ == "__main__":
    main() 