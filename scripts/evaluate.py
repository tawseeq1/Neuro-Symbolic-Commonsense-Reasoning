#!/usr/bin/env python3
"""
Evaluation script for neuro-symbolic models.
"""

import argparse
import torch
import json
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.factory import ModelFactory
from rules.psl import PSLReasoner, create_default_rules
from rules.knowledge import ConceptNetKnowledge, KnowledgeRetriever
from reasoning.fusion import PipelineFusion, JointFusion
from training.trainer import Trainer
from evaluation.metrics import compute_metrics


def load_checkpoint(checkpoint_path: str):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint


def setup_model_from_checkpoint(checkpoint_path: str):
    """Setup model from checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    
    # Create model
    model = ModelFactory.create_from_config(config)
    
    # Load state
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model, config


def setup_fusion_from_checkpoint(checkpoint_path: str, model):
    """Setup fusion model from checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    
    # Setup symbolic components
    rules = create_default_rules()
    psl_reasoner = PSLReasoner(rules, t_norm="lukasiewicz")
    
    knowledge_retriever = None
    if config.get("use_knowledge", False):
        kg = ConceptNetKnowledge()
        knowledge_retriever = KnowledgeRetriever(kg)
    
    # Setup fusion
    fusion_type = config.get("fusion_type", "pipeline")
    if fusion_type == "pipeline":
        fusion_model = PipelineFusion(
            neural_model=model,
            symbolic_reasoner=psl_reasoner,
            knowledge_retriever=knowledge_retriever,
            alpha=config.get("alpha", 0.5)
        )
    elif fusion_type == "joint":
        fusion_model = JointFusion(
            neural_model=model,
            symbolic_reasoner=psl_reasoner,
            knowledge_retriever=knowledge_retriever,
            lambda_logic=config.get("lambda_logic", 0.1),
            alpha=config.get("alpha", 0.5)
        )
    else:
        fusion_model = None
    
    if fusion_model and "fusion_model_state_dict" in checkpoint:
        fusion_model.load_state_dict(checkpoint["fusion_model_state_dict"])
    
    return fusion_model


def main():
    parser = argparse.ArgumentParser(description="Evaluate neuro-symbolic model")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="reports/evaluation.json",
                       help="Output file for evaluation results")
    parser.add_argument("--dataset", type=str, required=True,
                       choices=["commonsenseqa", "winogrande", "piqa", "socialiqa"],
                       help="Dataset to evaluate on")
    
    args = parser.parse_args()
    
    # Setup model
    print("Loading model from checkpoint...")
    model, config = setup_model_from_checkpoint(args.checkpoint)
    
    # Setup fusion model
    fusion_model = setup_fusion_from_checkpoint(args.checkpoint, model)
    
    # Setup trainer for evaluation
    trainer = Trainer(
        model=model,
        fusion_model=fusion_model,
        config=config
    )
    
    # Evaluate
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    # Save results
    results = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "metrics": metrics,
        "config": config
    }
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Evaluation results:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main() 