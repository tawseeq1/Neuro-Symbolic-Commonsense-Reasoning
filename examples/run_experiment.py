#!/usr/bin/env python3
"""
Example script demonstrating the neuro-symbolic commonsense reasoning pipeline.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import torch
import yaml
import json
from datetime import datetime

from data.schema import UnifiedSchema
from data.datasets import get_dataset_loader
from data.preprocessing import EntailmentPreprocessor
from models.factory import ModelFactory
from rules.psl import PSLReasoner, create_default_rules
from rules.knowledge import ConceptNetKnowledge, KnowledgeRetriever
from reasoning.fusion import PipelineFusion, JointFusion
from training.trainer import Trainer
from evaluation.metrics import compute_metrics


def run_experiment(dataset_name: str = "commonsenseqa", 
                  model_name: str = "roberta-base",
                  fusion_type: str = "pipeline",
                  use_knowledge: bool = True,
                  num_epochs: int = 3):
    """Run a complete neuro-symbolic experiment."""
    
    print(f"Running experiment: {dataset_name} + {model_name} + {fusion_type}")
    print(f"Knowledge retrieval: {use_knowledge}")
    print(f"Epochs: {num_epochs}")
    print("-" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # 1. Setup data
    print("1. Setting up data...")
    schema = UnifiedSchema()
    dataset_loader = get_dataset_loader(dataset_name, schema)
    
    # Load a small subset for demonstration
    train_examples = dataset_loader.load("train")[:100]  # Small subset
    val_examples = dataset_loader.load("validation")[:50]
    
    # Setup preprocessor
    preprocessor = EntailmentPreprocessor(schema)
    train_processed = preprocessor.preprocess(train_examples)
    val_processed = preprocessor.preprocess(val_examples)
    
    print(f"   Train examples: {len(train_processed)}")
    print(f"   Val examples: {len(val_processed)}")
    
    # 2. Setup neural model
    print("2. Setting up neural model...")
    model_config = {
        "model_name": model_name,
        "type": "entailment",
        "num_labels": 2
    }
    neural_model = ModelFactory.create_from_config(model_config)
    print(f"   Model: {neural_model.get_model_info()}")
    
    # 3. Setup symbolic components
    print("3. Setting up symbolic reasoning...")
    rules = create_default_rules()
    psl_reasoner = PSLReasoner(rules, t_norm="lukasiewicz")
    
    knowledge_retriever = None
    if use_knowledge:
        # Create a simple knowledge graph for demonstration
        kg = ConceptNetKnowledge()
        kg.add_triple("bird", "can_fly", "true", 0.8)
        kg.add_triple("penguin", "can_fly", "false", 1.0)
        kg.add_triple("fish", "lives_in", "water", 0.9)
        knowledge_retriever = KnowledgeRetriever(kg)
        print(f"   Knowledge graph: {len(kg.triples)} triples")
    
    # 4. Setup fusion
    print("4. Setting up neuro-symbolic fusion...")
    if fusion_type == "pipeline":
        fusion_model = PipelineFusion(
            neural_model=neural_model,
            symbolic_reasoner=psl_reasoner,
            knowledge_retriever=knowledge_retriever,
            alpha=0.5
        )
    elif fusion_type == "joint":
        fusion_model = JointFusion(
            neural_model=neural_model,
            symbolic_reasoner=psl_reasoner,
            knowledge_retriever=knowledge_retriever,
            lambda_logic=0.1,
            alpha=0.5
        )
    else:
        fusion_model = None
    
    # 5. Setup training
    print("5. Setting up training...")
    config = {
        "training": {
            "batch_size": 8,
            "learning_rate": 2e-5,
            "num_epochs": num_epochs
        },
        "optimizer": {
            "type": "adamw",
            "lr": 2e-5,
            "weight_decay": 0.01
        },
        "scheduler": {
            "type": "cosine",
            "warmup_steps": 10
        },
        "use_wandb": False
    }
    
    # Create simple dataloaders (in practice, you'd want proper Dataset classes)
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(train_processed, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_processed, batch_size=8, shuffle=False)
    
    trainer = Trainer(
        model=neural_model,
        fusion_model=fusion_model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config
    )
    
    # 6. Train
    print("6. Training...")
    history = trainer.train(num_epochs, save_dir="checkpoints")
    
    # 7. Evaluate
    print("7. Evaluating...")
    metrics = trainer.evaluate(val_dataloader)
    
    # 8. Save results
    print("8. Saving results...")
    results = {
        "experiment": {
            "dataset": dataset_name,
            "model": model_name,
            "fusion_type": fusion_type,
            "use_knowledge": use_knowledge,
            "num_epochs": num_epochs,
            "timestamp": datetime.now().isoformat()
        },
        "metrics": metrics,
        "training_history": history
    }
    
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/experiment_{dataset_name}_{model_name}_{fusion_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("9. Results:")
    for key, value in metrics.items():
        print(f"   {key}: {value:.4f}")
    
    print(f"\nExperiment completed! Results saved to reports/experiment_{dataset_name}_{model_name}_{fusion_type}.json")
    
    return results


def main():
    """Run multiple experiments for comparison."""
    
    # Experiment configurations
    experiments = [
        {
            "dataset": "commonsenseqa",
            "model": "roberta-base",
            "fusion": "none",
            "knowledge": False,
            "epochs": 2
        },
        {
            "dataset": "commonsenseqa", 
            "model": "roberta-base",
            "fusion": "pipeline",
            "knowledge": True,
            "epochs": 2
        },
        {
            "dataset": "commonsenseqa",
            "model": "roberta-base", 
            "fusion": "joint",
            "knowledge": True,
            "epochs": 2
        }
    ]
    
    results = {}
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}")
        print(f"{'='*60}")
        
        try:
            result = run_experiment(
                dataset_name=exp["dataset"],
                model_name=exp["model"],
                fusion_type=exp["fusion"],
                use_knowledge=exp["knowledge"],
                num_epochs=exp["epochs"]
            )
            results[f"exp_{i+1}"] = result
        except Exception as e:
            print(f"Experiment {i+1} failed: {e}")
            results[f"exp_{i+1}"] = {"error": str(e)}
    
    # Save all results
    with open("reports/all_experiments.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETED")
    print(f"{'='*60}")
    print("Summary:")
    for i, exp in enumerate(experiments):
        exp_key = f"exp_{i+1}"
        if exp_key in results and "error" not in results[exp_key]:
            metrics = results[exp_key]["metrics"]
            print(f"  {exp['fusion']} + {'KG' if exp['knowledge'] else 'no-KG'}: "
                  f"acc={metrics.get('accuracy', 0):.3f}, "
                  f"f1={metrics.get('f1_macro', 0):.3f}")
        else:
            print(f"  {exp['fusion']} + {'KG' if exp['knowledge'] else 'no-KG'}: FAILED")


if __name__ == "__main__":
    main() 