"""
Evaluation metrics for neuro-symbolic models.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch


def compute_metrics(predictions: List[int], labels: List[int], 
                   probabilities: Optional[List[List[float]]] = None) -> Dict[str, float]:
    """Compute evaluation metrics."""
    metrics = {}
    
    # Basic classification metrics
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["f1_macro"] = f1_score(labels, predictions, average="macro")
    metrics["f1_weighted"] = f1_score(labels, predictions, average="weighted")
    
    # Per-class F1 scores
    f1_scores = f1_score(labels, predictions, average=None)
    for i, f1 in enumerate(f1_scores):
        metrics[f"f1_class_{i}"] = f1
    
    # Expected Calibration Error if probabilities provided
    if probabilities is not None:
        ece = compute_ece(probabilities, labels)
        metrics["ece"] = ece
    
    return metrics


def compute_ece(probabilities: List[List[float]], labels: List[int], 
                n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    if not probabilities:
        return 0.0
    
    # Convert to numpy arrays
    probs = np.array(probabilities)
    true_labels = np.array(labels)
    
    # Get confidence and predictions
    confidence = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = np.logical_and(confidence > bin_lower, confidence <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Calculate accuracy and confidence for this bin
            bin_accuracy = np.sum(predictions[in_bin] == true_labels[in_bin]) / bin_size
            bin_confidence = np.mean(confidence[in_bin])
            
            # Add to ECE
            ece += bin_size * np.abs(bin_accuracy - bin_confidence)
    
    return ece / len(probabilities)


def compute_rule_satisfaction(rules: List[Dict[str, Any]], 
                             facts: Dict[str, float]) -> Dict[str, float]:
    """Compute rule satisfaction rates."""
    satisfaction_rates = {}
    
    for rule in rules:
        rule_name = rule["name"]
        antecedent = rule["antecedent"]
        consequent = rule["consequent"]
        weight = rule.get("weight", 1.0)
        
        # Check if facts support this rule
        antecedent_satisfied = antecedent in facts
        consequent_satisfied = consequent in facts
        
        if antecedent_satisfied and consequent_satisfied:
            # Both antecedent and consequent are present
            satisfaction = 1.0
        elif antecedent_satisfied and not consequent_satisfied:
            # Antecedent present but consequent not - rule violation
            satisfaction = 0.0
        else:
            # Antecedent not present - rule not applicable
            satisfaction = 1.0
        
        satisfaction_rates[rule_name] = satisfaction * weight
    
    return satisfaction_rates


def compute_confusion_matrix(predictions: List[int], labels: List[int], 
                           num_classes: int) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for pred, true in zip(predictions, labels):
        cm[true, pred] += 1
    
    return cm


def analyze_errors(predictions: List[int], labels: List[int], 
                  examples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze prediction errors."""
    errors = []
    
    for i, (pred, true) in enumerate(zip(predictions, labels)):
        if pred != true:
            error = {
                "index": i,
                "prediction": pred,
                "true_label": true,
                "example": examples[i] if i < len(examples) else None
            }
            errors.append(error)
    
    # Group errors by type
    error_types = {}
    for error in errors:
        error_type = f"{error['true_label']}_to_{error['prediction']}"
        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(error)
    
    return {
        "total_errors": len(errors),
        "error_rate": len(errors) / len(predictions),
        "error_types": error_types,
        "errors": errors
    } 