#!/usr/bin/env python3
"""
Script to download and prepare datasets.
"""

import argparse
import os
from datasets import load_dataset
import json


def download_dataset(dataset_name: str, output_dir: str = "data/benchmarks"):
    """Download a specific dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading {dataset_name}...")
    
    if dataset_name == "commonsenseqa":
        dataset = load_dataset("commonsense_qa")
    elif dataset_name == "winogrande":
        dataset = load_dataset("winogrande", "winogrande_xl")
    elif dataset_name == "piqa":
        dataset = load_dataset("piqa")
    elif dataset_name == "socialiqa":
        dataset = load_dataset("social_i_qa")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Save dataset info
    info = {
        "name": dataset_name,
        "splits": list(dataset.keys()),
        "features": dataset["train"].features if "train" in dataset else list(dataset.keys())[0]
    }
    
    with open(os.path.join(output_dir, f"{dataset_name}_info.json"), 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Downloaded {dataset_name} successfully!")
    print(f"Available splits: {list(dataset.keys())}")
    
    # Print some statistics
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Download datasets")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["commonsenseqa", "winogrande", "piqa", "socialiqa"],
                       default=["commonsenseqa", "winogrande", "piqa"],
                       help="Datasets to download")
    parser.add_argument("--output_dir", type=str, default="data/benchmarks",
                       help="Output directory")
    
    args = parser.parse_args()
    
    for dataset in args.datasets:
        download_dataset(dataset, args.output_dir)
    
    print("All datasets downloaded successfully!")


if __name__ == "__main__":
    main() 