#!/usr/bin/env python
"""
Train a quality evaluation model for ORBIT Stage 2 filtering.

This script trains a model to classify documents as high or low quality
based on labeled training data. It supports both transformer-based models
and simpler TF-IDF + Logistic Regression models.
"""

import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
import random

from orbit.datasets.curator import QualityEvaluator

def load_labeled_data(input_path: str) -> List[Dict[str, Any]]:
    """
    Load labeled data from a JSONL file.
    
    Args:
        input_path: Path to labeled data file
        
    Returns:
        List of documents with text and quality labels
    """
    documents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(input_path)}"):
            try:
                doc = json.loads(line)
                if "text" in doc and "quality" in doc:
                    documents.append(doc)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(documents)} labeled documents")
    return documents

def main():
    """Train a quality evaluation model."""
    parser = argparse.ArgumentParser(description="Train a quality evaluation model")
    parser.add_argument("--train", "-t", required=True, help="Path to training data")
    parser.add_argument("--val", "-v", help="Path to validation data (optional)")
    parser.add_argument("--output", "-o", default="quality_model", help="Output directory")
    parser.add_argument("--model-type", "-m", choices=["simple", "transformer"], default="simple",
                       help="Type of model to train (simple=TF-IDF+LogReg, transformer=DistilBERT)")
    parser.add_argument("--base-model", "-b", default="distilbert-base-uncased", 
                       help="Base model for transformer (ignored for simple model)")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", "-bs", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--threshold", "-th", type=float, default=0.5, help="Quality threshold")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load training data
    train_data = load_labeled_data(args.train)
    
    # Load validation data if provided
    val_data = None
    if args.val:
        val_data = load_labeled_data(args.val)
    
    # Initialize quality evaluator
    evaluator = QualityEvaluator(quality_threshold=args.threshold, model_type=args.model_type)
    
    # Train model
    model_path = evaluator.train_model(
        train_data=train_data,
        val_data=val_data,
        output_dir=args.output,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_type=args.model_type
    )
    
    if model_path:
        print(f"Model training complete. Model saved to: {model_path}")
    else:
        print("Model training failed.")

if __name__ == "__main__":
    main() 