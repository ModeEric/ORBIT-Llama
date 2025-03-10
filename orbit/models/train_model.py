#!/usr/bin/env python
"""
Train ORBIT models from the command line.

This script provides a command-line interface for training domain-specific
language models using the ORBIT framework.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from orbit.models.trainer import OrbitTrainer

def parse_additional_args(args_list):
    """Parse additional arguments passed as key=value pairs."""
    result = {}
    for arg in args_list:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # Try to convert to appropriate type
            try:
                # Try as int
                value = int(value)
            except ValueError:
                try:
                    # Try as float
                    value = float(value)
                except ValueError:
                    # Try as boolean
                    if value.lower() in ('true', 'yes', '1'):
                        value = True
                    elif value.lower() in ('false', 'no', '0'):
                        value = False
            result[key] = value
    return result

def main():
    """Train an ORBIT model from the command line."""
    parser = argparse.ArgumentParser(description="Train an ORBIT model")
    
    # Required arguments
    parser.add_argument("--model", "-m", required=True, help="Base model to fine-tune")
    parser.add_argument("--dataset", "-d", required=True, help="Path to dataset")
    
    # Optional arguments
    parser.add_argument("--domain", help="Domain to train for (astronomy, law, medical, etc.)")
    parser.add_argument("--output-dir", "-o", help="Directory to save the trained model")
    parser.add_argument("--config", "-c", help="Path to training configuration file")
    parser.add_argument("--method", choices=["full", "lora", "qlora"], default="lora",
                       help="Training method (full, lora, qlora)")
    
    # Parse known args first
    args, unknown = parser.parse_known_args()
    
    # Parse additional args as key=value pairs
    additional_args = parse_additional_args(unknown)
    
    # Create trainer
    trainer = OrbitTrainer(
        domain=args.domain,
        model_name=args.model,
        config_path=args.config
    )
    
    # Train model
    output_dir = trainer.train(
        base_model=args.model,
        dataset=args.dataset,
        output_dir=args.output_dir,
        method=args.method,
        **additional_args
    )
    
    print(f"\nTraining complete!")
    print(f"Model saved to: {output_dir}")
    print("\nTo evaluate this model, run:")
    
    if args.domain:
        print(f"python orbit/evaluation/run_evaluation.py --model {output_dir} --domain {args.domain}")
    else:
        print(f"python orbit/evaluation/run_evaluation.py --model {output_dir} --benchmark <benchmark_name>")
    
    # Export model if using LoRA
    if args.method in ["lora", "qlora"]:
        print("\nTo export this model (merge LoRA weights), run:")
        print(f"python orbit/models/export_model.py --model {output_dir} --output {output_dir}_merged")

if __name__ == "__main__":
    main() 