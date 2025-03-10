#!/usr/bin/env python
"""
Export ORBIT models for deployment.

This script provides a command-line interface for exporting trained models,
including merging LoRA weights with the base model.
"""

import os
import sys
import argparse

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from orbit.models.trainer import OrbitTrainer

def main():
    """Export an ORBIT model from the command line."""
    parser = argparse.ArgumentParser(description="Export an ORBIT model")
    
    # Required arguments
    parser.add_argument("--model", "-m", required=True, help="Path to trained model")
    
    # Optional arguments
    parser.add_argument("--output", "-o", help="Path to save the exported model")
    parser.add_argument("--no-merge", action="store_true", help="Don't merge LoRA weights")
    parser.add_argument("--domain", help="Domain of the model (for logging)")
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = OrbitTrainer(domain=args.domain)
    
    # Export model
    output_path = trainer.export_model(
        model_path=args.model,
        output_path=args.output,
        merge_lora=not args.no_merge
    )
    
    print(f"\nModel exported successfully!")
    print(f"Exported model saved to: {output_path}")

if __name__ == "__main__":
    main() 