#!/usr/bin/env python
"""
Run the ORBIT curation pipeline.

This script provides a simple way to run the complete ORBIT curation pipeline
without having to worry about Python module imports.
"""

import os
import sys
import argparse
import subprocess

def run_command(command):
    """Run a shell command and print output."""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, check=False)
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
    return result.returncode

def main():
    """Run the ORBIT curation pipeline."""
    parser = argparse.ArgumentParser(description="Run the ORBIT curation pipeline")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4], default=1,
                       help="Pipeline stage to run (1=generate, 2=label, 3=train, 4=process)")
    parser.add_argument("--samples", type=int, default=100,
                       help="Number of samples to generate")
    parser.add_argument("--embedding", help="Path to FastText embedding model")
    parser.add_argument("--quality-model", help="Path to quality evaluation model")
    parser.add_argument("--model-type", choices=["simple", "transformer"], default="simple",
                       help="Type of quality model to train")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of epochs for training")
    args = parser.parse_args()
    
    # Set environment variable to add project root to Python path
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    
    if args.stage <= 1:
        # Stage 1: Generate sample data
        cmd = f"python orbit/datasets/generate_sample_data.py --output domain_filtered_data.jsonl --samples {args.samples}"
        if run_command(cmd) != 0:
            return
    
    if args.stage <= 2:
        # Stage 2: Label data for quality evaluation
        cmd = "python orbit/datasets/stage2_label_data.py --input domain_filtered_data.jsonl --output labeled_data.jsonl --method heuristic"
        if run_command(cmd) != 0:
            return
    
    if args.stage <= 3:
        # Stage 3: Train quality evaluation model
        cmd = f"python orbit/datasets/stage2_train_quality_model.py --train labeled_data.jsonl --output quality_model --epochs {args.epochs} --model-type {args.model_type}"
        if run_command(cmd) != 0:
            return
    
    if args.stage <= 4:
        # Stage 4: Process dataset with full pipeline
        cmd = "python test_astro_processor.py"
        if args.embedding:
            cmd += f" --embedding {args.embedding}"
        if args.quality_model:
            cmd += f" --quality-model {args.quality_model} --evaluate-quality"
        else:
            cmd += " --quality-model quality_model --evaluate-quality"
        cmd += " --input domain_filtered_data.jsonl"
        run_command(cmd)

if __name__ == "__main__":
    main() 