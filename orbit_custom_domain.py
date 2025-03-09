#!/usr/bin/env python
"""
ORBIT Custom Domain Processor

This script allows users to define custom domains by providing a list of keywords,
and then process datasets through the ORBIT curation pipeline for that domain.
"""

import os
import sys
import json
import argparse
from typing import List
from pprint import pprint

# Add the project root to the path so we can import the orbit package
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from orbit.datasets.custom import CustomDomainProcessor

def load_keywords(keywords_file: str) -> List[str]:
    """
    Load keywords from a file.
    
    Args:
        keywords_file: Path to keywords file (one keyword per line)
        
    Returns:
        List of keywords
    """
    keywords = []
    with open(keywords_file, 'r', encoding='utf-8') as f:
        for line in f:
            keyword = line.strip()
            if keyword and not keyword.startswith('#'):
                keywords.append(keyword)
    
    return keywords

def main():
    """Run the custom domain processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ORBIT Custom Domain Processor")
    parser.add_argument("--domain", "-d", required=True, help="Name of the custom domain")
    parser.add_argument("--keywords", "-k", required=True, 
                       help="Path to keywords file (one keyword per line)")
    parser.add_argument("--input", "-i", required=True, help="Input file to process")
    parser.add_argument("--output-dir", "-o", help="Output directory for processed data")
    parser.add_argument("--embedding", "-e", help="Path to FastText embedding model")
    parser.add_argument("--mapping", "-m", help="Path to token similarity mapping")
    parser.add_argument("--quality-model", "-q", help="Path to quality evaluation model")
    parser.add_argument("--evaluate-quality", "-eq", action="store_true", 
                       help="Evaluate document quality")
    parser.add_argument("--quality-threshold", "-qt", type=float, default=0.5,
                       help="Quality threshold (0-1)")
    parser.add_argument("--model-type", "-mt", choices=["simple", "transformer"], 
                       default="simple", help="Type of quality model")
    parser.add_argument("--workers", "-w", type=int, default=1,
                       help="Number of worker processes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        # Load keywords
        print(f"Loading keywords from {args.keywords}...")
        keywords = load_keywords(args.keywords)
        print(f"Loaded {len(keywords)} keywords for domain '{args.domain}'")
        
        if len(keywords) < 10:
            print("Warning: It's recommended to provide at least 10-20 keywords for effective domain filtering.")
        
        # Initialize the custom domain processor
        processor = CustomDomainProcessor(
            domain_name=args.domain,
            keywords=keywords,
            sim_mapping_path=args.mapping,
            embedding_path=args.embedding,
            quality_threshold=args.quality_threshold,
            quality_model_type=args.model_type
        )
        
        print(f"\nORBIT Custom Domain Processor: {args.domain}")
        print("=" * (30 + len(args.domain)))
        
        if args.embedding:
            print(f"Using embedding model from: {args.embedding}")
        if args.mapping:
            print(f"Using token similarity mapping from: {args.mapping}")
        if args.quality_model:
            print(f"Using quality evaluation model from: {args.quality_model}")
        print()
        
        # Process the dataset
        print(f"Processing file: {args.input}")
        
        # Create output directory
        output_dir = args.output_dir
        if not output_dir:
            output_dir = f"{args.domain}_processed"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Process the dataset
        stats = processor.process_dataset(
            input_path=args.input,
            output_dir=output_dir,
            evaluate_quality=args.evaluate_quality,
            quality_model_path=args.quality_model,
            quality_threshold=args.quality_threshold,
            quality_model_type=args.model_type,
            num_workers=args.workers
        )
        
        print("\nProcessing Statistics:")
        print("---------------------")
        pprint(stats)
        
        print(f"\nResults saved to: {output_dir}")
        print(f"Final dataset: {output_dir}/final_{args.domain}_dataset.jsonl")
        
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 