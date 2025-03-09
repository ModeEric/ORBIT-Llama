#!/usr/bin/env python
"""
Test script for the ORBIT Law Dataset Processor.

This script demonstrates the basic functionality of the LawProcessor class
by analyzing a sample law text and showing the results.
"""

import os
import sys
import json
import argparse
from pprint import pprint

# Add the project root to the path so we can import the orbit package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit.datasets.law import LawProcessor

def main():
    """Run a test of the law processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the ORBIT Law Processor")
    parser.add_argument("--input", "-i", help="Input file to process")
    parser.add_argument("--embedding", "-e", help="Path to FastText embedding model")
    parser.add_argument("--mapping", "-m", help="Path to token similarity mapping")
    parser.add_argument("--quality-model", "-q", help="Path to quality evaluation model")
    parser.add_argument("--evaluate-quality", "-eq", action="store_true", 
                       help="Evaluate document quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        # Initialize the law processor
        processor = LawProcessor(
            sim_mapping_path=args.mapping,
            embedding_path=args.embedding
        )
        
        print("ORBIT Law Processor Test")
        print("=======================")
        
        if args.embedding:
            print(f"Using embedding model from: {args.embedding}")
        if args.mapping:
            print(f"Using token similarity mapping from: {args.mapping}")
        if args.quality_model:
            print(f"Using quality evaluation model from: {args.quality_model}")
            processor.quality_evaluator.model_path = args.quality_model
            processor.quality_evaluator._load_model()
        print()
        
        # Sample law text for testing
        sample_text = """
        The Supreme Court has long recognized that the Fourth Amendment protects 
        citizens against unreasonable searches and seizures by the government. 
        In Katz v. United States, the Court established that the Fourth Amendment 
        protects people, not places, and that what a person knowingly exposes to 
        the public is not subject to Fourth Amendment protection. This case law 
        has been further developed in subsequent rulings, including United States v. Jones, 
        which addressed GPS tracking, and Carpenter v. United States, which concerned 
        cell phone location data. The doctrine of stare decisis requires courts to 
        follow precedent, ensuring consistency in legal interpretations across similar cases.
        
        The plaintiff argued that the defendant's actions constituted negligence 
        under tort law, as they breached their duty of care and directly caused 
        the plaintiff's injuries. The defense counsel countered that the plaintiff 
        had assumed the risk and that comparative negligence principles should apply. 
        The judge instructed the jury on the applicable legal standards, and after 
        deliberation, they returned a verdict in favor of the plaintiff, awarding 
        both compensatory and punitive damages.
        """
        
        print("Sample Law Text:")
        print("--------------")
        print(sample_text[:200] + "...\n")
        
        # Analyze the text
        print("Analysis Results:")
        print("----------------")
        results = processor.analyze_law_content(sample_text)
        
        print(f"Law relevance score: {results['law_relevance']:.4f}")
        print(f"Passes threshold: {results['passes_threshold']}")
        print(f"Concept count: {results['concept_count']}")
        print(f"Concept density: {results['concept_density']:.2f} per 1000 words")
        print(f"Word count: {results['word_count']}")
        
        print("\nDetected Law Concepts:")
        print("---------------------")
        for concept in results['law_concepts']:
            print(f"- {concept}")
        
        # Evaluate quality if requested
        if args.evaluate_quality:
            print("\nQuality Evaluation:")
            print("------------------")
            quality = processor.quality_evaluator.evaluate_quality(sample_text)
            print(f"Quality score: {quality.get('quality_score', 0.0):.4f}")
            print(f"Passes threshold: {quality.get('passes_threshold', False)}")
            if 'error' in quality:
                print(f"Error: {quality['error']}")
        
        # Test with a file if provided
        if args.input and os.path.exists(args.input):
            input_file = args.input
            print(f"\n\nProcessing file: {input_file}")
            
            # Create a temporary output directory
            output_dir = "test_law_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the file
            stats = processor.process_dataset(
                input_file, 
                output_dir=output_dir,
                evaluate_quality=args.evaluate_quality,
                quality_model_path=args.quality_model,
                num_workers=1
            )
            
            print("\nProcessing Statistics:")
            print("---------------------")
            pprint(stats)
            
            print(f"\nResults saved to: {output_dir}")
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 