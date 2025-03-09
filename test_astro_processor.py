#!/usr/bin/env python
"""
Test script for the ORBIT Astronomy Dataset Processor.

This script demonstrates the basic functionality of the AstroProcessor class
by analyzing a sample astronomy text and showing the results.
"""

import os
import sys
import json
import argparse
from pprint import pprint

# Add the project root to the path so we can import the orbit package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit.datasets.astro import AstroProcessor

def main():
    """Run a test of the astronomy processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the ORBIT Astronomy Processor")
    parser.add_argument("--input", "-i", help="Input file to process")
    parser.add_argument("--embedding", "-e", help="Path to FastText embedding model")
    parser.add_argument("--mapping", "-m", help="Path to token similarity mapping")
    parser.add_argument("--quality-model", "-q", help="Path to quality evaluation model")
    parser.add_argument("--evaluate-quality", "-eq", action="store_true", 
                       help="Evaluate document quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    print("ORBIT Astronomy Processor Test")
    print("==============================\n")
    
    try:
        # Initialize the processor with embeddings if provided
        processor = AstroProcessor(
            sim_mapping_path=args.mapping,
            embedding_path=args.embedding,
            embedding_type="fasttext"
        )
        
        print(f"Initialized AstroProcessor with {len(processor.astro_filter.keywords)} astronomy keywords")
        if args.embedding:
            print(f"Using FastText embeddings from: {args.embedding}")
        if args.mapping:
            print(f"Using token similarity mapping from: {args.mapping}")
        if args.quality_model:
            print(f"Using quality evaluation model from: {args.quality_model}")
            processor.quality_evaluator.model_path = args.quality_model
            processor.quality_evaluator._load_model()
        print()
        
        # Sample astronomy text
        sample_text = """
        The Hubble Space Telescope has revolutionized our understanding of the cosmos.
        It has observed distant galaxies, nebulae, and supernovae, providing insights into
        the early universe. Black holes, once theoretical objects, have been indirectly
        observed through their effects on surrounding matter. The James Webb Space Telescope
        (JWST) will soon extend our observational capabilities even further into the infrared
        spectrum, allowing astronomers to peer through cosmic dust to observe star formation
        and potentially detect biosignatures in exoplanet atmospheres.
        
        Recent observations of gravitational waves from merging neutron stars have opened
        a new era of multi-messenger astronomy, combining electromagnetic and gravitational
        wave observations to study cosmic phenomena.
        """
        
        print("Sample Text:")
        print("-----------")
        print(sample_text)
        print("\n")
        
        # Analyze the text
        print("Astronomy Content Analysis:")
        print("--------------------------")
        analysis = processor.analyze_astronomy_content(sample_text)
        
        # Print the analysis results in a readable format
        print(f"Astronomy relevance score: {analysis['astronomy_relevance']:.4f}")
        print(f"Passes threshold: {analysis['passes_threshold']}")
        print(f"Word count: {analysis['word_count']}")
        print(f"Concept count: {analysis['concept_count']}")
        print(f"Concept density: {analysis['concept_density']:.2f} per 1000 words")
        
        print("\nDetected astronomy concepts:")
        for concept in analysis['astronomy_concepts']:
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
            output_dir = "test_astro_output"
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