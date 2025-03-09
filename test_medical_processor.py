#!/usr/bin/env python
"""
Test script for the ORBIT Medical Dataset Processor.

This script demonstrates the basic functionality of the MedicalProcessor class
by analyzing a sample medical text and showing the results.
"""

import os
import sys
import json
import argparse
from pprint import pprint

# Add the project root to the path so we can import the orbit package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit.datasets.medical import MedicalProcessor

def main():
    """Run a test of the medical processor."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the ORBIT Medical Processor")
    parser.add_argument("--input", "-i", help="Input file to process")
    parser.add_argument("--embedding", "-e", help="Path to FastText embedding model")
    parser.add_argument("--mapping", "-m", help="Path to token similarity mapping")
    parser.add_argument("--quality-model", "-q", help="Path to quality evaluation model")
    parser.add_argument("--evaluate-quality", "-eq", action="store_true", 
                       help="Evaluate document quality")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    args = parser.parse_args()
    
    try:
        # Initialize the medical processor
        processor = MedicalProcessor(
            sim_mapping_path=args.mapping,
            embedding_path=args.embedding
        )
        
        print("ORBIT Medical Processor Test")
        print("===========================")
        
        if args.embedding:
            print(f"Using embedding model from: {args.embedding}")
        if args.mapping:
            print(f"Using token similarity mapping from: {args.mapping}")
        if args.quality_model:
            print(f"Using quality evaluation model from: {args.quality_model}")
            processor.quality_evaluator.model_path = args.quality_model
            processor.quality_evaluator._load_model()
        print()
        
        # Sample medical text for testing
        sample_text = """
        The patient presented with acute respiratory distress syndrome (ARDS) 
        following a severe case of COVID-19 pneumonia. Initial assessment revealed 
        hypoxemia with an oxygen saturation of 85% on room air, tachypnea, and 
        bilateral crackles on auscultation. Chest radiography showed diffuse bilateral 
        infiltrates consistent with ARDS. Laboratory findings included elevated 
        inflammatory markers (CRP, ferritin, IL-6) and lymphopenia.
        
        Treatment was initiated with high-flow nasal cannula oxygen therapy, 
        dexamethasone 6mg daily, and prophylactic anticoagulation with enoxaparin. 
        The patient was placed in prone position for 16 hours per day. Despite these 
        interventions, the patient's respiratory status deteriorated, necessitating 
        intubation and mechanical ventilation with lung-protective strategies 
        (low tidal volume, optimal PEEP). Remdesivir was administered for 5 days, 
        and broad-spectrum antibiotics were initiated to cover possible bacterial 
        superinfection pending culture results.
        
        After 14 days in the ICU, the patient showed gradual improvement in oxygenation 
        and was successfully extubated. Rehabilitation therapy was initiated to address 
        ICU-acquired weakness. The patient was discharged after 21 days with home oxygen 
        therapy and follow-up appointments scheduled with pulmonology to monitor for 
        post-COVID pulmonary fibrosis.
        """
        
        print("Sample Medical Text:")
        print("------------------")
        print(sample_text[:200] + "...\n")
        
        # Analyze the text
        print("Analysis Results:")
        print("----------------")
        results = processor.analyze_medical_content(sample_text)
        
        print(f"Medical relevance score: {results['medical_relevance']:.4f}")
        print(f"Passes threshold: {results['passes_threshold']}")
        print(f"Concept count: {results['concept_count']}")
        print(f"Concept density: {results['concept_density']:.2f} per 1000 words")
        print(f"Word count: {results['word_count']}")
        
        print("\nDetected Medical Concepts:")
        print("------------------------")
        for concept in results['medical_concepts']:
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
            output_dir = "test_medical_output"
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