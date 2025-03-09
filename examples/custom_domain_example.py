#!/usr/bin/env python
"""
Example of using the ORBIT Custom Domain Processor.

This script demonstrates how to define a custom domain (in this case, "finance")
and process a dataset through the ORBIT curation pipeline.
"""

import os
import sys
import json
from pprint import pprint

# Add the project root to the path so we can import the orbit package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from orbit.datasets.custom import CustomDomainProcessor

def main():
    """Run the custom domain example."""
    # Define a custom domain (finance)
    domain_name = "finance"
    keywords = [
        "stock", "bond", "investment", "portfolio", "dividend", "equity",
        "asset", "liability", "balance sheet", "income statement", "cash flow",
        "market", "trading", "broker", "exchange", "securities", "derivative",
        "option", "futures", "hedge", "risk", "return", "yield", "interest rate",
        "inflation", "recession", "bull market", "bear market", "volatility",
        "diversification", "mutual fund", "ETF", "index fund", "retirement",
        "401k", "IRA", "pension", "annuity", "insurance", "tax", "capital gain",
        "dividend", "earnings", "revenue", "profit", "loss", "debt", "credit",
        "loan", "mortgage", "banking", "fintech", "cryptocurrency", "blockchain"
    ]
    
    # Sample finance text for testing
    sample_text = """
    The stock market experienced significant volatility today as investors reacted to the 
    Federal Reserve's announcement on interest rates. The S&P 500 index fell 2.3%, while 
    the Nasdaq composite dropped 3.1%. Technology stocks were particularly hard hit, with 
    major tech companies seeing their market capitalization decrease by billions.
    
    Analysts attribute the sell-off to concerns about inflation and the potential for 
    higher borrowing costs affecting corporate profits. Bond yields rose sharply, with 
    the 10-year Treasury yield reaching its highest level in three months. This inverse 
    relationship between bond prices and yields reflects the market's expectation of 
    tighter monetary policy in the coming quarters.
    
    Meanwhile, value stocks showed more resilience, with financial and energy sectors 
    outperforming the broader market. Banks stand to benefit from higher interest rates, 
    which can improve their net interest margins. Investment firms are advising clients 
    to rebalance portfolios, potentially increasing allocation to dividend-paying stocks 
    and inflation-protected securities.
    """
    
    # Initialize the custom domain processor
    processor = CustomDomainProcessor(
        domain_name=domain_name,
        keywords=keywords
    )
    
    print("ORBIT Custom Domain Example: Finance")
    print("===================================")
    
    # Analyze the sample text
    print("\nSample Finance Text:")
    print("------------------")
    print(sample_text[:200] + "...\n")
    
    # Analyze the text
    print("Analysis Results:")
    print("----------------")
    results = processor.analyze_domain_content(sample_text)
    
    print(f"Finance relevance score: {results['domain_relevance']:.4f}")
    print(f"Passes threshold: {results['passes_threshold']}")
    print(f"Concept count: {results['concept_count']}")
    print(f"Concept density: {results['concept_density']:.2f} per 1000 words")
    print(f"Word count: {results['word_count']}")
    
    print("\nDetected Finance Concepts:")
    print("------------------------")
    for concept in results['domain_concepts']:
        print(f"- {concept}")
    
    # Create a sample dataset
    print("\nCreating sample dataset...")
    sample_dir = "finance_sample"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_data_path = os.path.join(sample_dir, "finance_sample.jsonl")
    with open(sample_data_path, 'w', encoding='utf-8') as f:
        # Write the sample text as a document
        f.write(json.dumps({"text": sample_text}) + "\n")
        
        # Add a non-finance document
        non_finance_text = """
        The new movie premiered last night to critical acclaim. The director's vision 
        was praised for its innovative storytelling and stunning visuals. Audiences 
        responded positively, with many citing the lead actor's performance as 
        Oscar-worthy. The film explores themes of identity and belonging in a 
        post-modern society.
        """
        f.write(json.dumps({"text": non_finance_text}) + "\n")
    
    # Process the sample dataset
    print(f"\nProcessing sample dataset: {sample_data_path}")
    stats = processor.process_dataset(
        input_path=sample_data_path,
        output_dir=os.path.join(sample_dir, "processed")
    )
    
    print("\nProcessing Statistics:")
    print("---------------------")
    pprint(stats)
    
    # Prepare training data
    print("\nPreparing training data...")
    final_dataset = os.path.join(sample_dir, "processed", f"final_{domain_name}_dataset.jsonl")
    training_path = processor.prepare_training_data(
        input_path=final_dataset,
        output_dir=os.path.join(sample_dir, "training"),
        split=True
    )
    
    print(f"\nTraining data prepared at: {training_path}")
    print("\nExample complete!")

if __name__ == "__main__":
    main() 