#!/usr/bin/env python
"""
Label data for training the ORBIT Stage 2 quality evaluation model.

This script helps create labeled data for training the quality evaluation model
by allowing users to manually label documents or by using heuristics.
"""

import os
import json
import argparse
import random
from typing import List, Dict, Any
from tqdm import tqdm

def load_documents(input_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSONL file.
    
    Args:
        input_path: Path to input file
        
    Returns:
        List of documents
    """
    documents = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {os.path.basename(input_path)}"):
            try:
                doc = json.loads(line)
                if "text" in doc:
                    documents.append(doc)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(documents)} documents")
    return documents

def label_documents_manual(documents: List[Dict[str, Any]], 
                          sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Manually label documents for quality.
    
    Args:
        documents: List of documents to label
        sample_size: Number of documents to sample
        
    Returns:
        List of labeled documents
    """
    # Sample documents
    if sample_size < len(documents):
        sampled_docs = random.sample(documents, sample_size)
    else:
        sampled_docs = documents
    
    labeled_docs = []
    
    print(f"\nManually labeling {len(sampled_docs)} documents for quality")
    print("For each document, enter 1 for high quality or 0 for low quality")
    print("Enter 'q' to quit labeling\n")
    
    for i, doc in enumerate(sampled_docs):
        print(f"\nDocument {i+1}/{len(sampled_docs)}")
        print("-" * 80)
        
        # Print a preview of the document
        text = doc["text"]
        preview = text[:500] + "..." if len(text) > 500 else text
        print(preview)
        print("-" * 80)
        
        # Get label
        while True:
            label = input("Quality (1=high, 0=low, q=quit): ").strip().lower()
            if label == 'q':
                return labeled_docs
            elif label in ['0', '1']:
                break
            else:
                print("Invalid input. Please enter 0, 1, or q.")
        
        # Add labeled document
        labeled_doc = doc.copy()
        labeled_doc["quality"] = int(label)
        labeled_docs.append(labeled_doc)
    
    return labeled_docs

def label_documents_heuristic(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Label documents using heuristics.
    
    Args:
        documents: List of documents to label
        
    Returns:
        List of labeled documents
    """
    labeled_docs = []
    
    for doc in tqdm(documents, desc="Labeling with heuristics"):
        text = doc["text"]
        
        # Simple heuristics for quality
        word_count = len(text.split())
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # More heuristics
        has_urls = "http" in text.lower()
        has_code = "```" in text or "def " in text or "class " in text
        has_bullets = "* " in text or "- " in text
        has_numbers = any(c.isdigit() for c in text)
        
        # Calculate quality score based on heuristics
        quality_score = 0
        
        # Length-based quality
        if 100 <= word_count <= 2000:
            quality_score += 1
        
        # Sentence length (avoid very short or very long sentences)
        if 10 <= avg_sentence_length <= 30:
            quality_score += 1
        
        # Content features
        if has_urls:
            quality_score += 0.5
        if has_code:
            quality_score += 0.5
        if has_bullets:
            quality_score += 0.5
        if has_numbers:
            quality_score += 0.5
        
        # Normalize to 0-1 range
        normalized_score = quality_score / 4.0
        
        # Binary classification
        quality_label = 1 if normalized_score >= 0.5 else 0
        
        # Add labeled document
        labeled_doc = doc.copy()
        labeled_doc["quality"] = quality_label
        labeled_doc["quality_score"] = normalized_score
        labeled_docs.append(labeled_doc)
    
    # Print statistics
    high_quality = sum(1 for doc in labeled_docs if doc["quality"] == 1)
    low_quality = sum(1 for doc in labeled_docs if doc["quality"] == 0)
    print(f"Labeled {len(labeled_docs)} documents:")
    print(f"  - High quality: {high_quality} ({high_quality/len(labeled_docs)*100:.1f}%)")
    print(f"  - Low quality: {low_quality} ({low_quality/len(labeled_docs)*100:.1f}%)")
    
    return labeled_docs

def main():
    """Label data for quality evaluation."""
    parser = argparse.ArgumentParser(description="Label data for quality evaluation")
    parser.add_argument("--input", "-i", required=True, help="Path to input data")
    parser.add_argument("--output", "-o", required=True, help="Path to output labeled data")
    parser.add_argument("--method", "-m", choices=["manual", "heuristic"], default="heuristic",
                       help="Labeling method (manual or heuristic)")
    parser.add_argument("--sample", "-s", type=int, default=100,
                       help="Number of documents to sample for manual labeling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load documents
    documents = load_documents(args.input)
    
    # Label documents
    if args.method == "manual":
        labeled_docs = label_documents_manual(documents, args.sample)
    else:
        labeled_docs = label_documents_heuristic(documents)
    
    # Save labeled documents
    with open(args.output, 'w', encoding='utf-8') as f:
        for doc in labeled_docs:
            f.write(json.dumps(doc) + "\n")
    
    print(f"Saved {len(labeled_docs)} labeled documents to {args.output}")

if __name__ == "__main__":
    main() 