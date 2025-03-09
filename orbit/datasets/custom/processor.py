"""
Custom Domain Dataset Processor

This module provides specialized tools for processing datasets in user-defined domains,
including filtering, quality assessment, and preparation for model training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import pyarrow as pq

from orbit.datasets.curator import DomainFilter, QualityEvaluator
from orbit.datasets.custom.utils.embedding_utils import CustomEmbedding
from orbit.datasets.custom.utils.filtering_utils import CustomFilter


class CustomDomainProcessor:
    """
    Specialized processor for custom domain datasets.
    
    This class extends the general dataset curation pipeline with
    domain-specific processing steps and optimizations for user-defined domains.
    """
    
    def __init__(self, 
                 domain_name: str,
                 keywords: List[str],
                 sim_mapping_path: Optional[str] = None,
                 embedding_path: Optional[str] = None,
                 embedding_type: str = "fasttext",
                 quality_threshold: float = 0.5,
                 quality_model_type: str = "simple"):
        """
        Initialize the CustomDomainProcessor.
        
        Args:
            domain_name: Name of the custom domain
            keywords: List of domain-specific keywords
            sim_mapping_path: Path to token similarity mapping
            embedding_path: Path to embedding model or vectors
            embedding_type: Type of embedding ("fasttext", "word2vec", etc.)
            quality_threshold: Threshold for quality score (0-1)
            quality_model_type: Type of quality model ('transformer' or 'simple')
        """
        self.domain_name = domain_name.lower().replace(" ", "_")
        self.keywords = keywords
        self.sim_mapping_path = sim_mapping_path
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type
        self.quality_threshold = quality_threshold
        self.quality_model_type = quality_model_type
        
        # Initialize components
        self.domain_filter = DomainFilter(
            domains=[self.domain_name],
            sim_mapping_path=sim_mapping_path,
            embedding_path=embedding_path,
            embedding_type=embedding_type
        )
        
        # Add custom domain keywords to domain filter
        if self.domain_name not in self.domain_filter.domain_keywords:
            self.domain_filter.domain_keywords[self.domain_name] = self.keywords
        
        # Initialize domain-specific components
        self.custom_filter = CustomFilter(domain_name=self.domain_name, keywords=self.keywords)
        self.custom_embedding = CustomEmbedding(domain_name=self.domain_name, embedding_path=embedding_path)
        
        # Initialize quality evaluator
        self.quality_evaluator = QualityEvaluator(
            quality_threshold=quality_threshold,
            model_type=quality_model_type
        )
    
    def process_dataset(self, input_path: str, output_dir: str = None,
                        evaluate_quality: bool = False, quality_model_path: Optional[str] = None,
                        quality_threshold: float = 0.5, quality_model_type: str = "simple",
                        num_workers: int = 1) -> Dict[str, Any]:
        """
        Process a dataset through the full curation pipeline for the custom domain.
        
        Args:
            input_path: Path to input dataset
            output_dir: Directory to save processed data
            evaluate_quality: Whether to evaluate document quality
            quality_model_path: Path to quality evaluation model
            quality_threshold: Threshold for quality score (0-1)
            quality_model_type: Type of quality model ('transformer' or 'simple')
            num_workers: Number of worker processes
            
        Returns:
            Processing statistics
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), f"processed_{self.domain_name}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: Domain filtering
        print(f"Stage 1: {self.domain_name.capitalize()} domain filtering")
        domain_output_path = os.path.join(output_dir, "stage1_domain_filtered.jsonl")
        
        # Initialize domain filter stats
        stats = {
            "input_documents": 0,
            "domain_filtered": 0,
            "quality_filtered": 0,
            "final_documents": 0
        }
        
        # Process input file
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(domain_output_path, 'w', encoding='utf-8') as outfile:
            
            for line in tqdm(infile, desc="Domain filtering"):
                stats["input_documents"] += 1
                
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    
                    if not text:
                        continue
                    
                    # Check domain relevance
                    domain_sim = self.domain_filter.compute_document_similarity(text)
                    domain_score = domain_sim.get(self.domain_name, 0.0)
                    
                    # Extract domain concepts
                    domain_concepts = self.custom_filter.extract_domain_concepts(text)
                    
                    # Add domain metadata to document
                    doc["domain"] = self.domain_name
                    doc["domain_score"] = domain_score
                    doc["domain_concepts"] = domain_concepts
                    
                    # Check if document passes domain filter
                    if self.custom_filter.is_domain_content(text):
                        outfile.write(json.dumps(doc) + "\n")
                        stats["domain_filtered"] += 1
                    
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing document: {e}")
        
        print(f"Domain filtering: {stats['domain_filtered']}/{stats['input_documents']} documents kept")
        
        # Stage 2: Quality Evaluation (if enabled)
        if evaluate_quality:
            print("Stage 2: Quality evaluation")
            quality_output_path = os.path.join(output_dir, "stage2_quality_filtered.jsonl")
            
            # Initialize quality evaluator with model
            if quality_model_path:
                quality_evaluator = QualityEvaluator(
                    model_path=quality_model_path,
                    quality_threshold=quality_threshold,
                    model_type=quality_model_type
                )
            else:
                quality_evaluator = self.quality_evaluator
            
            # Process domain-filtered documents
            with open(domain_output_path, 'r', encoding='utf-8') as infile, \
                 open(quality_output_path, 'w', encoding='utf-8') as outfile:
                
                for line in tqdm(infile, desc="Quality evaluation"):
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        
                        # Evaluate quality
                        quality_result = quality_evaluator.evaluate_quality(text)
                        
                        # Add quality score to document
                        doc["quality_score"] = quality_result.get("quality_score", 0.0)
                        doc["quality_passes"] = quality_result.get("passes_threshold", False)
                        
                        # Keep if passes quality threshold
                        if doc["quality_passes"]:
                            outfile.write(json.dumps(doc) + "\n")
                            stats["quality_filtered"] += 1
                            
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        print(f"Error evaluating quality: {e}")
            
            print(f"Quality evaluation: {stats['quality_filtered']}/{stats['domain_filtered']} documents kept")
            
            # Final output is quality filtered
            final_output_path = quality_output_path
            stats["final_documents"] = stats["quality_filtered"]
        else:
            # Final output is domain filtered
            final_output_path = domain_output_path
            stats["final_documents"] = stats["domain_filtered"]
        
        # Copy final output to a clean file
        final_clean_path = os.path.join(output_dir, f"final_{self.domain_name}_dataset.jsonl")
        with open(final_output_path, 'r', encoding='utf-8') as infile, \
             open(final_clean_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                outfile.write(line)
        
        print(f"Final dataset: {stats['final_documents']} documents")
        print(f"Saved to: {final_clean_path}")
        
        return stats
    
    def analyze_domain_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for domain-specific content.
        
        Args:
            text: Text to analyze
            
        Returns:
            Analysis results
        """
        # Extract domain concepts
        concepts = self.custom_filter.extract_domain_concepts(text)
        
        # Calculate metrics
        word_count = len(text.split())
        concept_density = len(concepts) / max(1, word_count) * 1000  # per 1000 words
        
        # Get domain similarity score
        domain_sim = self.domain_filter.compute_document_similarity(text)
        domain_score = domain_sim.get(self.domain_name, 0.0)
        
        # Check if passes threshold
        default_threshold = 0.5
        threshold = getattr(self.domain_filter, 'thresholds', {}).get(self.domain_name, default_threshold)
        
        return {
            f"{self.domain_name}_relevance": domain_score,
            f"{self.domain_name}_concepts": concepts,
            "concept_count": len(concepts),
            "concept_density": concept_density,
            "word_count": word_count,
            "passes_threshold": domain_score >= threshold
        }
    
    def prepare_training_data(self, input_path: str, output_dir: str = None,
                             format: str = "jsonl", split: bool = True,
                             train_ratio: float = 0.9) -> str:
        """
        Prepare domain dataset for model training.
        
        Args:
            input_path: Path to curated domain dataset
            output_dir: Directory to save prepared data
            format: Output format (jsonl, csv, etc.)
            split: Whether to split into train/validation sets
            train_ratio: Ratio of data to use for training
            
        Returns:
            Path to the prepared dataset
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), f"{self.domain_name}_training_data")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the dataset
        documents = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError:
                    continue
        
        print(f"Loaded {len(documents)} {self.domain_name} documents")
        
        # Process documents into training format
        training_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            # Create training example
            training_doc = {
                "text": text,
                "domain": self.domain_name
            }
            training_docs.append(training_doc)
        
        print(f"Created {len(training_docs)} training examples")
        
        # Split into train/validation sets if requested
        if split:
            import random
            random.shuffle(training_docs)
            
            split_idx = int(len(training_docs) * train_ratio)
            train_docs = training_docs[:split_idx]
            val_docs = training_docs[split_idx:]
            
            # Save train set
            train_path = os.path.join(output_dir, f"train.{format}")
            with open(train_path, 'w', encoding='utf-8') as f:
                for doc in train_docs:
                    f.write(json.dumps(doc) + "\n")
            
            # Save validation set
            val_path = os.path.join(output_dir, f"validation.{format}")
            with open(val_path, 'w', encoding='utf-8') as f:
                for doc in val_docs:
                    f.write(json.dumps(doc) + "\n")
            
            print(f"Saved {len(train_docs)} training examples to {train_path}")
            print(f"Saved {len(val_docs)} validation examples to {val_path}")
            
            return output_dir
        else:
            # Save all documents to a single file
            output_path = os.path.join(output_dir, f"{self.domain_name}_training_data.{format}")
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in training_docs:
                    f.write(json.dumps(doc) + "\n")
            
            print(f"Saved {len(training_docs)} training examples to {output_path}")
            return output_path 