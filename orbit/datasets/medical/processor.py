"""
Medical Dataset Processor

This module provides specialized tools for processing medical datasets,
including filtering, quality assessment, and preparation for model training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import pyarrow as pq

from orbit.datasets.curator import DomainFilter, QualityEvaluator
from orbit.datasets.medical.utils.embedding_utils import MedicalEmbedding
from orbit.datasets.medical.utils.filtering_utils import MedicalFilter


class MedicalProcessor:
    """
    Specialized processor for medical datasets.
    
    This class extends the general dataset curation pipeline with
    medical-specific processing steps and optimizations.
    """
    
    def __init__(self, 
                 sim_mapping_path: Optional[str] = None,
                 embedding_path: Optional[str] = None,
                 embedding_type: str = "fasttext",
                 quality_threshold: float = 0.5,
                 quality_model_type: str = "simple",
                 custom_keywords_path: Optional[str] = None):
        """
        Initialize the MedicalProcessor.
        
        Args:
            sim_mapping_path: Path to token similarity mapping
            embedding_path: Path to embedding model or vectors
            embedding_type: Type of embedding ("fasttext", "word2vec", etc.)
            quality_threshold: Threshold for quality score (0-1)
            quality_model_type: Type of quality model ('transformer' or 'simple')
            custom_keywords_path: Path to custom medical keywords file
        """
        self.domain = "medical"
        self.sim_mapping_path = sim_mapping_path
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type
        self.quality_threshold = quality_threshold
        self.quality_model_type = quality_model_type
        
        # Load custom keywords if provided
        self.custom_keywords = []
        if custom_keywords_path and os.path.exists(custom_keywords_path):
            with open(custom_keywords_path, 'r', encoding='utf-8') as f:
                self.custom_keywords = [line.strip() for line in f if line.strip()]
        
        # Initialize components
        self.domain_filter = DomainFilter(
            domains=[self.domain],
            sim_mapping_path=sim_mapping_path,
            embedding_path=embedding_path,
            embedding_type=embedding_type
        )
        
        # Initialize medical-specific components
        self.medical_filter = MedicalFilter(custom_keywords=self.custom_keywords)
        self.medical_embedding = MedicalEmbedding(embedding_path=embedding_path)
        
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
        Process a medical dataset through the full curation pipeline.
        
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
            output_dir = os.path.join(os.path.dirname(input_path), "processed_medical")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Stage 1: Domain filtering
        print("Stage 1: Domain filtering")
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
                    medical_score = domain_sim.get("medical", 0.0)
                    
                    # Get domain threshold
                    threshold = getattr(self.domain_filter, 'thresholds', {}).get("medical", 0.238)
                    
                    if medical_score >= threshold:
                        # Add domain score to document
                        doc["domain_score"] = medical_score
                        
                        # Extract medical concepts
                        concepts = self.medical_filter.extract_medical_concepts(text)
                        doc["medical_concepts"] = concepts
                        doc["concept_count"] = len(concepts)
                        
                        # Write to output
                        outfile.write(json.dumps(doc) + "\n")
                        stats["domain_filtered"] += 1
                        
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"Error processing document: {e}")
        
        print(f"Domain filtering: {stats['domain_filtered']}/{stats['input_documents']} documents kept")
        
        # Stage 2: Quality evaluation (if requested)
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
            
            with open(domain_output_path, 'r', encoding='utf-8') as infile, \
                 open(quality_output_path, 'w', encoding='utf-8') as outfile:
                
                for line in tqdm(infile, desc="Quality evaluation"):
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        
                        if not text:
                            continue
                        
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
        final_clean_path = os.path.join(output_dir, "final_medical_dataset.jsonl")
        with open(final_output_path, 'r', encoding='utf-8') as infile, \
             open(final_clean_path, 'w', encoding='utf-8') as outfile:
            
            for line in infile:
                outfile.write(line)
        
        print(f"Final dataset: {stats['final_documents']} documents")
        print(f"Saved to: {final_clean_path}")
        
        return stats
    
    def analyze_medical_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for medical content.
        
        Args:
            text: The text to analyze
            
        Returns:
            Analysis results including relevance scores and key concepts
        """
        # Basic domain relevance
        domain_sim = self.domain_filter.compute_document_similarity(text)
        medical_score = domain_sim.get("medical", 0.0)
        
        # Extract medical concepts
        concepts = self.medical_filter.extract_medical_concepts(text)
        
        # Calculate concept density
        word_count = len(text.split())
        concept_density = len(concepts) / max(1, word_count) * 1000  # per 1000 words
        
        # Determine if it passes threshold - use a default if not available
        default_threshold = 0.238
        threshold = getattr(self.domain_filter, 'thresholds', {}).get("medical", default_threshold)
        
        return {
            "medical_relevance": medical_score,
            "medical_concepts": concepts,
            "concept_count": len(concepts),
            "concept_density": concept_density,
            "word_count": word_count,
            "passes_threshold": medical_score >= threshold
        }
    
    def prepare_training_data(self, input_path: str, output_dir: str = None,
                             format: str = "jsonl", split: bool = True,
                             train_ratio: float = 0.9) -> str:
        """
        Prepare medical dataset for model training.
        
        Args:
            input_path: Path to curated medical dataset
            output_dir: Directory to save prepared data
            format: Output format (jsonl, csv, etc.)
            split: Whether to split into train/validation sets
            train_ratio: Ratio of data to use for training
            
        Returns:
            Path to the prepared dataset
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), "medical_training_data")
        
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
        
        print(f"Loaded {len(documents)} medical documents")
        
        # Process documents into training format
        training_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            # Create training example
            training_doc = {
                "text": text,
                "domain": "medical"
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
            output_path = os.path.join(output_dir, f"medical_training_data.{format}")
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in training_docs:
                    f.write(json.dumps(doc) + "\n")
            
            print(f"Saved {len(training_docs)} training examples to {output_path}")
            return output_path 