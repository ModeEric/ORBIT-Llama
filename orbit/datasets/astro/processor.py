"""
Astronomy Dataset Processor

This module provides specialized tools for processing astronomy datasets,
including filtering, quality assessment, and preparation for model training.
"""

import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Union
from tqdm import tqdm
import pyarrow as pq

from orbit.datasets.curator import DomainFilter, QualityEvaluator
from orbit.datasets.astro.utils.embedding_utils import AstroEmbedding
from orbit.datasets.astro.utils.filtering_utils import AstroFilter


class AstroProcessor:
    """
    Specialized processor for astronomy datasets.
    
    This class extends the general dataset curation pipeline with
    astronomy-specific processing steps and optimizations.
    """
    
    def __init__(self, 
                 sim_mapping_path: Optional[str] = None,
                 embedding_path: Optional[str] = None,
                 embedding_type: str = "fasttext",
                 quality_threshold: float = 0.5,
                 quality_model_type: str = "simple",
                 custom_keywords_path: Optional[str] = None):
        """
        Initialize the AstroProcessor.
        
        Args:
            sim_mapping_path: Path to token similarity mapping
            embedding_path: Path to embedding model or vectors
            embedding_type: Type of embedding ("fasttext", "word2vec", etc.)
            quality_threshold: Threshold for quality score (0-1)
            quality_model_type: Type of quality model ('transformer' or 'simple')
            custom_keywords_path: Path to custom astronomy keywords file
        """
        self.domain = "astronomy"
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
        
        # Initialize astronomy-specific components
        self.astro_filter = AstroFilter(custom_keywords=self.custom_keywords)
        self.astro_embedding = AstroEmbedding(embedding_path=embedding_path)
        
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
        Process an astronomy dataset through the full curation pipeline.
        
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
            output_dir = os.path.join(os.path.dirname(input_path), "processed_astro")
        
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
        
        # Process the file based on format
        if input_path.endswith(".parquet"):
            # Process parquet file
            table = pq.read_table(input_path)
            df = table.to_pandas()
            stats["input_documents"] = len(df)
            
            with open(domain_output_path, 'w', encoding='utf-8') as outfile:
                for _, row in tqdm(df.iterrows(), total=len(df), desc="Domain filtering"):
                    text = row.get("text", "")
                    if not text:
                        continue
                    
                    # Check domain relevance
                    domain_result = self.domain_filter.filter_document(text)
                    if domain_result.get(self.domain, False):
                        # Document passes domain filter
                        doc = {"text": text}
                        outfile.write(json.dumps(doc) + "\n")
                        stats["domain_filtered"] += 1
        else:
            # Process text file (assuming JSONL)
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
                        domain_result = self.domain_filter.filter_document(text)
                        if domain_result.get(self.domain, False):
                            # Document passes domain filter
                            outfile.write(json.dumps({"text": text}) + "\n")
                            stats["domain_filtered"] += 1
                            
                    except json.JSONDecodeError:
                        continue
        
        print(f"Domain filter: kept {stats['domain_filtered']}/{stats['input_documents']} documents")
        
        # Stage 2: Quality evaluation (if requested)
        if evaluate_quality:
            print("Stage 2: Quality evaluation")
            
            # Initialize quality evaluator with model
            if quality_model_path:
                quality_evaluator = QualityEvaluator(
                    model_path=quality_model_path,
                    quality_threshold=quality_threshold,
                    model_type=quality_model_type
                )
            else:
                quality_evaluator = self.quality_evaluator
                
            quality_output_path = os.path.join(output_dir, "stage2_quality_filtered.jsonl")
            
            with open(domain_output_path, 'r', encoding='utf-8') as infile, \
                 open(quality_output_path, 'w', encoding='utf-8') as outfile:
                
                for line in tqdm(infile, desc="Quality evaluation"):
                    try:
                        doc = json.loads(line)
                        text = doc.get("text", "")
                        
                        # Evaluate quality
                        quality_result = quality_evaluator.evaluate_quality(text)
                        
                        if quality_result.get("passes_threshold", False):
                            # Document passes quality filter
                            doc["quality_score"] = quality_result.get("quality_score", 0.0)
                            outfile.write(json.dumps(doc) + "\n")
                            stats["quality_filtered"] += 1
                            
                    except json.JSONDecodeError:
                        continue
            
            print(f"Quality filter: kept {stats['quality_filtered']}/{stats['domain_filtered']} documents")
            stats["final_documents"] = stats["quality_filtered"]
            final_output_path = quality_output_path
        else:
            # Skip quality evaluation
            stats["final_documents"] = stats["domain_filtered"]
            final_output_path = domain_output_path
        
        # Copy final output to main output file
        output_path = os.path.join(output_dir, "curated_astronomy_dataset.jsonl")
        with open(final_output_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                outfile.write(line)
        
        print(f"Processed dataset saved to {output_path}")
        print(f"Final dataset contains {stats['final_documents']} documents")
        
        return stats
    
    def analyze_astronomy_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze astronomy content for key concepts and quality.
        
        Args:
            text: The text to analyze
            
        Returns:
            Analysis results including relevance scores and key concepts
        """
        # Basic domain relevance
        domain_sim = self.domain_filter.compute_document_similarity(text)
        astro_score = domain_sim.get("astronomy", 0.0)
        
        # Extract astronomy concepts
        concepts = self.astro_filter.extract_astronomy_concepts(text)
        
        # Calculate concept density
        word_count = len(text.split())
        concept_density = len(concepts) / max(1, word_count) * 1000  # per 1000 words
        
        # Determine if it passes threshold - use a default if not available
        default_threshold = 0.242
        threshold = getattr(self.domain_filter, 'thresholds', {}).get("astronomy", default_threshold)
        
        return {
            "astronomy_relevance": astro_score,
            "astronomy_concepts": concepts,
            "concept_count": len(concepts),
            "concept_density": concept_density,
            "word_count": word_count,
            "passes_threshold": astro_score >= threshold
        }
    
    def prepare_training_data(self, input_path: str, output_dir: str = None,
                             format: str = "jsonl", split: bool = True,
                             train_ratio: float = 0.9) -> str:
        """
        Prepare astronomy dataset for model training.
        
        Args:
            input_path: Path to curated astronomy dataset
            output_dir: Directory to save prepared data
            format: Output format (jsonl, csv, etc.)
            split: Whether to split into train/validation sets
            train_ratio: Ratio of data to use for training
            
        Returns:
            Path to the prepared dataset
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), "astro_training_data")
        
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
        
        print(f"Loaded {len(documents)} astronomy documents")
        
        # Process documents into training format
        training_docs = []
        for doc in documents:
            text = doc.get("text", "")
            if not text:
                continue
                
            # Create training example
            training_doc = {
                "text": text,
                "domain": "astronomy"
            }
            training_docs.append(training_doc)
        
        # Split into train/validation if requested
        if split:
            from sklearn.model_selection import train_test_split
            train_docs, val_docs = train_test_split(
                training_docs, train_size=train_ratio, random_state=42
            )
            
            # Save train set
            train_path = os.path.join(output_dir, f"astronomy_train.{format}")
            with open(train_path, 'w', encoding='utf-8') as f:
                for doc in train_docs:
                    f.write(json.dumps(doc) + '\n')
            
            # Save validation set
            val_path = os.path.join(output_dir, f"astronomy_val.{format}")
            with open(val_path, 'w', encoding='utf-8') as f:
                for doc in val_docs:
                    f.write(json.dumps(doc) + '\n')
                    
            print(f"Saved {len(train_docs)} training and {len(val_docs)} validation examples")
            return output_dir
        else:
            # Save all as one dataset
            output_path = os.path.join(output_dir, f"astronomy_dataset.{format}")
            with open(output_path, 'w', encoding='utf-8') as f:
                for doc in training_docs:
                    f.write(json.dumps(doc) + '\n')
            
            print(f"Saved {len(training_docs)} examples to {output_path}")
            return output_path 