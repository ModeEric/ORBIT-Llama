"""
Dataset Curation Tools for ORBIT

This module provides classes for curating domain-specific datasets, including:
- Filtering content by domain relevance
- Evaluating content quality
- Processing and preparing datasets for model training
"""

import os
import json
import re
import gc
import pickle
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union, Any
from abc import ABC, abstractmethod

class DomainFilter:
    """
    Filters content based on domain relevance using embedding similarity.
    
    This class implements the first stage of the ORBIT curation pipeline,
    identifying content that is relevant to specific domains.
    """
    
    # Domain-specific keyword lists
    DOMAIN_KEYWORDS = {
        "astronomy": [
            "Albedo", "Aphelion", "Apogee", "Asteroid", "Astronomy", "Aurora",
            "Axion", "Azimuth", "Barycenter", "Baryon", "Blackbody", "Bolide",
            "Brilliance", "Cepheid", "Comet", "Constellation", "Corona", "Cosmic",
            "Cosmology", "DESC", "Dyne", "Eclipse", "Ecliptic", "Emission",
            "Erg", "Exoplanet", "Extinction", "Fluence", "Frequency", "Galaxy",
            "Geocentric", "Gibbous", "Gravity", "Heliocentric", "Interferometry", 
            "Isotropic", "JWST", "kpc", "Light-Year", "LSST", "Luminosity", 
            "Magnetar", "Magnetosphere", "Metallicity", "Meteor", "Meteorite", 
            "Microlensing", "Moon", "Morphology", "Multiverse", "Nebula",
            "Neutrino", "Noctilucent", "Nova", "Nucleosynthesis", "Orbit", 
            "Parallax", "Parsec", "Perihelion", "Phase", "Photometry", 
            "Photosphere", "Planck", "Planetesimal", "Pulsar", "Quasar",
            "Quiescence", "Recombination", "Reddening", "Redshift", "Reionization", 
            "Satellite", "Seyfert", "Simulation", "Singularity", "Spectroscopy", 
            "SPT", "Sublimation", "Sunspot", "Supercomputer", "Supermassive",
            "Supernova", "Telescope", "Transit", "Universe", "Voids", 
            "Wavelength", "Waxing", "Wormhole", "X-ray", "Zenith", "Zodiac", 
            "Optical", "Infrared", "Ultraviolet", "Microwave", "Proton", 
            "Neutron", "Electron", "Flux", "Intensity", "Companion",
            "Outflow", "QSO", "Pulse", "Progenitor"
        ],
        "law": [
            "Constitution", "Legislation", "Jurisprudence", "Statute", "Regulation", 
            "Ordinance", "Litigation", "Arbitration", "Mediation", "Contract", 
            "Agreement", "Obligation", "Tort", "Negligence", "Liability",
            "Defendant", "Plaintiff", "Witness", "Indictment", "Prosecution", 
            "Defense", "Verdict", "Sentence", "Appeal", "Trial", "Hearing", 
            "Subpoena", "Judgment", "Ruling", "Decision", "Courtroom", "Tribunal", 
            "Bench", "Attorney", "Counsel", "Lawyer", "Barrister", "Solicitor", 
            "Evidence", "Testimony", "Exhibit", "Discovery", "Deposition", 
            "Interrogatory", "Precedent", "StareDecisis", "Injunction", "Amicus",
            "Appellate", "Bail", "Parole", "Probation", "Incarceration", 
            "Detention", "Custody", "Warrant", "Arrest", "Search", "DueProcess", 
            "EqualProtection", "CivilRights", "HumanRights", "Liberty",
            "Impeachment", "Sovereignty", "Taxation", "Revenue", "Compliance", 
            "Oversight", "Malpractice", "Fraud", "Antitrust", "Monopoly", 
            "Patent", "Copyright", "Trademark", "Licensing", "Franchise",
            "Mergers", "Acquisitions", "Employment", "Labor", "Union", 
            "Privacy", "Defamation", "Cyberlaw", "Immigration", "Asylum", 
            "Citizenship", "Environmental", "Treaty", "Diplomacy", "Criminal",
            "Felony", "Misdemeanor", "Substantive", "Procedural"
        ],
        "medicine": [
            "Anatomy", "Pathology", "Physiology", "Oncology", "Cardiology", 
            "Neurology", "Radiology", "Pharmacology", "Surgery", "Pediatrics", 
            "Dermatology", "Gastroenterology", "Endocrinology", "Hematology", 
            "Immunology", "Nephrology", "Pulmonology", "Psychiatry",
            "Rheumatology", "Urology", "Obstetrics", "Gynecology", "Orthopedics", 
            "Ophthalmology", "Otolaryngology", "Infectious", "Microbiology",
            "Epidemiology", "Toxicology", "Genetics", "Biochemistry", "Histology", 
            "Embryology", "Virology", "Bacteriology", "Parasitology",
            "Cytology", "Prognosis", "Diagnosis", "Treatment", "Therapy", 
            "Vaccination", "Antibiotic", "Antiviral", "Pathogen", "Tumor", 
            "Cancer", "Leukemia", "Diabetes", "Hypertension", "Cardiomyopathy",
            "Stroke", "Sepsis", "Inflammation", "Autoimmune", "Fibrosis", 
            "Circulation", "Respiration", "Homeostasis", "Anesthesia", "Trauma", 
            "Fracture", "Hemorrhage", "Venous", "Arterial", "Renal",
            "Hepatic", "Liver", "Kidney", "Lung", "Heart", "Brain", "Spinal", 
            "Nerve", "Bone", "Muscle", "Skin", "Blood", "Plasma", "Lymph", 
            "Hormone", "Enzyme", "Protein", "Gene", "DNA", "RNA", "Chromosome",
            "Cell", "Tissue", "Organ", "Organism", "Metabolism", "Nutrition",
            "Obesity", "Malnutrition", "Infection", "Immunity", "Allergy", 
            "Vaccine", "Mutation", "Carcinogen", "Biopsy", "MRI", "CT", "X-ray",
            "Ultrasound", "PET", "Radiotherapy", "Chemotherapy", "Surgical", 
            "Endoscopy", "Laparoscopy", "Thermography", "Pharmacokinetics",
            "Pharmacodynamics", "Clinical", "Hospital", "Ambulance", "ICU", 
            "Ward", "Therapist", "Psychologist", "Psychiatrist", "Physician", 
            "Surgeon", "Nurse", "Paramedic", "Dentist", "Optometrist",
            "Audiologist", "Dietitian", "Nutritionist", "Emergency", "CPR", 
            "Defibrillator", "Vaccination", "Inoculation", "Antibody", "Antigen", 
            "Biomarker", "Cytokine", "Pathogenesis", "Therapeutics", "Rehabilitation",
            "Prosthesis", "Implant", "Transplant", "Donor", "Recipient", 
            "ClinicalTrial", "Placebo", "DoubleBlind", "Epidemic", "Pandemic", 
            "Outbreak", "Quarantine", "Contagion", "Immunotherapy", "PrecisionMedicine"
        ],
        "material_science": [
            "Alloy", "Ceramic", "Composite", "Diffusion", "Elasticity",
            "Fatigue", "Graphene", "Hardness", "Magnetism", "Microstructure",
            "Nanoparticle", "Polymer", "Quenching", "Semiconductors", "Strain",
            "Stress", "Superconductor", "Tensile", "Thermodynamics", "Tribology",
            "Viscosity", "Wear", "Yield", "Crystallography", "Adhesion",
            "Amorphous", "Corrosion", "Ductility", "Fracture", "Grain",
            "Lattice", "Metallurgy", "Morphology", "Phase", "Plasticity",
            "Porosity", "Refractory", "Resilience", "Solubility", "Toughness"
        ]
    }
    
    def __init__(self, domains: List[str], 
                 sim_mapping_path: Optional[str] = None,
                 embedding_path: Optional[str] = None,
                 embedding_type: str = "fasttext",
                 thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize the DomainFilter.
        
        Args:
            domains: List of domains to filter for
            sim_mapping_path: Path to token similarity mapping
            embedding_path: Path to embedding model or vectors
            embedding_type: Type of embedding ("fasttext", "word2vec", etc.)
            thresholds: Dictionary mapping domains to similarity thresholds
        """
        self.domains = domains
        self.embedding_path = embedding_path
        self.embedding_type = embedding_type
        
        # Default thresholds based on empirical testing
        self.thresholds = {
            "astronomy": 0.242,
            "law": 0.235,
            "medicine": 0.238,
            "material_science": 0.240
        }
        
        # Override with provided thresholds
        if thresholds:
            self.thresholds.update(thresholds)
        
        # Load token similarity mapping if provided
        self.token_sim_mapping = None
        if sim_mapping_path and os.path.exists(sim_mapping_path):
            self._load_sim_mapping(sim_mapping_path)
        
        # Load embeddings if provided
        self.embedding_model = None
        self.domain_embeddings = {}
        if embedding_path and os.path.exists(embedding_path):
            self._load_embeddings(embedding_path, embedding_type)
    
    def _load_sim_mapping(self, path: str):
        """Load token similarity mapping from file."""
        try:
            with open(path, 'rb') as f:
                self.token_sim_mapping = pickle.load(f)
            print(f"Loaded token similarity mapping with {len(self.token_sim_mapping)} tokens")
        except Exception as e:
            print(f"Error loading token similarity mapping: {e}")
            self.token_sim_mapping = None
    
    def _load_embeddings(self, path: str, embedding_type: str):
        """
        Load embeddings from file.
        
        Args:
            path: Path to embedding file
            embedding_type: Type of embedding
        """
        try:
            if embedding_type.lower() == "fasttext":
                try:
                    import fasttext
                    self.embedding_model = fasttext.load_model(path)
                    print(f"Loaded FastText model from {path}")
                except ImportError:
                    print("FastText not installed. Install with: pip install fasttext")
                    return
                except Exception as e:
                    print(f"Error loading FastText model: {e}")
                    return
            
            # Precompute domain embeddings
            self._compute_domain_embeddings()
            
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.embedding_model = None
    
    def _compute_domain_embeddings(self):
        """Compute average embeddings for each domain's keywords."""
        if not self.embedding_model:
            return
        
        for domain in self.domains:
            keywords = self.DOMAIN_KEYWORDS.get(domain, [])
            if not keywords:
                continue
            
            # Get embeddings for all keywords
            keyword_embeddings = []
            for keyword in keywords:
                if self.embedding_type.lower() == "fasttext":
                    # FastText returns a tuple (list of words, list of embeddings)
                    embedding = self.embedding_model.get_word_vector(keyword.lower())
                    keyword_embeddings.append(embedding)
            
            # Compute average embedding for domain
            if keyword_embeddings:
                self.domain_embeddings[domain] = np.mean(keyword_embeddings, axis=0)
                print(f"Computed domain embedding for {domain} using {len(keyword_embeddings)} keywords")
    
    @staticmethod
    def simple_tokenize(text: str) -> List[str]:
        """Tokenize text into words."""
        return re.findall(r'\w+', text.lower())
    
    def compute_document_similarity(self, text: str) -> Dict[str, float]:
        """
        Compute similarity of a document to each domain.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping domain names to similarity scores
        """
        # If embedding model is available, use it
        if self.embedding_model and self.domain_embeddings:
            return self._compute_embedding_similarity(text)
        
        # If token similarity mapping is available, use it
        if hasattr(self, 'token_sim_mapping') and self.token_sim_mapping is not None:
            return self._compute_token_similarity(text)
        
        # Fallback to keyword-based approach
        return self._compute_keyword_based_similarity(text)
    
    def _compute_embedding_similarity(self, text: str) -> Dict[str, float]:
        """
        Compute similarity using embeddings.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping domain names to similarity scores
        """
        # Preprocess text for FastText (remove newlines and extra whitespace)
        processed_text = ' '.join(text.lower().split())
        
        # Get document embedding
        if self.embedding_type.lower() == "fasttext":
            try:
                # For FastText, we can get sentence vector directly
                doc_embedding = self.embedding_model.get_sentence_vector(processed_text)
            except ValueError as e:
                # If there's still an error, fall back to word-by-word approach
                print(f"Warning: {e}. Falling back to word-by-word embedding.")
                tokens = self.simple_tokenize(text)
                token_embeddings = []
                for token in tokens:
                    embedding = self.embedding_model.get_word_vector(token)
                    token_embeddings.append(embedding)
                
                if not token_embeddings:
                    return {domain: 0.0 for domain in self.domains}
                
                doc_embedding = np.mean(token_embeddings, axis=0)
        else:
            # For other models, average word vectors
            tokens = self.simple_tokenize(text)
            token_embeddings = []
            for token in tokens:
                if self.embedding_type.lower() == "fasttext":
                    embedding = self.embedding_model.get_word_vector(token)
                    token_embeddings.append(embedding)
            
            if not token_embeddings:
                return {domain: 0.0 for domain in self.domains}
            
            doc_embedding = np.mean(token_embeddings, axis=0)
        
        # Compute cosine similarity with each domain embedding
        domain_scores = {}
        for domain, domain_embedding in self.domain_embeddings.items():
            # Cosine similarity
            dot_product = np.dot(doc_embedding, domain_embedding)
            norm_doc = np.linalg.norm(doc_embedding)
            norm_domain = np.linalg.norm(domain_embedding)
            
            if norm_doc == 0 or norm_domain == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm_doc * norm_domain)
            
            domain_scores[domain] = similarity
        
        return domain_scores
    
    def _compute_token_similarity(self, text: str) -> Dict[str, float]:
        """
        Compute similarity using token similarity mapping.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping domain names to similarity scores
        """
        tokens = self.simple_tokenize(text)
        domain_scores = {domain: 0.0 for domain in self.domains}
        token_count = 0
        
        for token in tokens:
            if token in self.token_sim_mapping:
                token_count += 1
                for domain in self.domains:
                    domain_scores[domain] += self.token_sim_mapping[token].get(domain, 0.0)
        
        # Normalize scores
        if token_count > 0:
            for domain in domain_scores:
                domain_scores[domain] /= token_count
        
        return domain_scores

    def _compute_keyword_based_similarity(self, text: str) -> Dict[str, float]:
        """
        Compute similarity based on keyword presence when no embedding is available.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary mapping domain names to similarity scores
        """
        text_lower = text.lower()
        domain_scores = {}
        
        for domain in self.domains:
            # Get keywords for this domain
            keywords = self.DOMAIN_KEYWORDS.get(domain, [])
            if not keywords:
                domain_scores[domain] = 0.0
                continue
            
            # Count keyword matches
            match_count = 0
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    match_count += 1
            
            # Calculate score as percentage of keywords matched, with a scaling factor
            # Adjust scaling factor to make it harder to reach 1.0
            score = min(1.0, match_count / (len(keywords) * 0.2))  # Increased from 0.1 to 0.2
            domain_scores[domain] = score
        
        return domain_scores
    
    def filter_document(self, text: str) -> Dict[str, bool]:
        """
        Determine if a document passes the threshold for each domain.
        
        Args:
            text: The document text to filter
            
        Returns:
            Dictionary mapping domain names to boolean (pass/fail)
        """
        similarities = self.compute_document_similarity(text)
        return {
            domain: similarities.get(domain, 0.0) >= self.thresholds.get(domain, 0.0)
            for domain in self.domains
        }
    
    def process_parquet_file(self, parquet_path: str, output_dir: str) -> Dict[str, int]:
        """
        Process a parquet file and extract domain-specific content.
        
        Args:
            parquet_path: Path to the parquet file
            output_dir: Directory to save filtered results
            
        Returns:
            Statistics about the filtering process
        """
        base_name = os.path.basename(parquet_path).replace('.parquet', '')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output files for each domain
        domain_files = {}
        for domain in self.domains:
            domain_code = domain.split('_')[0][:3]  # First 3 chars of first word
            out_path = os.path.join(output_dir, f"{base_name}_{domain_code}.jsonl")
            domain_files[domain] = open(out_path, 'w', encoding='utf-8')
        
        # Initialize statistics
        stats = {
            'document_count': 0,
            **{f"{domain}_kept": 0 for domain in self.domains}
        }
        
        # Process the parquet file
        try:
            pf = pq.ParquetFile(parquet_path)
            
            # Process in batches
            for batch in tqdm(pf.iter_batches(batch_size=self.batch_size, columns=['text']),
                             desc=f"Processing {base_name}", leave=False):
                texts = batch.column('text').to_pylist()
                
                for text in texts:
                    stats['document_count'] += 1
                    domain_results = self.filter_document(text)
                    
                    # Save document to domain-specific files if it passes the threshold
                    for domain, passes in domain_results.items():
                        if passes:
                            domain_files[domain].write(json.dumps({"text": text}) + '\n')
                            stats[f"{domain}_kept"] += 1
                
                # Garbage collection to manage memory
                gc.collect()
                
        except Exception as e:
            print(f"Error processing {parquet_path}: {e}")
        finally:
            # Close all output files
            for f in domain_files.values():
                f.close()
                
        return stats
    
    def process_dataset(self, input_dir: str, output_dir: str, 
                        num_workers: int = 16) -> Dict[str, Dict[str, int]]:
        """
        Process all parquet files in a directory using multiple workers.
        
        Args:
            input_dir: Directory containing parquet files
            output_dir: Directory to save filtered results
            num_workers: Number of parallel workers
            
        Returns:
            Statistics for each processed file
        """
        parquet_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir)
                         if f.endswith('.parquet')]
        
        if not parquet_files:
            print("No parquet files found.")
            return {}
        
        print(f"Starting parallel processing with {num_workers} workers.")
        
        # Initialize worker pool with shared data
        with mp.Pool(processes=num_workers, 
                     initializer=self._worker_init,
                     initargs=(self.thresholds, self.sim_mapping_path)) as pool:
            # Process files in parallel
            results = list(tqdm(
                pool.starmap(
                    self._process_file_worker, 
                    [(pf, output_dir) for pf in parquet_files]
                ),
                total=len(parquet_files),
                desc="Processing files"
            ))
        
        # Combine results
        overall_results = {}
        for res in results:
            overall_results.update(res)
            
        return overall_results
    
    @staticmethod
    def _worker_init(thresholds, sim_mapping_path):
        """Initialize worker process with shared data."""
        global worker_thresholds, worker_token_sim_mapping
        worker_thresholds = thresholds
        
        with open(sim_mapping_path, 'rb') as f:
            worker_token_sim_mapping = pickle.load(f)
    
    @staticmethod
    def _process_file_worker(parquet_path, output_dir):
        """Worker function for processing a single parquet file."""
        global worker_thresholds, worker_token_sim_mapping
        
        # Create a temporary DomainFilter instance for this worker
        filter_instance = DomainFilter()
        filter_instance.thresholds = worker_thresholds
        filter_instance.token_sim_mapping = worker_token_sim_mapping
        
        # Process the file
        stats = filter_instance.process_parquet_file(parquet_path, output_dir)
        return {parquet_path: stats}


class QualityEvaluator:
    """
    Evaluates content quality using a classifier.
    
    This class implements the second stage of the ORBIT curation pipeline,
    filtering out low-quality content based on various quality metrics.
    It can use either a transformer-based model (like DistilBERT) or a
    simple TF-IDF + Logistic Regression model for faster processing.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 quality_threshold: float = 0.5,
                 device: Optional[str] = None,
                 model_type: str = "simple"):
        """
        Initialize the QualityEvaluator.
        
        Args:
            model_path: Path to pre-trained quality evaluation model
            quality_threshold: Threshold for quality score (0-1)
            device: Device to run model on ('cpu', 'cuda', etc.)
            model_type: Type of model to use ('transformer' or 'simple')
        """
        self.model_path = model_path
        self.quality_threshold = quality_threshold
        self.model_type = model_type
        
        # Set device
        if device is None:
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Model and tokenizer will be loaded on first use
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained quality evaluation model."""
        try:
            print(f"Loading quality evaluation model from {self.model_path}")
            
            if self.model_type == "transformer":
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                self.model.to(self.device)
                self.model.eval()
                print("Transformer-based quality evaluation model loaded successfully")
            else:
                # Load simple model (TF-IDF + LogisticRegression)
                import pickle
                
                with open(os.path.join(self.model_path, "vectorizer.pkl"), "rb") as f:
                    self.vectorizer = pickle.load(f)
                
                with open(os.path.join(self.model_path, "classifier.pkl"), "rb") as f:
                    self.model = pickle.load(f)
                
                print("Simple quality evaluation model loaded successfully")
                
        except Exception as e:
            print(f"Error loading quality evaluation model: {e}")
            self.model = None
            self.tokenizer = None
            self.vectorizer = None
    
    def evaluate_quality(self, text: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a document.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary with quality evaluation results
        """
        # If no model is loaded, return default values
        if self.model is None:
            return {
                "quality_score": 0.0,
                "passes_threshold": False,
                "error": "No quality evaluation model loaded"
            }
        
        try:
            if self.model_type == "transformer":
                import torch
                
                # Preprocess text
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512,
                    padding="max_length"
                ).to(self.device)
                
                # Get model predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Process outputs to get quality score
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                quality_score = probabilities[0][1].item()  # Assuming binary classification (0=low, 1=high quality)
            else:
                # Use simple model
                # Transform text using TF-IDF vectorizer
                features = self.vectorizer.transform([text])
                
                # Get probability prediction
                probabilities = self.model.predict_proba(features)
                quality_score = probabilities[0][1]  # Assuming binary classification (0=low, 1=high quality)
            
            return {
                "quality_score": quality_score,
                "passes_threshold": quality_score >= self.quality_threshold
            }
            
        except Exception as e:
            return {
                "quality_score": 0.0,
                "passes_threshold": False,
                "error": str(e)
            }
    
    def train_model(self, 
                   train_data: List[Dict[str, Any]],
                   val_data: Optional[List[Dict[str, Any]]] = None,
                   output_dir: str = "quality_model",
                   base_model: str = "distilbert-base-uncased",
                   epochs: int = 3,
                   batch_size: int = 8,
                   learning_rate: float = 2e-5,
                   model_type: str = "simple") -> str:
        """
        Train a quality evaluation model.
        
        Args:
            train_data: List of documents with text and quality labels
            val_data: Optional validation data
            output_dir: Directory to save the model
            base_model: Base model to use (for transformer models)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for training
            model_type: Type of model to train ('transformer' or 'simple')
            
        Returns:
            Path to the trained model
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Extract texts and labels
            train_texts = [doc["text"] for doc in train_data]
            train_labels = [int(doc["quality"]) for doc in train_data]
            
            if model_type == "transformer":
                # Train transformer-based model
                from transformers import (
                    AutoTokenizer, 
                    AutoModelForSequenceClassification,
                    TrainingArguments, 
                    Trainer,
                    DataCollatorWithPadding
                )
                from datasets import Dataset
                import numpy as np
                from sklearn.metrics import precision_recall_fscore_support, accuracy_score
                import torch
                
                # Create datasets
                train_dataset = Dataset.from_dict({
                    "text": train_texts,
                    "label": train_labels
                })
                
                if val_data:
                    val_texts = [doc["text"] for doc in val_data]
                    val_labels = [int(doc["quality"]) for doc in val_data]
                    
                    val_dataset = Dataset.from_dict({
                        "text": val_texts,
                        "label": val_labels
                    })
                else:
                    # Split train data if no validation data provided
                    train_val = train_dataset.train_test_split(test_size=0.1)
                    train_dataset = train_val["train"]
                    val_dataset = train_val["test"]
                
                # Load tokenizer and model
                tokenizer = AutoTokenizer.from_pretrained(base_model)
                model = AutoModelForSequenceClassification.from_pretrained(
                    base_model, num_labels=2
                )
                
                # Tokenize data
                def tokenize_function(examples):
                    return tokenizer(
                        examples["text"], 
                        padding="max_length", 
                        truncation=True, 
                        max_length=512
                    )
                
                print("Tokenizing training data...")
                tokenized_train = train_dataset.map(tokenize_function, batched=True)
                print("Tokenizing validation data...")
                tokenized_val = val_dataset.map(tokenize_function, batched=True)
                
                # Define metrics function
                def compute_metrics(eval_pred):
                    logits, labels = eval_pred
                    predictions = np.argmax(logits, axis=-1)
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        labels, predictions, average='binary'
                    )
                    acc = accuracy_score(labels, predictions)
                    return {
                        'accuracy': acc,
                        'f1': f1,
                        'precision': precision,
                        'recall': recall
                    }
                
                # Set up training arguments
                training_args = TrainingArguments(
                    output_dir=output_dir,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch_size,
                    per_device_eval_batch_size=batch_size,
                    warmup_steps=500,
                    weight_decay=0.01,
                    logging_dir=os.path.join(output_dir, "logs"),
                    logging_steps=10,
                    evaluation_strategy="epoch",
                    save_strategy="epoch",
                    load_best_model_at_end=True,
                    metric_for_best_model="f1",
                    greater_is_better=True,
                    report_to="none",  # Disable wandb, etc.
                )
                
                # Create trainer
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=tokenized_train,
                    eval_dataset=tokenized_val,
                    tokenizer=tokenizer,
                    data_collator=DataCollatorWithPadding(tokenizer),
                    compute_metrics=compute_metrics,
                )
                
                # Train model
                print("Training transformer-based quality evaluation model...")
                trainer.train()
                
                # Evaluate model
                eval_results = trainer.evaluate()
                print(f"Evaluation results: {eval_results}")
                
                # Save model
                model_path = os.path.join(output_dir, "final_model")
                trainer.save_model(model_path)
                tokenizer.save_pretrained(model_path)
                
                # Update instance model
                self.model_path = model_path
                self.model = model
                self.tokenizer = tokenizer
                self.model_type = "transformer"
                
                print(f"Transformer model saved to {model_path}")
                return model_path
                
            else:
                # Train simple model (TF-IDF + LogisticRegression)
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.linear_model import LogisticRegression
                from sklearn.pipeline import Pipeline
                from sklearn.metrics import classification_report
                from sklearn.model_selection import train_test_split
                import pickle
                from tqdm import tqdm
                
                print("Training simple quality evaluation model...")
                
                # Split data if validation data not provided
                if val_data:
                    val_texts = [doc["text"] for doc in val_data]
                    val_labels = [int(doc["quality"]) for doc in val_data]
                else:
                    train_texts, val_texts, train_labels, val_labels = train_test_split(
                        train_texts, train_labels, test_size=0.1, random_state=42
                    )
                
                # Create and train vectorizer
                print("Creating TF-IDF vectorizer...")
                vectorizer = TfidfVectorizer(
                    max_features=10000,
                    min_df=2,
                    max_df=0.8,
                    ngram_range=(1, 2)
                )
                
                # Transform training data
                print("Transforming training data...")
                X_train = vectorizer.fit_transform(tqdm(train_texts, desc="Vectorizing training texts"))
                
                # Train classifier
                print("Training logistic regression classifier...")
                classifier = LogisticRegression(
                    C=1.0,
                    max_iter=100,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                )
                classifier.fit(X_train, train_labels)
                
                # Evaluate on validation data
                print("Evaluating model...")
                X_val = vectorizer.transform(tqdm(val_texts, desc="Vectorizing validation texts"))
                val_predictions = classifier.predict(X_val)
                
                # Print classification report
                report = classification_report(val_labels, val_predictions)
                print(f"Classification Report:\n{report}")
                
                # Save model
                model_path = output_dir
                os.makedirs(model_path, exist_ok=True)
                
                with open(os.path.join(model_path, "vectorizer.pkl"), "wb") as f:
                    pickle.dump(vectorizer, f)
                
                with open(os.path.join(model_path, "classifier.pkl"), "wb") as f:
                    pickle.dump(classifier, f)
                
                # Update instance variables
                self.model_path = model_path
                self.vectorizer = vectorizer
                self.model = classifier
                self.model_type = "simple"
                
                print(f"Simple model saved to {model_path}")
                return model_path
                
        except Exception as e:
            print(f"Error training quality evaluation model: {e}")
            import traceback
            traceback.print_exc()
            return ""


class DatasetCurator:
    """
    Main class for curating domain-specific datasets.
    
    This class orchestrates the full ORBIT curation pipeline, combining
    domain filtering and quality evaluation to create high-quality
    domain-specific datasets.
    """
    
    def __init__(self, domain: str = None, 
                 sim_mapping_path: str = None,
                 quality_threshold: float = 0.5,
                 model_type: str = "simple"):
        """
        Initialize the DatasetCurator.
        
        Args:
            domain: Primary domain to curate for
            sim_mapping_path: Path to token similarity mapping
            quality_threshold: Threshold for quality score (0-1)
            model_type: Type of quality model to use ('transformer' or 'simple')
        """
        self.domain = domain
        self.sim_mapping_path = sim_mapping_path
        self.quality_threshold = quality_threshold
        self.model_type = model_type
        
        # Initialize components
        domains = [domain] if domain else None
        self.domain_filter = DomainFilter(domains=domains, sim_mapping_path=sim_mapping_path)
        self.quality_evaluator = QualityEvaluator(quality_threshold=quality_threshold, model_type=model_type)
    
    def process_dataset(self, input_path: str, output_dir: str = None,
                        evaluate_quality: bool = True,
                        num_workers: int = 16,
                        max_docs: int = None) -> Dict[str, Any]:
        """
        Process a dataset through the full curation pipeline.
        
        Args:
            input_path: Path to input dataset (directory of parquet files or single file)
            output_dir: Directory to save results
            evaluate_quality: Whether to perform quality evaluation
            num_workers: Number of parallel workers for filtering
            max_docs: Maximum documents to process
            
        Returns:
            Statistics and metadata about the curation process
        """
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(input_path), "orbit_curated")
        
        # Create output directories
        filtered_dir = os.path.join(output_dir, "filtered")
        evaluated_dir = os.path.join(output_dir, "evaluated")
        final_dir = os.path.join(output_dir, "final")
        
        os.makedirs(filtered_dir, exist_ok=True)
        if evaluate_quality:
            os.makedirs(evaluated_dir, exist_ok=True)
        os.makedirs(final_dir, exist_ok=True)
        
        # Step 1: Domain Filtering
        print("Step 1: Filtering documents by domain relevance")
        if os.path.isdir(input_path):
            filter_stats = self.domain_filter.process_dataset(
                input_path, filtered_dir, num_workers
            )
        else:
            # Single file processing
            filter_stats = {
                input_path: self.domain_filter.process_parquet_file(input_path, filtered_dir)
            }
        
        # Step 2: Quality Evaluation (if enabled)
        eval_stats = {}
        if evaluate_quality:
            print("Step 2: Evaluating document quality")
            
            # Create domain mapping for evaluation
            domain_mapping = {}
            if self.domain:
                # If specific domain is set, only evaluate that domain
                domain_code = self.domain.split('_')[0][:3]
                domain_mapping[f"_{domain_code}.jsonl"] = self.domain
            else:
                # Otherwise, evaluate all domains
                domain_mapping = {
                    "_astro.jsonl": "astronomy",
                    "_law.jsonl": "law",
                    "_med.jsonl": "medicine",
                    "_mat.jsonl": "material_science"
                }
            
            # Process the filtered files
            eval_results = self.quality_evaluator.process_dataset(
                filtered_dir, domain_mapping, evaluated_dir, 
                max_docs_per_file=max_docs or 10000
            )
            
            # Compile statistics
            eval_stats = {
                domain: {
                    "total_evaluated": len(results),
                    "high_quality": sum(1 for r in results if r.get("score", 0) >= self.quality_threshold)
                }
                for domain, results in eval_results.items()
            }
            
            # Create final dataset with high-quality documents
            print("Step 3: Creating final high-quality dataset")
            for domain, results in eval_results.items():
                domain_code = domain.split('_')[0][:3]
                final_path = os.path.join(final_dir, f"{domain_code}_high_quality.jsonl")
                
                with open(final_path, "w", encoding="utf-8") as outfile:
                    for entry in results:
                        if entry.get("score", 0) >= self.quality_threshold:
                            # Extract just the original text for the final dataset
                            doc = entry.get("input_document", {})
                            outfile.write(json.dumps(doc) + "\n")
                
                print(f"Created high-quality dataset for {domain} at {final_path}")
        
        # Compile overall statistics
        stats = {
            "filtering": filter_stats,
            "evaluation": eval_stats,
            "quality_threshold": self.quality_threshold,
            "domains": self.domain_filter.domains,
            "output_dir": output_dir
        }
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "curation_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        
        return stats
    
    def prepare_dataset(self, input_path: str, output_format: str = "jsonl",
                    split: bool = True, train_ratio: float = 0.9) -> str:
        """
        Prepare a curated dataset for model training.
        
        Args:
            input_path: Path to curated dataset directory or file
            output_format: Format for the output dataset (jsonl, csv, etc.)
            split: Whether to split into train/validation sets
            train_ratio: Ratio of data to use for training
            
        Returns:
            Path to the prepared dataset (directory or file, depending on split)
        """
        import os, json, random, csv

        # Collect documents from input path
        docs = []
        if os.path.isdir(input_path):
            # Look for JSONL files within the directory
            for filename in os.listdir(input_path):
                if filename.endswith(".jsonl"):
                    file_path = os.path.join(input_path, filename)
                    with open(file_path, "r", encoding="utf-8") as infile:
                        for line in infile:
                            try:
                                doc = json.loads(line)
                                docs.append(doc)
                            except Exception:
                                continue
        else:
            # Input is a single file
            with open(input_path, "r", encoding="utf-8") as infile:
                for line in infile:
                    try:
                        doc = json.loads(line)
                        docs.append(doc)
                    except Exception:
                        continue

        if not docs:
            print("No documents found in input_path.")
            return ""

        # Shuffle the documents to randomize train/validation splits
        random.shuffle(docs)

        # Create output directory for the prepared dataset
        output_dir = os.path.join(os.path.dirname(input_path), "prepared_dataset")
        os.makedirs(output_dir, exist_ok=True)

        if split:
            train_size = int(len(docs) * train_ratio)
            train_docs = docs[:train_size]
            val_docs = docs[train_size:]

            train_path = os.path.join(output_dir, f"train.{output_format}")
            val_path = os.path.join(output_dir, f"validation.{output_format}")

            if output_format == "jsonl":
                with open(train_path, "w", encoding="utf-8") as outfile:
                    for doc in train_docs:
                        outfile.write(json.dumps(doc) + "\n")
                with open(val_path, "w", encoding="utf-8") as outfile:
                    for doc in val_docs:
                        outfile.write(json.dumps(doc) + "\n")
            elif output_format == "csv":
                # Assume documents are flat dictionaries
                if train_docs:
                    fieldnames = list(train_docs[0].keys())
                    with open(train_path, "w", newline="", encoding="utf-8") as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(train_docs)
                    with open(val_path, "w", newline="", encoding="utf-8") as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(val_docs)
            else:
                print("Unsupported output format. Supported formats: jsonl, csv")
                return ""

            print(f"Prepared dataset with {len(train_docs)} training and {len(val_docs)} validation documents in '{output_dir}'")
            return output_dir

        else:
            # No split; output a single file
            dataset_path = os.path.join(output_dir, f"dataset.{output_format}")
            if output_format == "jsonl":
                with open(dataset_path, "w", encoding="utf-8") as outfile:
                    for doc in docs:
                        outfile.write(json.dumps(doc) + "\n")
            elif output_format == "csv":
                if docs:
                    fieldnames = list(docs[0].keys())
                    with open(dataset_path, "w", newline="", encoding="utf-8") as outfile:
                        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(docs)
            else:
                print("Unsupported output format. Supported formats: jsonl, csv")
                return ""

            print(f"Prepared dataset with {len(docs)} documents at '{dataset_path}'")
            return dataset_path
