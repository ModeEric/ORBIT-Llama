import time
import os
import random
import json
import logging
import sys
from typing import List, Dict, Any, Optional, Callable

import nltk
from nltk.tokenize import word_tokenize
import ahocorasick
import gensim
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# **New Import for FastText**
import fasttext
import fasttext.util

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("script.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Ensure NLTK data is downloaded
nltk.download('punkt')

# =========================
# Configuration Parameters
# =========================

GLOVE_DIR = './glove_embeddings/'  # Directory where GloVe embeddings are stored
GLOVE_100D = os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')
GLOVE_300D = os.path.join(GLOVE_DIR, 'glove.6B.300d.txt')

# **Updated Configuration for FastText**
FASTTEXT_DIR = './fasttext_embeddings/'  # Directory where FastText embeddings are stored
FASTTEXT_BIN = os.path.join(FASTTEXT_DIR, 'cc.en.300.bin')  # FastText binary file downloaded via fasttext.util.download_model

DATASET_NAME = 'HuggingFaceFW/fineweb-edu'  # Hugging Face dataset name
OUTPUT_DIR = 'filtering_results_jsonl'  # Directory to save the JSONL files

# **Updated FILTERING_METHODS: Only multiples of five from 5-50**
FILTERING_METHODS = [
    'no_filter',
    # Word filters for multiples of five from 5 to 50 will be added dynamically below
    # Threshold filters will be added dynamically below
]

WORD_FILTER_KEYWORDS = [
    'Albedo', 'Aphelion', 'Apogee', 'Asteroid', 'Astronomy', 'Aurora', 'Axion', 'Azimuth',
    'Barycenter', 'Baryon', 'Blackbody', 'Bolide', 'Brilliance', 'Cepheid', 'Comet',
    'Constellation', 'Corona', 'Cosmic', 'Cosmology', 'DESC', 'Dyne', 'Eclipse',
    'Ecliptic', 'Emission', 'Erg', 'Exoplanet', 'Extinction', 'Fluence', 'Frequency',
    'Galaxy', 'Geocentric', 'Gibbous', 'Gravity', 'Heliocentric', 'Interferometry',
    'Isotropic', 'JWST', 'kpc', 'Light-Year', 'LSST', 'Luminosity', 'Magnetar',
    'Magnetosphere', 'Metallicity', 'Meteor', 'Meteorite', 'Microlensing', 'Moon',
    'Morphology', 'Multiverse', 'Nebula', 'Neutrino', 'Noctilucent', 'Nova',
    'Nucleosynthesis', 'Orbit', 'Parallax', 'Parsec', 'Perihelion', 'Phase',
    'Photometry', 'Photosphere', 'Planck', 'Planetesimal', 'Pulsar', 'Quasar',
    'Quiescence', 'Recombination', 'Reddening', 'Redshift', 'Reionization',
    'Satellite', 'Seyfert', 'Simulation', 'Singularity', 'Spectroscopy', 'SPT',
    'Sublimation', 'Sunspot', 'Supercomputer', 'Supermassive', 'Supernova',
    'Telescope', 'Transit', 'Universe', 'Voids', 'Wavelength', 'Waxing',
    'Wormhole', 'X-ray', 'Zenith', 'Zodiac', 'Optical', 'Infrared', 'Ultraviolet',
    'Microwave', 'Proton', 'Neutron', 'Electron', 'Flux', 'Intensity', 'Companion',
    'Outflow', 'QSO', 'Pulse', 'Progenitor'
]

# =========================
# Utility Functions
# =========================

def load_glove_embeddings(glove_path: str, embedding_dim: int) -> gensim.models.KeyedVectors:
    """
    Load GloVe embeddings using gensim.

    Parameters:
    - glove_path: Path to the GloVe file.
    - embedding_dim: Dimension of the embeddings (100 or 300).

    Returns:
    - gensim KeyedVectors object.
    """
    if not os.path.exists(glove_path):
        logger.error(f"GloVe file not found at {glove_path}.")
        raise FileNotFoundError(f"GloVe file not found at {glove_path}.")

    logger.info(f"Loading GloVe embeddings from {glove_path} with dimension {embedding_dim}...")
    # gensim's KeyedVectors can load GloVe format by specifying no_header=True
    glove_model = gensim.models.KeyedVectors.load_word2vec_format(glove_path, binary=False, no_header=True)
    logger.info(f"Loaded {len(glove_model.key_to_index)} word vectors from {glove_path}.")
    return glove_model

# **Modified Utility Function for FastText**
def load_fasttext_embeddings(fasttext_dir: str, lang: str = 'en') -> fasttext.FastText:
    """
    Load FastText embeddings using the fasttext library.

    Parameters:
    - fasttext_dir: Directory where FastText embeddings are stored.
    - lang: Language code for the FastText model.

    Returns:
    - fasttext.FastText model.
    """
    os.makedirs(fasttext_dir, exist_ok=True)
    model_filename = f'cc.{lang}.300.bin'
    model_path = os.path.join(fasttext_dir, model_filename)

    if not os.path.exists(model_path):
        logger.info(f"Downloading FastText model for language '{lang}'...")
        # Download the FastText model. It downloads to the current directory by default.
        downloaded_path = fasttext.util.download_model(lang, if_exists='ignore')  # Downloads 'cc.en.300.bin'
        logger.info(f"Moving downloaded FastText model to '{fasttext_dir}'...")
        os.rename(downloaded_path, model_path)

    logger.info(f"Loading FastText model from '{model_path}'...")
    fasttext_model = fasttext.load_model(model_path)
    logger.info(f"FastText model loaded from '{model_path}'.")
    return fasttext_model

def build_aho_corasick(keywords: List[str]) -> ahocorasick.Automaton:
    """
    Build the Aho-Corasick automaton for keyword filtering.

    Parameters:
    - keywords: List of keywords to include in the automaton.

    Returns:
    - ahocorasick.Automaton object.
    """
    A = ahocorasick.Automaton()
    for idx, key in enumerate(keywords):
        A.add_word(key.lower(), (idx, key))
    A.make_automaton()
    logger.info("Aho-Corasick automaton built with specified keywords.")
    return A

def compute_astronomy_embedding(keywords: List[str], model: Any) -> np.ndarray:
    """
    Compute the aggregated astronomy embedding by averaging the embeddings of the keywords.

    Parameters:
    - keywords: List of astronomy-related keywords.
    - model: Embedding model (gensim KeyedVectors or fasttext.FastText).

    Returns:
    - Normalized numpy array representing the astronomy embedding.
    """
    vectors = []
    missing_keywords = []
    for word in keywords:
        try:
            if hasattr(model, 'get_word_vector'):
                vector = model.get_word_vector(word.lower())
            elif word.lower() in model:
                vector = model[word.lower()]
            else:
                raise KeyError
            vectors.append(vector)
        except KeyError:
            missing_keywords.append(word)
    if missing_keywords:
        logger.warning(f"The following keywords are missing in the model and will be skipped: {missing_keywords}")
    if not vectors:
        logger.error("None of the astronomy keywords are present in the embeddings.")
        raise ValueError("None of the astronomy keywords are present in the embeddings.")
    astronomy_vector = np.mean(vectors, axis=0)
    # Normalize the vector
    norm = np.linalg.norm(astronomy_vector)
    if norm != 0:
        astronomy_vector /= norm
    logger.info("Astronomy aggregated embedding computed and normalized.")
    return astronomy_vector

def compute_document_embedding(text: str, model: Any, embedding_dim: int) -> np.ndarray:
    """
    Compute the embedding of a document by averaging its word embeddings.

    Parameters:
    - text: The document text.
    - model: Embedding model (gensim KeyedVectors or fasttext.FastText).
    - embedding_dim: Dimension of the embeddings (100, 300).

    Returns:
    - Normalized numpy array representing the document embedding.
    """
    tokens = word_tokenize(text.lower())
    vectors = []
    for word in tokens:
        try:
            if hasattr(model, 'get_word_vector'):
                vector = model.get_word_vector(word)
            elif word in model:
                vector = model[word]
            else:
                continue
            vectors.append(vector)
        except KeyError:
            continue
    if not vectors:
        return np.zeros(embedding_dim)
    doc_embedding = np.mean(vectors, axis=0)
    # Normalize the vector
    norm = np.linalg.norm(doc_embedding)
    if norm != 0:
        doc_embedding /= norm
    return doc_embedding

# =========================
# Filtering Classes
# =========================

class Filter:
    """
    Base filter class that keeps all documents.
    """
    def filter(self, text: str) -> bool:
        return True  # By default, keep all documents

class WordFilter(Filter):
    """
    Filters documents containing a minimum number of specified keywords using Aho-Corasick.
    """
    def __init__(self, keywords: List[str], min_matches: int = 1):
        self.automaton = build_aho_corasick(keywords)
        self.min_matches = min_matches

    def filter(self, text: str) -> bool:
        match_count = 0
        for _, (_, _) in self.automaton.iter(text.lower()):
            match_count += 1
            if match_count >= self.min_matches:
                return True
        return False

class ThresholdEmbeddingFilter(Filter):
    """
    Filters documents by keeping those with cosine similarity above a fixed threshold with the astronomy aggregated embedding.
    """
    def __init__(
        self,
        compute_embedding: Callable[[str], np.ndarray],
        astronomy_vector: np.ndarray,
        threshold: float
    ):
        """
        Initialize the ThresholdEmbeddingFilter.

        Parameters:
        - compute_embedding: Function to compute the embedding of a document.
        - astronomy_vector: numpy array representing the astronomy embedding.
        - threshold: Cosine similarity threshold.
        """
        self.compute_embedding = compute_embedding
        self.astronomy_vector = astronomy_vector
        self.threshold = threshold
        self.kept_documents: List[str] = []
        self.total_processing_time = 0.0
        self.document_count = 0

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        """
        if not np.any(vec1) or not np.any(vec2):
            return 0.0
        return float(np.dot(vec1, vec2))

    def filter(self, text: str) -> bool:
        """
        Keep the document if its similarity with the astronomy vector is above the threshold.
        """
        start_time = time.time()
        doc_embedding = self.compute_embedding(text)
        similarity = self.cosine_similarity(doc_embedding, self.astronomy_vector)
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.document_count += 1

        if similarity >= self.threshold:
            self.kept_documents.append(text)
            return True
        return False

    def get_average_processing_time(self) -> float:
        """
        Calculate the average processing time per document.
        """
        if self.document_count == 0:
            return 0.0
        return self.total_processing_time / self.document_count

# =========================
# Dataset Loading Function
# =========================

def load_hf_dataset_streaming(dataset_name: str, split: str = 'train') -> Any:
    """
    Load a dataset from Hugging Face in streaming mode.

    Parameters:
    - dataset_name: Name of the Hugging Face dataset.
    - split: Dataset split to load.

    Returns:
    - Iterable of document texts.
    """
    logger.info(f"Loading Hugging Face dataset: {dataset_name}, split: {split} with streaming...")
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    logger.info("Dataset loaded in streaming mode.")
    return (example['text'] for example in dataset)

# =========================
# Main Testing Function
# =========================

def run_tests_streaming(
    dataset_stream: Any,
    filters: Dict[str, Filter],
    top_n_filters: List[str],
    max_documents: int = 1000*100  # Increased to 1,000,000 documents
) -> Dict[str, Any]:
    """
    Run filtering methods on a streaming dataset and collect documents per method based on their criteria.

    Parameters:
    - dataset_stream: Iterable stream of document texts.
    - filters: Dictionary of filtering methods.
    - top_n_filters: List of filter method names that require top N selection.
    - max_documents: Maximum number of documents to process to prevent infinite loops.

    Returns:
    - results: Dictionary containing metrics and kept documents for each filtering method.
    """
    results = {
        method: {
            'kept_documents': [],
            'total_processing_time': 0.0,  # Accumulated processing time
            'document_count': 0,            # Number of documents processed
            'percent_kept': 0.0,
            'similarity_threshold_top_1_percent': None  # Added to store similarity threshold
        } for method in filters.keys()
    }

    # Initialize counters for each filter method
    kept_counts = {method: 0 for method in filters.keys()}
    reservoirs = {method: [] for method in filters.keys()}

    logger.info("Starting to process the dataset for each filtering method...")

    # To keep track of total documents processed
    total_processed = 0

    for idx, text in enumerate(tqdm(dataset_stream, desc="Processing documents")):
        if idx >= max_documents:
            logger.warning(f"Reached the maximum number of documents to process: {max_documents}")
            break

        total_processed += 1

        for method, filter_obj in filters.items():
            if method in top_n_filters:
                # Currently, no top N selection filters are present; this block can be omitted or kept for future use
                logger.warning(f"Method '{method}' is marked as top_n but no top N selection filters are present.")
            elif isinstance(filter_obj, ThresholdEmbeddingFilter):
                # Threshold-based embedding filters
                keep = filter_obj.filter(text)
                if keep:
                    kept_counts[method] += 1
                    reservoirs[method].append(text)
                    if kept_counts[method] % 10 == 0:  # Reduced logging frequency for brevity
                        logger.info(f"Method '{method}': Collected {kept_counts[method]} documents.")
            elif isinstance(filter_obj, WordFilter):
                # Word-based filters
                keep = filter_obj.filter(text)
                if keep:
                    kept_counts[method] += 1
                    reservoirs[method].append(text)
                    if kept_counts[method] % 10 == 0:  # Reduced logging frequency for brevity
                        logger.info(f"Method '{method}': Collected {kept_counts[method]} documents.")
            elif isinstance(filter_obj, Filter):
                # Handle the 'no_filter' case: collect all documents
                kept_counts[method] += 1
                reservoirs[method].append(text)
                if kept_counts[method] % 1000 == 0:  # Increased logging frequency for 'no_filter'
                    logger.info(f"Method '{method}': Collected {kept_counts[method]} documents.")
            else:
                logger.warning(f"Filter '{method}' is not recognized as a valid filter type.")

    # After processing, calculate metrics
    for method, filter_obj in filters.items():
        kept_documents = reservoirs[method]
        results[method]['kept_documents'] = kept_documents
        kept_counts[method] = len(kept_documents)
        results[method]['percent_kept'] = (kept_counts[method] / total_processed) * 100 if total_processed > 0 else 0.0

        logger.info(f"Method '{method}': Kept {kept_counts[method]} documents out of {total_processed} processed ({results[method]['percent_kept']:.2f}%).")
        
        if isinstance(filter_obj, ThresholdEmbeddingFilter):
            average_time = filter_obj.get_average_processing_time()
            results[method]['average_processing_time'] = average_time
            logger.info(f"Method '{method}': Average processing time per document: {average_time:.6f} seconds.")
        else:
            # Calculate average processing time
            if hasattr(filter_obj, 'total_processing_time') and hasattr(filter_obj, 'document_count') and filter_obj.document_count > 0:
                average_time = filter_obj.total_processing_time / filter_obj.document_count
            else:
                average_time = 0.0
            results[method]['average_processing_time'] = average_time
            logger.info(f"Method '{method}': Average processing time per document: {average_time:.6f} seconds.")

    return results

# =========================
# Entry Point
# =========================

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    try:
        # Ensure output directory exists
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load dataset with streaming
        logger.info("Loading dataset from Hugging Face with streaming...")
        dataset_stream = load_hf_dataset_streaming(DATASET_NAME, split='train')

        # Define individual filters
        logger.info("Initializing filters...")
        no_filter = Filter()

        # **Initialize word filters only for multiples of five from 5 to 50**
        word_filters = {}
        for min_matches in range(5, 51, 5):  # From min_matches=5 to min_matches=50 in steps of 5
            method_name = f'word_filter_min{min_matches}'
            word_filters[method_name] = WordFilter(keywords=WORD_FILTER_KEYWORDS, min_matches=min_matches)
            FILTERING_METHODS.append(method_name)  # Optionally keep the FILTERING_METHODS list updated

        # Load GloVe embeddings using gensim
        logger.info("Loading GloVe embeddings using gensim...")
        glove_100d = load_glove_embeddings(GLOVE_100D, embedding_dim=100)
        glove_300d = load_glove_embeddings(GLOVE_300D, embedding_dim=300)

        # **Load FastText embeddings using fasttext**
        logger.info("Loading FastText embeddings using fasttext...")
        fasttext_model = load_fasttext_embeddings(FASTTEXT_DIR, lang='en')

        # Compute astronomy aggregated embeddings
        logger.info("Computing astronomy aggregated embeddings for GloVe 100d...")
        astronomy_vector_100d = compute_astronomy_embedding(WORD_FILTER_KEYWORDS, glove_100d)
        logger.info("Computing astronomy aggregated embeddings for GloVe 300d...")
        astronomy_vector_300d = compute_astronomy_embedding(WORD_FILTER_KEYWORDS, glove_300d)

        logger.info("Computing astronomy aggregated embeddings for FastText...")
        astronomy_vector_fasttext = compute_astronomy_embedding(WORD_FILTER_KEYWORDS, fasttext_model)

        # **Add threshold-based embedding filters with new thresholds**
        logger.info("Adding threshold-based embedding filters...")
        # **Adjusted thresholds to be more permissive**
        thresholds_fasttext = [.45,.4,0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
        thresholds_glove = [.3,0.25, 0.2, 0.15, 0.1, 0.05, 0.0]

        threshold_filters = {}
        # GloVe 100d thresholds
        for threshold in thresholds_glove:
            method_name = f'threshold_filter_100d_{threshold}'
            threshold_filters[method_name] = ThresholdEmbeddingFilter(
                compute_embedding=lambda text, model=glove_100d: compute_document_embedding(text, model, 100),
                astronomy_vector=astronomy_vector_100d,
                threshold=threshold
            )
        # GloVe 300d thresholds
        for threshold in thresholds_glove:
            method_name = f'threshold_filter_300d_{threshold}'
            threshold_filters[method_name] = ThresholdEmbeddingFilter(
                compute_embedding=lambda text, model=glove_300d: compute_document_embedding(text, model, 300),
                astronomy_vector=astronomy_vector_300d,
                threshold=threshold
            )
        # FastText thresholds
        for threshold in thresholds_fasttext:
            method_name = f'threshold_filter_fasttext_{threshold}'
            threshold_filters[method_name] = ThresholdEmbeddingFilter(
                compute_embedding=lambda text, model=fasttext_model: compute_document_embedding(text, model, 300),
                astronomy_vector=astronomy_vector_fasttext,
                threshold=threshold
            )

        # Initialize filters dictionary without combined filters and top N selection filters
        filters = {
            'no_filter': no_filter,
            **word_filters,  # Add all word filters
            **threshold_filters  # Add all threshold filters
        }

        # Define which filters require top N selection
        top_n_filters = []  # All top N selection filters have been removed

        # Run tests with streaming and per-method reservoir sampling
        logger.info("Running tests for each filtering method...")
        results = run_tests_streaming(
            dataset_stream=dataset_stream,
            filters=filters,
            top_n_filters=top_n_filters,
            max_documents=1000*100  # Increased to 1,000,000 documents
        )

        # =========================
        # Save kept documents per filtering method in JSONL files
        # =========================
        logger.info("Saving kept documents to JSONL files...")
        for method, data in results.items():
            file_path = os.path.join(OUTPUT_DIR, f"{method}.jsonl")
            kept_documents = data['kept_documents']
            logger.info(f"Saving {len(kept_documents)} documents for method '{method}' to {file_path}...")
            with open(file_path, 'w', encoding='utf-8') as f:
                for text in kept_documents:
                    json_line = json.dumps({"text": text})
                    f.write(json_line + '\n')
            logger.info(f"Saved {len(kept_documents)} documents to {file_path}.")

        # Optionally, save metrics to a separate JSON file
        metrics_output_path = os.path.join(OUTPUT_DIR, "metrics.json")
        save_metrics = {}
        for method, data in results.items():
            if isinstance(filters[method], ThresholdEmbeddingFilter):
                save_metrics[method] = {
                    'percent_kept': data.get('percent_kept', 0.0),
                    'average_processing_time': data.get('average_processing_time', 0.0)
                }
            else:
                save_metrics[method] = {
                    'average_processing_time': data.get('average_processing_time', 0.0),
                    'percent_kept': data.get('percent_kept', 0.0)
                }

        logger.info(f"Saving metrics to {metrics_output_path}...")
        with open(metrics_output_path, 'w', encoding='utf-8') as f:
            json.dump(save_metrics, f, indent=4)
        logger.info(f"Metrics saved to {metrics_output_path}")

    except Exception as e:
        logger.exception(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
