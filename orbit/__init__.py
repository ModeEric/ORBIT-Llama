"""
ORBIT: Domain-Specific AI for Astronomy, Law, and Medicine
==========================================================

ORBIT is an open-source framework that revolutionizes how we create and evaluate 
domain-specific language models. By combining intelligent dataset curation with 
advanced fine-tuning techniques, ORBIT enables the development of highly specialized 
AI models that excel in domains like astronomy, law, and medicine.

Main Components:
---------------
- DatasetCurator: For domain-specific dataset curation
- OrbitTrainer: For fine-tuning models on domain-specific data
- OrbitEvaluator: For evaluating model performance on domain benchmarks

Example:
--------
>>> from orbit import OrbitTrainer
>>> trainer = OrbitTrainer(domain="astronomy")
>>> dataset = trainer.prepare_dataset("your_dataset")
>>> model = trainer.train(base_model="llama-3b", dataset=dataset)

This package provides tools for creating and evaluating domain-specific language models.
"""

__version__ = "0.1.0"

# Import main components for easy access
from orbit.datasets.curator import DatasetCurator
# TODO: Uncomment these when the modules are implemented
# from orbit.models.trainer import OrbitTrainer
# from orbit.evaluation.benchmarks import OrbitEvaluator

# Define what's available when using `from orbit import *`
__all__ = [
    "DatasetCurator",
    # "OrbitTrainer", 
    # "OrbitEvaluator",
] 