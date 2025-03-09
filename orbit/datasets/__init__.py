"""
ORBIT Dataset Curation Tools
============================

This package provides tools for curating domain-specific datasets for training
specialized language models in astronomy, law, medicine, and other domains.

Main Components:
---------------
- DatasetCurator: Main class for dataset curation
- DomainFilter: For filtering content by domain relevance
- QualityEvaluator: For evaluating the quality of domain-specific content
"""

from orbit.datasets.curator import DomainFilter, QualityEvaluator, DatasetCurator

__all__ = [
    "DatasetCurator",
    "DomainFilter",
    "QualityEvaluator"
] 