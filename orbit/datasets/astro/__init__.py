"""
ORBIT Astronomy Dataset Tools
=============================

This module provides specialized tools for processing and curating
astronomy-specific datasets for training language models.

Main Components:
---------------
- AstroProcessor: For processing astronomy-specific data
- AstroKeywords: Access to astronomy domain keywords
- AstroUtils: Utility functions for astronomy data processing
"""

from orbit.datasets.astro.processor import AstroProcessor
from orbit.datasets.astro.utils.embedding_utils import AstroEmbedding
from orbit.datasets.astro.utils.filtering_utils import AstroFilter

__all__ = [
    "AstroProcessor",
    "AstroEmbedding",
    "AstroFilter"
] 