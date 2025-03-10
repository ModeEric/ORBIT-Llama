"""
ORBIT Evaluation Tools
======================

This package provides tools for evaluating language models
on domain-specific benchmarks.

Main Components:
---------------
- OrbitEvaluator: For evaluating model performance on domain benchmarks
- DomainBenchmarks: Collection of domain-specific evaluation benchmarks
- CustomBenchmark: For creating custom domain-specific benchmarks
"""

from orbit.evaluation.benchmarks import OrbitEvaluator
from orbit.evaluation.custom_benchmark import CustomBenchmark

__all__ = [
    "OrbitEvaluator",
    "CustomBenchmark"
] 