"""
ORBIT Evaluation Benchmarks

This module provides tools for evaluating language models
on domain-specific benchmarks.
"""

import os
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

class OrbitEvaluator:
    """
    Evaluator for assessing language model performance on domain-specific benchmarks.
    
    This class provides methods for evaluating models on standard benchmarks
    like MMLU domain subsets, as well as custom domain-specific benchmarks.
    """
    
    # Domain to benchmark mapping
    DOMAIN_BENCHMARKS = {
        "astronomy": ["mmlu_astronomy", "mmlu_college_physics", "mmlu_conceptual_physics", 
                     "astrobench_knowledge", "astrobench_calculation"],
        "law": ["mmlu_professional_law", "mmlu_international_law", "mmlu_jurisprudence", 
               "lawbench_reasoning", "lawbench_knowledge"],
        "medical": ["mmlu_medical_genetics", "mmlu_anatomy", "mmlu_clinical_knowledge", 
                   "medicalbench_diagnosis", "medicalbench_treatment"],
        # Add more domains and their benchmarks here
    }
    
    def __init__(self, domain: Optional[str] = None):
        """
        Initialize the OrbitEvaluator.
        
        Args:
            domain: Domain to evaluate for (e.g., "astronomy", "law", "medical")
        """
        self.domain = domain
        self.logger = logging.getLogger("orbit.evaluation")
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def evaluate(self, model_path: str, benchmark: Optional[str] = None, 
                output_dir: Optional[str] = None, device: str = "cuda:0",
                batch_size: int = 8, num_fewshot: int = 0) -> Dict[str, Any]:
        """
        Evaluate a model on a domain-specific benchmark.
        
        Args:
            model_path: Path to the model
            benchmark: Name of the benchmark to use (if None, uses domain benchmarks)
            output_dir: Directory to save evaluation results
            device: Device to run evaluation on (e.g., "cuda:0", "cpu")
            batch_size: Batch size for evaluation
            num_fewshot: Number of few-shot examples to use
            
        Returns:
            Evaluation results
        """
        if output_dir is None:
            output_dir = f"orbit_evaluations/{os.path.basename(model_path)}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine which benchmarks to run
        benchmarks_to_run = []
        if benchmark is not None:
            # Run a specific benchmark
            benchmarks_to_run = [benchmark]
        elif self.domain is not None and self.domain in self.DOMAIN_BENCHMARKS:
            # Run all benchmarks for the specified domain
            benchmarks_to_run = self.DOMAIN_BENCHMARKS[self.domain]
        else:
            self.logger.warning(f"No benchmarks specified for domain: {self.domain}")
            benchmarks_to_run = ["mmlu_default"]
        
        self.logger.info(f"Evaluating model {model_path} on benchmarks: {benchmarks_to_run}")
        
        # Run evaluations
        results = {
            "model": os.path.basename(model_path),
            "domain": self.domain,
            "benchmarks": {}
        }
        
        for bench in benchmarks_to_run:
            try:
                bench_result = self._run_benchmark(
                    model_path=model_path,
                    benchmark=bench,
                    output_dir=output_dir,
                    device=device,
                    batch_size=batch_size,
                    num_fewshot=num_fewshot
                )
                results["benchmarks"][bench] = bench_result
            except Exception as e:
                self.logger.error(f"Error running benchmark {bench}: {str(e)}")
                results["benchmarks"][bench] = {"error": str(e)}
        
        # Calculate aggregate scores
        if results["benchmarks"]:
            scores = [b.get("score", 0) for b in results["benchmarks"].values() 
                     if isinstance(b, dict) and "score" in b]
            if scores:
                results["average_score"] = sum(scores) / len(scores)
        
        # Save results
        results_path = os.path.join(output_dir, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation complete. Results saved to {results_path}")
        return results
    
    def _run_benchmark(self, model_path: str, benchmark: str, output_dir: str,
                      device: str, batch_size: int, num_fewshot: int) -> Dict[str, Any]:
        """
        Run a specific benchmark.
        
        Args:
            model_path: Path to the model
            benchmark: Name of the benchmark
            output_dir: Directory to save results
            device: Device to run on
            batch_size: Batch size
            num_fewshot: Number of few-shot examples
            
        Returns:
            Benchmark results
        """
        # Handle different benchmark types
        if benchmark.startswith("mmlu_"):
            return self._run_mmlu_benchmark(
                model_path=model_path,
                mmlu_subset=benchmark.replace("mmlu_", ""),
                output_dir=output_dir,
                device=device,
                batch_size=batch_size,
                num_fewshot=num_fewshot
            )
        elif benchmark.startswith("astrobench_"):
            return self._run_astrobench(
                model_path=model_path,
                astrobench_subset=benchmark.replace("astrobench_", ""),
                output_dir=output_dir,
                device=device
            )
        else:
            # Generic benchmark runner
            self.logger.warning(f"Using generic benchmark runner for {benchmark}")
            return {
                "benchmark": benchmark,
                "score": 0.5,  # Placeholder
                "note": "Generic benchmark runner used"
            }
    
    def _run_mmlu_benchmark(self, model_path: str, mmlu_subset: str, output_dir: str,
                           device: str, batch_size: int, num_fewshot: int) -> Dict[str, Any]:
        """
        Run MMLU benchmark.
        
        Args:
            model_path: Path to the model
            mmlu_subset: MMLU subset to evaluate on
            output_dir: Directory to save results
            device: Device to run on
            batch_size: Batch size
            num_fewshot: Number of few-shot examples
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Running MMLU benchmark for subset: {mmlu_subset}")
        
        # Construct command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model_path}",
            "--tasks", mmlu_subset if mmlu_subset != "default" else "mmlu",
            "--device", device,
            "--batch_size", str(batch_size),
            "--num_fewshot", str(num_fewshot),
            "--output_path", os.path.join(output_dir, f"mmlu_{mmlu_subset}_results.json")
        ]
        
        # Run command
        try:
            self.logger.info(f"Running command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Parse results
            results_path = os.path.join(output_dir, f"mmlu_{mmlu_subset}_results.json")
            if os.path.exists(results_path):
                with open(results_path, "r") as f:
                    mmlu_results = json.load(f)
                
                # Extract score
                if "results" in mmlu_results and mmlu_subset in mmlu_results["results"]:
                    score = mmlu_results["results"][mmlu_subset]["acc"]
                else:
                    score = mmlu_results.get("acc", 0)
                
                return {
                    "benchmark": f"mmlu_{mmlu_subset}",
                    "score": score,
                    "details": mmlu_results
                }
            else:
                return {
                    "benchmark": f"mmlu_{mmlu_subset}",
                    "error": "Results file not found",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error running MMLU benchmark: {e}")
            return {
                "benchmark": f"mmlu_{mmlu_subset}",
                "error": str(e),
                "stdout": e.stdout,
                "stderr": e.stderr
            }
    
    def _run_astrobench(self, model_path: str, astrobench_subset: str, 
                       output_dir: str, device: str) -> Dict[str, Any]:
        """
        Run AstroBench benchmark.
        
        Args:
            model_path: Path to the model
            astrobench_subset: AstroBench subset to evaluate on
            output_dir: Directory to save results
            device: Device to run on
            
        Returns:
            Benchmark results
        """
        self.logger.info(f"Running AstroBench for subset: {astrobench_subset}")
        
        # Import the astrobench test module
        try:
            from orbit.evaluation.astrobench_tests import evaluate_model_on_dataset
            
            # Map subset to dataset
            dataset_mapping = {
                "knowledge": "AstroMLab/Astrobench_MCQ_v1_Public",
                "calculation": "astroBench/scientific-calculation-test",
                "application": "astroBench/knowledge_application",
                "basic": "astroBench/basic-knowledge-test"
            }
            
            dataset = dataset_mapping.get(astrobench_subset, dataset_mapping["knowledge"])
            
            # Run evaluation
            results_file = evaluate_model_on_dataset(
                model_path=model_path,
                dataset_name=dataset,
                device=device,
                output_dir=output_dir
            )
            
            # Load results
            if os.path.exists(results_file):
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                return {
                    "benchmark": f"astrobench_{astrobench_subset}",
                    "score": results.get("accuracy", 0),
                    "details": results
                }
            else:
                return {
                    "benchmark": f"astrobench_{astrobench_subset}",
                    "error": "Results file not found"
                }
        except ImportError:
            self.logger.error("AstroBench tests module not found")
            return {
                "benchmark": f"astrobench_{astrobench_subset}",
                "error": "AstroBench tests module not found"
            }
        except Exception as e:
            self.logger.error(f"Error running AstroBench: {str(e)}")
            return {
                "benchmark": f"astrobench_{astrobench_subset}",
                "error": str(e)
            }
    
    def list_available_benchmarks(self) -> Dict[str, List[str]]:
        """
        List all available benchmarks by domain.
        
        Returns:
            Dictionary mapping domains to available benchmarks
        """
        return self.DOMAIN_BENCHMARKS 