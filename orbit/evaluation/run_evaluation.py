#!/usr/bin/env python
"""
Run evaluations on ORBIT models.

This script provides a command-line interface for evaluating models
on domain-specific benchmarks.
"""

import os
import sys
import argparse
import json
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from orbit.evaluation.benchmarks import OrbitEvaluator
from orbit.evaluation.custom_benchmark import CustomBenchmark

def main():
    """Run evaluations on ORBIT models."""
    parser = argparse.ArgumentParser(description="Evaluate ORBIT models")
    
    # Model arguments
    parser.add_argument("--model", "-m", required=True, help="Path to model")
    
    # Domain arguments
    parser.add_argument("--domain", "-d", help="Domain to evaluate (astronomy, law, medical, etc.)")
    parser.add_argument("--custom-domain", "-cd", help="Name of custom domain")
    
    # Benchmark arguments
    parser.add_argument("--benchmark", "-b", help="Specific benchmark to run")
    parser.add_argument("--custom-benchmark", "-cb", help="Path to custom benchmark file")
    
    # Evaluation settings
    parser.add_argument("--output-dir", "-o", help="Directory to save results")
    parser.add_argument("--device", default="cuda:0", help="Device to run on (cuda:0, cpu, etc.)")
    parser.add_argument("--batch-size", "-bs", type=int, default=8, help="Batch size")
    parser.add_argument("--num-fewshot", "-fs", type=int, default=0, help="Number of few-shot examples")
    
    # MMLU specific
    parser.add_argument("--mmlu-subset", help="MMLU subset to evaluate on")
    
    # List available benchmarks
    parser.add_argument("--list-benchmarks", "-l", action="store_true", help="List available benchmarks")
    
    args = parser.parse_args()
    
    # List available benchmarks if requested
    if args.list_benchmarks:
        evaluator = OrbitEvaluator()
        benchmarks = evaluator.list_available_benchmarks()
        print("Available benchmarks by domain:")
        for domain, domain_benchmarks in benchmarks.items():
            print(f"\n{domain.capitalize()}:")
            for bench in domain_benchmarks:
                print(f"  - {bench}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model path does not exist: {args.model}")
        return
    
    # Handle custom domain evaluation
    if args.custom_domain:
        if not args.custom_benchmark:
            print("Error: Custom domain evaluation requires a custom benchmark file")
            return
        
        if not os.path.exists(args.custom_benchmark):
            print(f"Error: Custom benchmark file does not exist: {args.custom_benchmark}")
            return
        
        print(f"Evaluating model on custom {args.custom_domain} benchmark")
        custom_evaluator = CustomBenchmark(domain_name=args.custom_domain)
        results = custom_evaluator.evaluate_model(
            model_path=args.model,
            benchmark_path=args.custom_benchmark,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size
        )
        
        print(f"\nResults for {args.custom_domain} benchmark:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Correct: {results['correct']}/{results['num_questions']}")
        
        if args.output_dir:
            print(f"Detailed results saved to: {args.output_dir}")
    
    # Handle standard domain evaluation
    elif args.domain or args.benchmark or args.mmlu_subset:
        # Set up evaluator
        domain = args.domain
        benchmark = args.benchmark
        
        # Handle MMLU subset
        if args.mmlu_subset:
            benchmark = f"mmlu_{args.mmlu_subset}"
        
        print(f"Evaluating model on {'domain: ' + domain if domain else 'benchmark: ' + benchmark}")
        
        evaluator = OrbitEvaluator(domain=domain)
        results = evaluator.evaluate(
            model_path=args.model,
            benchmark=benchmark,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            num_fewshot=args.num_fewshot
        )
        
        print("\nEvaluation Results:")
        if "average_score" in results:
            print(f"Average Score: {results['average_score']:.4f}")
        
        print("\nBenchmark Scores:")
        for bench_name, bench_result in results.get("benchmarks", {}).items():
            if isinstance(bench_result, dict) and "score" in bench_result:
                print(f"  {bench_name}: {bench_result['score']:.4f}")
            elif isinstance(bench_result, dict) and "error" in bench_result:
                print(f"  {bench_name}: Error - {bench_result['error']}")
        
        if args.output_dir:
            print(f"Detailed results saved to: {args.output_dir}")
    
    else:
        print("Error: Please specify either a domain, benchmark, or custom domain to evaluate")
        parser.print_help()

if __name__ == "__main__":
    main() 