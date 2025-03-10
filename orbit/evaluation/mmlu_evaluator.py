#!/usr/bin/env python
"""
MMLU Evaluation for ORBIT

This script provides a wrapper around the lm-evaluation-harness for running
MMLU evaluations on ORBIT models.
"""

import os
import sys
import argparse
import subprocess
import json
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("orbit.evaluation.mmlu")

# MMLU subject to domain mapping
MMLU_DOMAIN_MAPPING = {
    "astronomy": ["astronomy", "college_physics", "conceptual_physics", "high_school_physics"],
    "law": ["professional_law", "international_law", "jurisprudence", "high_school_government_and_politics"],
    "medical": ["medical_genetics", "anatomy", "clinical_knowledge", "college_medicine", "professional_medicine"],
    "computer_science": ["computer_security", "machine_learning", "college_computer_science", "high_school_computer_science"],
    "mathematics": ["abstract_algebra", "high_school_mathematics", "college_mathematics", "elementary_mathematics"],
    "biology": ["college_biology", "high_school_biology", "anatomy"],
    "chemistry": ["college_chemistry", "high_school_chemistry"],
    "physics": ["college_physics", "high_school_physics", "conceptual_physics"],
    "psychology": ["high_school_psychology", "social_psychology", "professional_psychology"],
    "economics": ["high_school_economics", "global_facts", "microeconomics", "macroeconomics"],
    "history": ["high_school_world_history", "high_school_european_history", "high_school_us_history", "world_history"],
    "philosophy": ["philosophy", "moral_scenarios", "logical_fallacies", "formal_logic"],
    "general": ["high_school_government_and_politics", "public_relations", "security_studies", "sociology"]
}

def run_mmlu_evaluation(model_path: str, subjects: List[str], output_dir: Optional[str] = None,
                       device: str = "cuda:0", batch_size: int = 8, num_fewshot: int = 0) -> Dict[str, Any]:
    """
    Run MMLU evaluation on a model.
    
    Args:
        model_path: Path to the model
        subjects: List of MMLU subjects to evaluate on
        output_dir: Directory to save results
        device: Device to run on
        batch_size: Batch size for evaluation
        num_fewshot: Number of few-shot examples
        
    Returns:
        Evaluation results
    """
    if output_dir is None:
        output_dir = f"orbit_evaluations/{os.path.basename(model_path)}/mmlu"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "mmlu_evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    # Format subjects for lm-eval command
    tasks = [f"mmlu_{subject}" for subject in subjects]
    tasks_str = ",".join(tasks)
    
    # Build command
    cmd = [
        "lm_eval",
        "--model", "hf",
        f"--model_args", f"pretrained={model_path}",
        "--tasks", tasks_str,
        "--device", device,
        "--batch_size", str(batch_size),
        "--num_fewshot", str(num_fewshot),
        "--output_path", os.path.join(output_dir, "mmlu_results.json")
    ]
    
    logger.info(f"Running MMLU evaluation on {model_path} for subjects: {subjects}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        # Run evaluation
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse results
        results_file = os.path.join(output_dir, "mmlu_results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                results = json.load(f)
            
            # Extract scores
            scores = {}
            for task in tasks:
                if task in results["results"]:
                    scores[task] = results["results"][task]["acc"]
            
            # Calculate average
            avg_score = sum(scores.values()) / len(scores) if scores else 0
            
            return {
                "model": os.path.basename(model_path),
                "subjects": subjects,
                "scores": scores,
                "average_score": avg_score,
                "details": results
            }
        else:
            logger.error("Results file not found")
            return {
                "model": os.path.basename(model_path),
                "subjects": subjects,
                "error": "Results file not found"
            }
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running MMLU evaluation: {e}")
        return {
            "model": os.path.basename(model_path),
            "subjects": subjects,
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            "model": os.path.basename(model_path),
            "subjects": subjects,
            "error": str(e)
        }

def get_subjects_for_domain(domain: str) -> List[str]:
    """
    Get MMLU subjects for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        List of MMLU subjects
    """
    return MMLU_DOMAIN_MAPPING.get(domain, [])

def main():
    """Run MMLU evaluation from command line."""
    parser = argparse.ArgumentParser(description="Run MMLU evaluation")
    
    parser.add_argument("--model", "-m", required=True, help="Path to model")
    parser.add_argument("--domain", "-d", help="Domain to evaluate (astronomy, law, medical, etc.)")
    parser.add_argument("--subjects", "-s", help="Comma-separated list of MMLU subjects")
    parser.add_argument("--output-dir", "-o", help="Directory to save results")
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    parser.add_argument("--batch-size", "-bs", type=int, default=8, help="Batch size")
    parser.add_argument("--num-fewshot", "-fs", type=int, default=0, help="Number of few-shot examples")
    parser.add_argument("--list-subjects", "-l", action="store_true", help="List available subjects by domain")
    
    args = parser.parse_args()
    
    # List subjects if requested
    if args.list_subjects:
        print("Available MMLU subjects by domain:")
        for domain, subjects in MMLU_DOMAIN_MAPPING.items():
            print(f"\n{domain.capitalize()}:")
            for subject in subjects:
                print(f"  - {subject}")
        return
    
    # Determine subjects to evaluate
    subjects = []
    if args.subjects:
        subjects = args.subjects.split(",")
    elif args.domain:
        subjects = get_subjects_for_domain(args.domain)
        if not subjects:
            print(f"Error: No subjects found for domain '{args.domain}'")
            return
    else:
        print("Error: Either --domain or --subjects must be specified")
        return
    
    # Run evaluation
    results = run_mmlu_evaluation(
        model_path=args.model,
        subjects=subjects,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_fewshot=args.num_fewshot
    )
    
    # Print results
    print(f"\nMMLU Results for {os.path.basename(args.model)}:")
    
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    print(f"Average Score: {results['average_score']:.4f}")
    print("\nSubject Scores:")
    for subject, score in results["scores"].items():
        print(f"  {subject.replace('mmlu_', '')}: {score:.4f}")
    
    if args.output_dir:
        print(f"\nDetailed results saved to: {args.output_dir}/mmlu_results.json")

if __name__ == "__main__":
    main() 