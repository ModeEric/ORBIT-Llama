#!/usr/bin/env python
"""
Create custom domain benchmarks for ORBIT.

This script provides a command-line interface for creating custom
domain-specific benchmarks.
"""

import os
import sys
import argparse
import json
import csv
from typing import List, Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from orbit.evaluation.custom_benchmark import CustomBenchmark

def load_questions_from_csv(csv_path: str) -> List[Dict[str, Any]]:
    """
    Load questions from a CSV file.
    
    The CSV should have the following columns:
    - question: The question text
    - option_a, option_b, option_c, option_d: The options
    - answer: The correct answer (A, B, C, or D)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        List of question dictionaries
    """
    questions = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for i, row in enumerate(reader):
            try:
                # Extract options
                options = []
                for opt in ['option_a', 'option_b', 'option_c', 'option_d']:
                    if opt in row and row[opt].strip():
                        options.append(row[opt].strip())
                
                # Convert answer letter to index
                answer_letter = row.get('answer', '').strip().upper()
                if answer_letter in 'ABCD':
                    answer_idx = ord(answer_letter) - ord('A')
                else:
                    print(f"Warning: Invalid answer '{answer_letter}' in row {i+2}, skipping")
                    continue
                
                # Create question
                question = {
                    "question": row.get('question', '').strip(),
                    "options": options,
                    "answer": answer_idx
                }
                
                # Validate
                if not question["question"]:
                    print(f"Warning: Empty question in row {i+2}, skipping")
                    continue
                
                if len(options) < 2:
                    print(f"Warning: Not enough options in row {i+2}, skipping")
                    continue
                
                if answer_idx >= len(options):
                    print(f"Warning: Answer index out of range in row {i+2}, skipping")
                    continue
                
                questions.append(question)
                
            except Exception as e:
                print(f"Error processing row {i+2}: {e}")
                continue
    
    return questions

def load_questions_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load questions from a JSONL file.
    
    Each line should be a JSON object with:
    - question: The question text
    - options: List of possible answers
    - answer: Index of the correct answer (0-based)
    
    Args:
        jsonl_path: Path to JSONL file
        
    Returns:
        List of question dictionaries
    """
    questions = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                
                # Validate
                if "question" not in data or "options" not in data or "answer" not in data:
                    print(f"Warning: Line {i+1} is missing required fields, skipping")
                    continue
                
                if not isinstance(data["options"], list) or len(data["options"]) < 2:
                    print(f"Warning: Line {i+1} has invalid options, skipping")
                    continue
                
                if not isinstance(data["answer"], int) or data["answer"] < 0 or data["answer"] >= len(data["options"]):
                    print(f"Warning: Line {i+1} has invalid answer index, skipping")
                    continue
                
                questions.append(data)
                
            except json.JSONDecodeError:
                print(f"Warning: Line {i+1} is not valid JSON, skipping")
                continue
            except Exception as e:
                print(f"Error processing line {i+1}: {e}")
                continue
    
    return questions

def main():
    """Create a custom domain benchmark."""
    parser = argparse.ArgumentParser(description="Create a custom domain benchmark")
    
    # Domain arguments
    parser.add_argument("--domain", "-d", required=True, help="Name of the custom domain")
    
    # Input arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--csv", help="Path to CSV file with questions")
    input_group.add_argument("--jsonl", help="Path to JSONL file with questions")
    input_group.add_argument("--interactive", "-i", action="store_true", help="Create questions interactively")
    
    # Output arguments
    parser.add_argument("--output", "-o", help="Path to save the benchmark")
    
    args = parser.parse_args()
    
    # Create benchmark creator
    benchmark_creator = CustomBenchmark(domain_name=args.domain)
    
    # Load questions
    questions = []
    
    if args.csv:
        print(f"Loading questions from CSV: {args.csv}")
        questions = load_questions_from_csv(args.csv)
    elif args.jsonl:
        print(f"Loading questions from JSONL: {args.jsonl}")
        questions = load_questions_from_jsonl(args.jsonl)
    elif args.interactive:
        print("Creating questions interactively")
        print("Enter 'done' when finished")
        
        while True:
            print("\nNew Question:")
            question_text = input("Question: ")
            if question_text.lower() == 'done':
                break
            
            options = []
            for i in range(4):
                option = input(f"Option {chr(65+i)}: ")
                options.append(option)
            
            answer_letter = input("Correct answer (A, B, C, or D): ").strip().upper()
            if answer_letter in 'ABCD':
                answer_idx = ord(answer_letter) - ord('A')
            else:
                print(f"Invalid answer '{answer_letter}', skipping question")
                continue
            
            questions.append({
                "question": question_text,
                "options": options,
                "answer": answer_idx
            })
    
    if not questions:
        print("Error: No valid questions found")
        return
    
    print(f"Loaded {len(questions)} questions")
    
    # Create benchmark
    output_path = args.output
    benchmark_path = benchmark_creator.create_benchmark(questions, output_path)
    
    print(f"Benchmark created at: {benchmark_path}")
    print(f"To evaluate a model on this benchmark, run:")
    print(f"python orbit/evaluation/run_evaluation.py --model <model_path> --custom-domain {args.domain} --custom-benchmark {benchmark_path}")

if __name__ == "__main__":
    main() 