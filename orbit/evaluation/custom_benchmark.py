"""
Custom Domain Benchmark Creator

This module provides tools for creating and running custom domain-specific benchmarks.
"""

import os
import json
import random
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import torch
from tqdm import tqdm

class CustomBenchmark:
    """
    Create and run custom domain-specific benchmarks.
    
    This class allows users to create benchmarks for their custom domains
    by providing domain-specific questions and evaluating models on them.
    """
    
    def __init__(self, domain_name: str):
        """
        Initialize the CustomBenchmark.
        
        Args:
            domain_name: Name of the custom domain
        """
        self.domain_name = domain_name
        self.logger = logging.getLogger(f"orbit.evaluation.{domain_name}")
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def create_benchmark(self, questions: List[Dict[str, Any]], 
                        output_path: Optional[str] = None) -> str:
        """
        Create a custom benchmark from a list of questions.
        
        Args:
            questions: List of question dictionaries, each containing:
                - question: The question text
                - options: List of possible answers
                - answer: Index of the correct answer (0-based)
            output_path: Path to save the benchmark
            
        Returns:
            Path to the created benchmark file
        """
        if output_path is None:
            output_dir = f"orbit_benchmarks/{self.domain_name}"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{self.domain_name}_benchmark.json")
        
        # Validate questions
        validated_questions = []
        for i, q in enumerate(questions):
            if "question" not in q or "options" not in q or "answer" not in q:
                self.logger.warning(f"Question {i} is missing required fields, skipping")
                continue
            
            if not isinstance(q["options"], list) or len(q["options"]) < 2:
                self.logger.warning(f"Question {i} has invalid options, skipping")
                continue
            
            if not isinstance(q["answer"], int) or q["answer"] < 0 or q["answer"] >= len(q["options"]):
                self.logger.warning(f"Question {i} has invalid answer index, skipping")
                continue
            
            validated_questions.append(q)
        
        if not validated_questions:
            raise ValueError("No valid questions provided")
        
        # Create benchmark
        benchmark = {
            "domain": self.domain_name,
            "name": f"{self.domain_name}_benchmark",
            "description": f"Custom benchmark for {self.domain_name} domain",
            "questions": validated_questions,
            "version": "1.0",
            "num_questions": len(validated_questions)
        }
        
        # Save benchmark
        with open(output_path, "w") as f:
            json.dump(benchmark, f, indent=2)
        
        self.logger.info(f"Created benchmark with {len(validated_questions)} questions at {output_path}")
        return output_path
    
    def evaluate_model(self, model_path: str, benchmark_path: str, 
                      output_dir: Optional[str] = None, device: str = "cuda:0",
                      batch_size: int = 4) -> Dict[str, Any]:
        """
        Evaluate a model on a custom benchmark.
        
        Args:
            model_path: Path to the model
            benchmark_path: Path to the benchmark file
            output_dir: Directory to save results
            device: Device to run on
            batch_size: Batch size for evaluation
            
        Returns:
            Evaluation results
        """
        if output_dir is None:
            output_dir = f"orbit_evaluations/{os.path.basename(model_path)}/{self.domain_name}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Load benchmark
        with open(benchmark_path, "r") as f:
            benchmark = json.load(f)
        
        questions = benchmark.get("questions", [])
        if not questions:
            raise ValueError("Benchmark contains no questions")
        
        self.logger.info(f"Evaluating model {model_path} on {len(questions)} questions")
        
        # Load model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
        # Run evaluation
        results = {
            "model": os.path.basename(model_path),
            "benchmark": os.path.basename(benchmark_path),
            "domain": self.domain_name,
            "num_questions": len(questions),
            "correct": 0,
            "questions": []
        }
        
        for i, question in enumerate(tqdm(questions, desc=f"Evaluating {self.domain_name} benchmark")):
            try:
                # Create prompt
                prompt = self._create_prompt(question)
                
                # Get model prediction
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model.generate(
                        inputs["input_ids"],
                        max_new_tokens=50,
                        temperature=0.1,
                        do_sample=False
                    )
                
                # Decode output
                output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                prediction = self._extract_answer(output_text, prompt)
                
                # Check if correct
                correct_answer = question["options"][question["answer"]]
                is_correct = prediction.lower() == correct_answer.lower()
                
                if is_correct:
                    results["correct"] += 1
                
                # Save result
                results["questions"].append({
                    "question": question["question"],
                    "options": question["options"],
                    "correct_answer": correct_answer,
                    "prediction": prediction,
                    "is_correct": is_correct,
                    "model_output": output_text
                })
            except Exception as e:
                self.logger.error(f"Error evaluating question {i}: {str(e)}")
                results["questions"].append({
                    "question": question["question"],
                    "error": str(e)
                })
        
        # Calculate accuracy
        results["accuracy"] = results["correct"] / len(questions) if questions else 0
        
        # Save results
        results_path = os.path.join(output_dir, f"{self.domain_name}_benchmark_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Evaluation complete. Accuracy: {results['accuracy']:.2f}")
        self.logger.info(f"Results saved to {results_path}")
        
        return results
    
    def _create_prompt(self, question: Dict[str, Any]) -> str:
        """
        Create a prompt for a question.
        
        Args:
            question: Question dictionary
            
        Returns:
            Formatted prompt
        """
        prompt = f"Question: {question['question']}\n\nOptions:\n"
        
        for i, option in enumerate(question["options"]):
            prompt += f"{chr(65+i)}. {option}\n"
        
        prompt += "\nPlease select the correct answer (A, B, C, or D)."
        return prompt
    
    def _extract_answer(self, output_text: str, prompt: str) -> str:
        """
        Extract the answer from the model output.
        
        Args:
            output_text: Model output text
            prompt: Original prompt
            
        Returns:
            Extracted answer
        """
        # Remove the prompt from the output
        response = output_text[len(prompt):].strip()
        
        # Look for answer patterns
        answer_patterns = [
            r"(?:Answer|The answer is|I choose|The correct answer is)[:\s]*([A-D])",
            r"^([A-D])\.?$",
            r"^([A-D])[.\s]"
        ]
        
        for pattern in answer_patterns:
            import re
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer_letter = match.group(1).upper()
                return answer_letter
        
        # If no clear answer found, look for the first occurrence of A, B, C, or D
        for char in response:
            if char.upper() in "ABCD":
                return char.upper()
        
        # Default to a random answer if nothing found
        return random.choice(["A", "B", "C", "D"])
    
    @staticmethod
    def create_question(question_text: str, options: List[str], 
                       correct_answer_idx: int) -> Dict[str, Any]:
        """
        Create a question dictionary.
        
        Args:
            question_text: The question text
            options: List of possible answers
            correct_answer_idx: Index of the correct answer (0-based)
            
        Returns:
            Question dictionary
        """
        return {
            "question": question_text,
            "options": options,
            "answer": correct_answer_idx
        } 