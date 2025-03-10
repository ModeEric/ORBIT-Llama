"""
AstroBench Evaluation Tests

This module provides tools for evaluating language models on astronomy-specific benchmarks.
It's based on the original astrobench_tests.py but integrated into the ORBIT evaluation framework.
"""

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import logging
from tqdm import tqdm
from collections import defaultdict
import os

logger = logging.getLogger("orbit.evaluation.astrobench")

def create_prompt(test_example):
    """
    Creates a zero-shot prompt following the expert format.
    
    Parameters:
        test_example (dict): A dictionary containing 'question' and 'options' keys.
    
    Returns:
        str: The formatted prompt string.
    """
    prompt = (
        "You are an expert in general astrophysics. Your task is to answer and explain the following multiple-choice "
        "question on astrophysics, sourced from a dataset. The question is:\n"
    )
    prompt += f"Question: {test_example['question']}\n"
    prompt += "Options:\n"
    for j, opt in enumerate(test_example['options']):
        prompt += f"{chr(65+j)}: {opt}\n"
    prompt += (
        "Determine the correct answer using your astrophysics knowledge and provide a detailed explanation for why "
        "this answer is correct.\n"
        "Ensure your explanation is thorough, clearly articulating your thought process based on astrophysical principles.\n"
        "Output format:\n"
        "{\n"
        '  "ANSWER": "[The choice you decide to choose]",\n'
        '  "EXPLANATION": "[Provide a valid explanation for the answer mentioned in ANSWER]"\n'
        "}\n"
        "Give only one answer, either A, B, C or D, but not more than one, and always give an answer.\n"
        "Adhere to the output format.\n"
    )
    prompt += 'Response:\n{\n"ANSWER": "'
    return prompt

def calculate_option_probabilities(prompt, tokenizer, model, device, options):
    """
    Calculates the probabilities of each option being the next token after the prompt.
    
    Parameters:
        prompt (str): The prompt text.
        tokenizer: The tokenizer for the model.
        model: The language model.
        device (str): The device to run the model on.
        options (list): List of option strings.
    
    Returns:
        dict: A dictionary mapping option letters to their probabilities.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits[0, -1, :]
    probs = torch.nn.functional.softmax(logits, dim=0)
    
    option_probs = {}
    for i, option in enumerate(options):
        option_letter = chr(65 + i)  # A, B, C, D
        token_id = tokenizer.encode(option_letter)[0]
        option_probs[option_letter] = probs[token_id].item()
    
    return option_probs

def extract_answer_from_response(response_text):
    """
    Extracts the answer from the model's response.
    
    Parameters:
        response_text (str): The model's response text.
    
    Returns:
        str: The extracted answer (A, B, C, or D).
    """
    # Try to extract from JSON format
    try:
        # Find the JSON part of the response
        json_match = re.search(r'{\s*"ANSWER":\s*"([A-D])"', response_text)
        if json_match:
            return json_match.group(1)
    except:
        pass
    
    # Try to find any mention of A, B, C, or D as an answer
    answer_match = re.search(r'(?:answer|choice|option)\s+(?:is|:)?\s*([A-D])', response_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Look for standalone A, B, C, or D
    standalone_match = re.search(r'\b([A-D])\b', response_text)
    if standalone_match:
        return standalone_match.group(1).upper()
    
    # Default to the most probable option
    return None

def evaluate_model_on_dataset(model_path, dataset_name, device="cuda:0", output_dir=None):
    """
    Evaluates a model on an astronomy benchmark dataset.
    
    Parameters:
        model_path (str): Path to the model.
        dataset_name (str): Name of the dataset to evaluate on.
        device (str): Device to run the model on.
        output_dir (str): Directory to save results.
    
    Returns:
        str: Path to the results file.
    """
    if output_dir is None:
        output_dir = f"orbit_evaluations/{os.path.basename(model_path)}/astrobench"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up logging
    log_file = os.path.join(output_dir, "evaluation.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    logger.info(f"Evaluating model {model_path} on dataset {dataset_name}")
    
    # Load dataset
    try:
        dataset = load_dataset(dataset_name)
        test_set = dataset["test"] if "test" in dataset else dataset["validation"]
        logger.info(f"Loaded dataset with {len(test_set)} examples")
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise
    
    # Load model and tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        logger.info(f"Loaded model and tokenizer from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise
    
    # Prepare results storage
    results = {
        "model": os.path.basename(model_path),
        "dataset": dataset_name,
        "examples": [],
        "correct": 0,
        "total": 0,
        "accuracy": 0.0
    }
    
    # Evaluate each example
    for i, example in enumerate(tqdm(test_set, desc=f"Evaluating {os.path.basename(model_path)}")):
        try:
            # Create prompt
            prompt = create_prompt(example)
            
            # Get model prediction
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=200,
                    temperature=0.1,
                    do_sample=False
                )
            
            # Decode output
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract answer
            prediction = extract_answer_from_response(output_text)
            
            # If extraction failed, use token probabilities
            if prediction is None:
                option_probs = calculate_option_probabilities(prompt, tokenizer, model, device, example['options'])
                prediction = max(option_probs.items(), key=lambda x: x[1])[0]
            
            # Check if correct
            correct_idx = example.get('answer', None)
            if correct_idx is not None:
                correct_answer = chr(65 + correct_idx)
                is_correct = prediction == correct_answer
                
                if is_correct:
                    results["correct"] += 1
            else:
                is_correct = None
            
            results["total"] += 1
            
            # Save result
            results["examples"].append({
                "question": example["question"],
                "options": example["options"],
                "correct_answer": correct_answer if correct_idx is not None else None,
                "prediction": prediction,
                "is_correct": is_correct,
                "model_output": output_text
            })
            
            # Log progress
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_set)} examples")
        
        except Exception as e:
            logger.error(f"Error processing example {i}: {str(e)}")
            results["examples"].append({
                "question": example.get("question", ""),
                "error": str(e)
            })
    
    # Calculate accuracy
    if results["total"] > 0:
        results["accuracy"] = results["correct"] / results["total"]
    
    # Save results
    results_file = os.path.join(output_dir, f"{os.path.basename(dataset_name)}_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Results saved to {results_file}")
    
    return results_file

def main():
    """Run AstroBench evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model on AstroBench")
    parser.add_argument("--model", "-m", required=True, help="Path to model")
    parser.add_argument("--dataset", "-d", default="AstroMLab/Astrobench_MCQ_v1_Public", 
                       help="Dataset to evaluate on")
    parser.add_argument("--output-dir", "-o", help="Directory to save results")
    parser.add_argument("--device", default="cuda:0", help="Device to run on")
    
    args = parser.parse_args()
    
    results_file = evaluate_model_on_dataset(
        model_path=args.model,
        dataset_name=args.dataset,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Load and print results
    with open(results_file, "r") as f:
        results = json.load(f)
    
    print(f"\nResults for {os.path.basename(args.model)} on {os.path.basename(args.dataset)}:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Correct: {results['correct']}/{results['total']}")
    print(f"Detailed results saved to: {results_file}")

if __name__ == "__main__":
    main() 