from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import re
import logging
from tqdm import tqdm
from collections import defaultdict
import os

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
        prompt (str): The prompt string up to "ANSWER": "
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
        model (transformers.PreTrainedModel): The loaded language model.
        device (torch.device): Computation device.
        options (list): List of option letters, e.g., ['A', 'B', 'C', 'D'].
    
    Returns:
        dict: Mapping from option to its probability.
    """
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)
    
    # Get logits for the next token after the prompt
    next_token_logits = logits[0, -1, :]  # Shape: (vocab_size,)
    probabilities = torch.softmax(next_token_logits, dim=-1)
    
    option_probs = {}
    for option in options:
        token_ids = tokenizer.encode(option, add_special_tokens=False)
        if not token_ids:
            continue
        # Assuming each option is a single token
        token_id = token_ids[-1]
        option_probs[option] = probabilities[token_id].item()
    
    return option_probs

def evaluate_model_on_dataset(model_path, dataset_name, device, tokenizer, model, output_dir="results"):
    """
    Evaluates a single model on a single dataset.
    Generates answers using zero-shot prompt-based method with perplexity measurements and selects the answer with the highest probability.
    Outputs results in JSONL format.
    
    Parameters:
        model_path (str): Identifier or path of the model.
        dataset_name (str): Name of the dataset.
        device (torch.device): Computation device (cuda or cpu).
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer associated with the model.
        model (transformers.PreTrainedModel): The loaded language model.
        output_dir (str): Directory to store output JSONL files.
    """
    # Load the dataset
    try:
        dataset = load_dataset(dataset_name)
    except Exception as e:
        logging.error(f"Failed to load dataset {dataset_name}: {str(e)}")
        return
    
    # Determine the split to use (prefer 'train', then 'validation', then 'test')
    if 'train' in dataset:
        split = 'train'
    elif 'validation' in dataset:
        split = 'validation'
    elif 'test' in dataset:
        split = 'test'
    else:
        split = list(dataset.keys())[0]  # Use the first available split
    
    dataset_split = dataset[split]
    
    # Prepare output file
    model_safe_name = model_path.replace('/', '_')
    dataset_safe_name = dataset_name.replace('/', '_')
    output_file = os.path.join(output_dir, f"{model_safe_name}_{dataset_safe_name}.jsonl")
    
    # Define answer options
    answer_options = ['A', 'B', 'C', 'D']
    
    # Open the output file
    with open(output_file, "w", encoding='utf-8') as f:
        # Iterate over the test examples
        for example in tqdm(dataset_split, desc=f"Processing {dataset_name} with {model_path}"):
            question = example['question']
            options = [example['A'], example['B'], example['C'], example['D']]
            
            test_example = {
                'question': question,
                'options': options
            }
            
            # Create the prompt up to "ANSWER": "
            prompt = create_prompt(test_example)
            
            # Calculate probabilities for each answer option
            option_probs = calculate_option_probabilities(
                prompt, tokenizer, model, device, answer_options
            )
            
            if not option_probs:
                logging.warning(f"No probabilities calculated for model {model_path} on dataset {dataset_name}.")
                answer = None
            else:
                # Select the option with the highest probability
                selected_option = max(option_probs, key=option_probs.get)
                
                # Validate the selected option
                if selected_option not in ['A', 'B', 'C', 'D']:
                    logging.warning(f"Invalid selected option '{selected_option}' for model {model_path} on dataset {dataset_name}.")
                    answer = None
                else:
                    answer = selected_option
            
            # Prepare the JSON object with ANSWER
            answer_output = {
                "ANSWER": answer
            }
            
            # Write the JSON object to the file
            f.write(json.dumps(answer_output, ensure_ascii=False) + "\n")

def main():
    """
    Main function to orchestrate the evaluation of multiple models across multiple datasets.
    Generates predictions and saves them to separate JSONL files.
    Logs any discrepancies or errors encountered during evaluation.
    """
    # Configure logging
    logging.basicConfig(
        filename='evaluation_errors.log',
        level=logging.WARNING,  # Capture WARNING and ERROR messages
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define your models and datasets here
    models = [
        "ourmodel",
        "AstroMLab/astrollama-3-8b-base_aic",
        "meta-llama/Meta-Llama-3-8B",
    ]
    
    datasets = [
        "AstroMLab/Astrobench_MCQ_v1_Public",
        "astroBench/knowledge_application",
        "astroBench/scientific-calculation-test",
        "astroBench/basic-knowledge-test"
    ]
    
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    results = defaultdict(dict)
    
    print("Starting evaluation across all models and datasets...")
    
    for model_path in models:
        print(f"\nLoading model: {model_path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(model_path)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_path}: {str(e)}")
            logging.error(f"Error loading model {model_path}: {str(e)}")
            for dataset_name in datasets:
                results[model_path][dataset_name] = None
            continue
        
        for dataset_name in datasets:
            print(f"Evaluating model {model_path} on dataset {dataset_name}...")
            try:
                evaluate_model_on_dataset(
                    model_path, dataset_name, device, tokenizer, model, output_dir=output_dir
                )
                # Since there's no accuracy, we just note that evaluation was successful
                results[model_path][dataset_name] = "Completed"
                print(f"Evaluation completed for {model_path} on {dataset_name}.")
            except Exception as e:
                print(f"Error evaluating {model_path} on {dataset_name}: {str(e)}")
                logging.error(f"Error evaluating {model_path} on {dataset_name}: {str(e)}")
                results[model_path][dataset_name] = None
    
    # Print final results table
    print("\n=== Final Results ===")
    print("\nEvaluation Status")
    print("-" * 100)
    header = f"{'Model':<40} | {'Dataset':<35} | {'Status':<15}"
    print(header)
    print("-" * 100)
    
    for model in models:
        model_name = model.split('/')[-1]
        for dataset in datasets:
            dataset_name_short = dataset.split('/')[-1]
            status = results[model].get(dataset, 'N/A')
            print(f"{model_name:<40} | {dataset_name_short:<35} | {status:<15}")
    
    print("-" * 100)
    print("\nAll predictions have been saved in the 'results' directory as JSONL files.")
    print("Check 'evaluation_errors.log' for any warnings or errors encountered during evaluation.")

if __name__ == "__main__":
    main()
