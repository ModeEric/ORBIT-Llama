import openai
import os
import sys
import json
import re
import time
from pathlib import Path
from tqdm import tqdm
import itertools

def load_api_key():
    """
    Loads the OpenAI API key from the environment variable.
    Make sure to set your API key in the environment variable 'OPENAI_API_KEY'.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    return api_key

def construct_prompt(text):
    """
    Constructs the prompt with the given legal-related text.
    """
    prompt = f"""Please evaluate the educational value of the following legal-related text from a web document. Use this 6-point scoring system:
0 points: No legal content at all.
1 point: Minimal legal information, or legal information mixed with non-legal content.
2 points: Covers basic legal concepts but lacks depth or comprehensive explanation.
3 points: Clear explanation of concepts with relevant examples, educational for a general audience.
4 points: In-depth knowledge, covers advanced legal principles or recent legal developments, well-structured and engaging.
5 points: Exceptionally high educational value, expert-level insights, connects multiple legal concepts, addresses misconceptions, inspires further learning.
Provide a brief justification (up to 100 words) and conclude with the score in the format "Score: X".
Here's the text to evaluate:
{text}"""
    return prompt

def get_gpt4_score(prompt, api_key, model="gpt-4o", max_retries=5):
    """
    Sends the prompt to the OpenAI gpt-4o model and retrieves the response.
    Implements exponential backoff in case of rate limit errors.
    """
    openai.api_key = api_key
    backoff_time = 1  # Start with 1 second
    for attempt in range(1, max_retries + 1):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an assistant that evaluates educational legal content."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,  # Adjust as needed
                temperature=0.4,  # Lower temperature for more deterministic responses
            )
            return response.choices[0].message.content.strip()
        except openai.error.RateLimitError:
            print(f"Rate limit exceeded. Attempt {attempt} of {max_retries}. Retrying in {backoff_time} seconds...")
        except openai.error.APIError as e:
            print(f"API error: {e}. Attempt {attempt} of {max_retries}. Retrying in {backoff_time} seconds...")
        except Exception as e:
            print(f"Unexpected error: {e}. Attempt {attempt} of {max_retries}. Retrying in {backoff_time} seconds...")
        time.sleep(backoff_time)
        backoff_time *= 2  # Exponential backoff
    raise RuntimeError("Max retries exceeded. Failed to get a response from OpenAI API.")

def extract_score(response_text):
    """
    Extracts the score from the response text.
    Assumes the score is in the format "Score: X".
    """
    match = re.search(r"Score:\s*(\d)", response_text)
    if match:
        return int(match.group(1)), response_text
    else:
        raise ValueError("Score not found in the response.")

def process_single_jsonl_file(file_path, api_key, model="gpt-4o", max_entries=1000):
    """
    Processes a single JSONL file and returns a list of results.
    Each result is a dictionary containing the original text, source file, score, and justification.
    Only processes up to max_entries lines.
    """
    results = []
    with open(file_path, 'r', encoding='utf-8') as infile:
        for line_number, line in enumerate(tqdm(itertools.islice(infile, max_entries), desc=f"Processing {os.path.basename(file_path)}", total=max_entries, unit="lines"), start=1):
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if not text:
                    print(f"Warning: Empty 'text' field in {file_path} at line {line_number}. Skipping.")
                    continue

                prompt = construct_prompt(text)
                response = get_gpt4_score(prompt, api_key, model=model)
                score, justification = extract_score(response)

                result = {
                    "text": text,
                    "source_file": os.path.basename(file_path),
                    "score": score,
                    "justification": justification
                }
                results.append(result)
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in {file_path} at line {line_number}. Skipping.")
            except Exception as e:
                print(f"Error processing {file_path} at line {line_number}: {e}")
    return results

def write_results_to_jsonl(results, output_file):
    """
    Writes the list of result dictionaries to a JSONL file.
    """
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for result in tqdm(results, desc="Writing results", unit="entries"):
            json_line = json.dumps(result, ensure_ascii=False)
            outfile.write(json_line + '\n')
    print(f"Results written to {output_file}")

def main(file_path, output_file="evaluation_results_law.jsonl", model="gpt-4o", max_entries=1000):
    """
    Main function to evaluate the educational value of a single JSONL file containing legal-related text.
    Processes up to max_entries lines from the file.
    """
    # Load API key
    api_key = load_api_key()

    # Verify file path
    file = Path(file_path)
    if not file.is_file():
        print(f"Error: {file_path} is not a valid file.")
        sys.exit(1)

    print(f"Processing file: {file_path}")
    results = process_single_jsonl_file(file_path, api_key, model=model, max_entries=max_entries)

    # Write results to the output JSONL file
    write_results_to_jsonl(results, output_file)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate_single_file_law.py <path_to_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    main(file_path)
