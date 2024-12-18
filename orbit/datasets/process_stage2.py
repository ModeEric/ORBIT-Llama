import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

# Configuration
MODEL_DIR = 'path_to_your_finetuned_model'  # Replace with your model directory
INPUT_JSONL = 'input_file.jsonl'           # Replace with your input JSONL file path
OUTPUT_JSONL = 'filtered_output.jsonl'     # Desired output JSONL file path
TEXT_FIELD = 'text'                        # The field in JSON containing text
THRESHOLD = 3.0                            # Threshold score

def load_model_and_tokenizer(model_dir):
    """
    Load the fine-tuned BERT model and tokenizer.
    """
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    model.eval()
    return tokenizer, model

def get_regression_score(tokenizer, model, text, device):
    """
    Get the regression score for a given text using the model.
    """
    inputs = tokenizer(
        text,
        return_tensors='pt',
        truncation=True,
        padding=True,
        max_length=512  # Adjust based on your model's max length
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        # Assuming the model outputs logits for regression
        # If your model uses a different head, adjust accordingly
        score = outputs.logits.squeeze().item()
    return score

def filter_jsonl(input_path, output_path, tokenizer, model, device, text_field, threshold):
    """
    Iterate through the input JSONL file, filter documents based on the model's score,
    and write the filtered documents to the output JSONL file.
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in tqdm(infile, desc="Processing Documents"):
            try:
                doc = json.loads(line)
                text = doc.get(text_field, "")
                if not text:
                    continue  # Skip if text field is empty
                
                score = get_regression_score(tokenizer, model, text, device)
                
                if score >= threshold:
                    outfile.write(json.dumps(doc) + '\n')
            except json.JSONDecodeError:
                print("Skipping invalid JSON line.")
            except Exception as e:
                print(f"Error processing line: {e}")

def main():
    # Check if CUDA is available and set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
    model.to(device)

    # Filter the JSONL file
    filter_jsonl(
        input_path=INPUT_JSONL,
        output_path=OUTPUT_JSONL,
        tokenizer=tokenizer,
        model=model,
        device=device,
        text_field=TEXT_FIELD,
        threshold=THRESHOLD
    )

    print(f"Filtering complete. Filtered documents saved to {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
