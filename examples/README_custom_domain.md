# ORBIT Custom Domain Processor

This tool allows you to define your own custom domains for the ORBIT dataset curation pipeline by simply providing a list of domain-specific keywords.

## Quick Start

1. Create a text file with your domain keywords (one per line):

```
# finance_keywords.txt
stock
bond
investment
portfolio
dividend
equity
...
```

2. Run the custom domain processor:

```bash
python orbit_custom_domain.py --domain finance --keywords finance_keywords.txt --input your_dataset.jsonl
```

3. The processed dataset will be saved to `finance_processed/final_finance_dataset.jsonl`

## Command Line Options

```
usage: orbit_custom_domain.py [-h] --domain DOMAIN --keywords KEYWORDS --input INPUT
                             [--output-dir OUTPUT_DIR] [--embedding EMBEDDING]
                             [--mapping MAPPING] [--quality-model QUALITY_MODEL]
                             [--evaluate-quality] [--quality-threshold QUALITY_THRESHOLD]
                             [--model-type {simple,transformer}] [--workers WORKERS]
                             [--verbose]

ORBIT Custom Domain Processor

optional arguments:
  -h, --help            show this help message and exit
  --domain DOMAIN, -d DOMAIN
                        Name of the custom domain
  --keywords KEYWORDS, -k KEYWORDS
                        Path to keywords file (one keyword per line)
  --input INPUT, -i INPUT
                        Input file to process
  --output-dir OUTPUT_DIR, -o OUTPUT_DIR
                        Output directory for processed data
  --embedding EMBEDDING, -e EMBEDDING
                        Path to FastText embedding model
  --mapping MAPPING, -m MAPPING
                        Path to token similarity mapping
  --quality-model QUALITY_MODEL, -q QUALITY_MODEL
                        Path to quality evaluation model
  --evaluate-quality, -eq
                        Evaluate document quality
  --quality-threshold QUALITY_THRESHOLD, -qt QUALITY_THRESHOLD
                        Quality threshold (0-1)
  --model-type {simple,transformer}, -mt {simple,transformer}
                        Type of quality model
  --workers WORKERS, -w WORKERS
                        Number of worker processes
  --verbose, -v         Enable verbose output
```

## Using the API

You can also use the `CustomDomainProcessor` directly in your Python code:

```python
from orbit.datasets.custom import CustomDomainProcessor

# Define your domain
domain_name = "finance"
keywords = ["stock", "bond", "investment", "portfolio", ...]

# Initialize the processor
processor = CustomDomainProcessor(
    domain_name=domain_name,
    keywords=keywords
)

# Process a dataset
stats = processor.process_dataset(
    input_path="your_dataset.jsonl",
    output_dir="finance_processed"
)

# Analyze a specific text
results = processor.analyze_domain_content(
    "The stock market experienced significant volatility today..."
)
```

## Tips for Effective Domain Definition

1. Include at least 20-30 keywords that are highly specific to your domain
2. Include both general terms and specialized vocabulary
3. Consider including multi-word phrases for more precise matching
4. Test your keyword list on sample texts to ensure it captures the right content
5. Refine your keyword list based on the results of initial processing

## Example Domains

The `examples` directory contains sample implementations for several domains:
- Finance
- Sports
- Technology
- Cooking
- Travel

You can use these as templates for defining your own custom domains. 