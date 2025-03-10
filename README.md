# ORBIT: Domain-Specific AI for Astronomy, Law, and Medicine

<p align="center">
  <img src="docs/images/orbit-logo.png" alt="ORBIT Logo" width="200"/>
</p>

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Tests](https://github.com/yourusername/orbit/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/orbit/actions)
[![Coverage](https://codecov.io/gh/yourusername/orbit/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/orbit)

ORBIT is an open-source framework that revolutionizes how we create and evaluate domain-specific language models. By combining intelligent dataset curation with advanced fine-tuning techniques, ORBIT enables the development of highly specialized AI models that excel in domains like astronomy, law, and medicine.

## ✨ Why ORBIT?

- 🎯 **Domain Expertise**: Achieves state-of-the-art performance in specialized fields
- 🔍 **Smart Filtering**: Uses advanced embedding techniques to identify high-quality domain content
- 🚀 **Easy to Use**: Simple API and CLI tools for dataset processing and model training
- 📊 **Rigorous Evaluation**: Comprehensive benchmarking suite for each domain
- 🔧 **Extensible**: Support for custom domains beyond the built-in ones

## 🌟 Features

### Domain-Specific Dataset Curation

```python
from orbit.datasets import DatasetCurator

# Create a curator for astronomy data
curator = DatasetCurator(domain="astronomy")

# Process a dataset
processed_data = curator.process_dataset(
    input_path="raw_data.jsonl",
    output_dir="processed_data",
    evaluate_quality=True
)

# Prepare for training
training_data = curator.prepare_training_data(
    input_path="processed_data/final_dataset.jsonl",
    output_dir="training_data",
    split=True
)
```

### Custom Domain Support

```python
from orbit.datasets.custom import CustomDomainProcessor

# Define a custom domain with keywords
processor = CustomDomainProcessor(
    domain_name="finance",
    keywords=["stock", "bond", "investment", "portfolio", "dividend"]
)

# Process a dataset for your custom domain
processor.process_dataset(
    input_path="raw_data.jsonl",
    output_dir="finance_data"
)
```

### Model Training

```python
from orbit.models import OrbitTrainer

# Create a trainer for astronomy
trainer = OrbitTrainer(domain="astronomy")

# Train a model using LoRA
model_path = trainer.train(
    base_model="meta-llama/Llama-2-7b-hf",
    dataset="astronomy_data/train.jsonl",
    method="lora"
)

# Export the model (merge LoRA weights)
exported_model = trainer.export_model(model_path)
```

### Model Evaluation

```python
from orbit.evaluation import OrbitEvaluator

# Create an evaluator for astronomy
evaluator = OrbitEvaluator(domain="astronomy")

# Evaluate on domain-specific benchmarks
results = evaluator.evaluate(
    model_path="orbit_models/astronomy_llama",
    output_dir="evaluation_results"
)

# Print results
print(f"Average Score: {results['average_score']}")
for benchmark, score in results['benchmarks'].items():
    print(f"{benchmark}: {score['score']}")
```

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/orbit.git
cd orbit

# Install the package
pip install -e .

# Install additional dependencies for training
pip install -e ".[train]"
```

### Quick Start

#### 1. Dataset Curation

```bash
# Generate sample data for testing
python orbit/datasets/generate_sample_data.py --samples 1000 --output raw_data.jsonl

# Process the data for astronomy
python test_astro_processor.py --input raw_data.jsonl --evaluate-quality
```

#### 2. Model Training

```bash
# Train a model using LoRA
python orbit/models/train_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset processed_data/astronomy_train.jsonl \
    --domain astronomy \
    --method lora \
    --output-dir orbit_models/astronomy_llama
```

#### 3. Model Evaluation

```bash
# Evaluate on astronomy benchmarks
python orbit/evaluation/run_evaluation.py \
    --model orbit_models/astronomy_llama \
    --domain astronomy
```

## 📚 Documentation

### Dataset Curation Pipeline

ORBIT uses a multi-stage pipeline for curating domain-specific datasets:

1. **Domain Filtering**: Identifies content relevant to the target domain
2. **Quality Assessment**: Evaluates and filters for high-quality content
3. **Deduplication**: Removes duplicate or near-duplicate content
4. **Training Preparation**: Formats data for model training

### Training Methods

ORBIT supports multiple training approaches:

- **Full Fine-tuning**: Complete model parameter update (high resource requirements)
- **LoRA**: Low-Rank Adaptation for efficient fine-tuning (recommended)
- **QLoRA**: Quantized LoRA for even more efficient training on consumer hardware

### Evaluation Framework

The evaluation framework includes:

- **MMLU Domain Subsets**: Subject-specific evaluations from the MMLU benchmark
- **Domain-Specific Benchmarks**: Custom benchmarks for astronomy, law, and medicine
- **Custom Benchmark Creation**: Tools to create benchmarks for your own domains

## 🧩 Custom Domains

ORBIT makes it easy to define your own domains:

1. Create a text file with domain-specific keywords
2. Use the `CustomDomainProcessor` to process your data
3. Train a model for your domain
4. Create and run custom benchmarks

Example:

```bash
# Define finance domain and process data
python orbit_custom_domain.py \
    --domain finance \
    --keywords finance_keywords.txt \
    --input raw_data.jsonl

# Train a model for finance
python orbit/models/train_model.py \
    --model meta-llama/Llama-2-7b-hf \
    --dataset finance_processed/final_finance_dataset.jsonl \
    --domain finance \
    --method lora

# Create a custom benchmark
python orbit/evaluation/create_custom_benchmark.py \
    --domain finance \
    --csv finance_questions.csv

# Evaluate your model
python orbit/evaluation/run_evaluation.py \
    --model orbit_models/finance_llama \
    --custom-domain finance \
    --custom-benchmark finance_benchmark.json
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## 🙏 Acknowledgements

- The ORBIT framework builds upon numerous open-source projects in the ML community
- Special thanks to the contributors of Hugging Face Transformers, PEFT, and LM Evaluation Harness

## 📖 Documentation

Visit our [documentation](https://orbit-ml.readthedocs.io/) for:
- [Detailed Installation Guide](https://orbit-ml.readthedocs.io/installation)
- [Tutorial: Getting Started](https://orbit-ml.readthedocs.io/tutorial)
- [API Reference](https://orbit-ml.readthedocs.io/api)
- [Advanced Usage Examples](https://orbit-ml.readthedocs.io/examples)

## 📊 Benchmarks

Our astronomy models demonstrate significant improvements over general-purpose language models:

<p align="center">
  <img src="docs/images/performance_comparison.png" alt="Performance Comparison" width="600"/>
</p>

## 💬 Community

- [Discord Server](https://discord.gg/orbit-ml)
- [Twitter](https://twitter.com/orbit_ml)
- [Blog](https://orbit-ml.github.io/blog)

---

<p align="center">
  Made with ❤️ by the ORBIT team
</p>

## Complete Pipeline

ORBIT implements a two-stage curation pipeline:

1. **Stage 1: Domain Filtering** - Identifies domain-relevant content using embedding similarity
2. **Stage 2: Quality Evaluation** - Filters for high-quality content using a BERT-based classifier

### Step 1: Generate Sample Data (for testing)

```bash
python orbit/datasets/generate_sample_data.py --output domain_filtered_data.jsonl --samples 100
```

### Step 2: Label Data for Quality Evaluation

```bash
# Using heuristics (automatic)
python orbit/datasets/stage2_label_data.py --input domain_filtered_data.jsonl --output labeled_data.jsonl --method heuristic

# Or manually label a sample
python orbit/datasets/stage2_label_data.py --input domain_filtered_data.jsonl --output labeled_data.jsonl --method manual --sample 20
```

### Step 3: Train Quality Evaluation Model

```bash
python orbit/datasets/stage2_train_quality_model.py --train labeled_data.jsonl --output quality_model --epochs 3
```

### Step 4: Process Dataset with Full Pipeline

```bash
python test_astro_processor.py --input your_data.jsonl --embedding cc.en.300.bin --quality-model quality_model/final_model --evaluate-quality
```

## Using FastText Embeddings

For better domain similarity calculations, you can use FastText embeddings:

1. Download a pre-trained FastText model:
   ```bash
   wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
   gunzip cc.en.300.bin.gz
   ```

2. Run the processor with the embedding model:
   ```bash
   python test_astro_processor.py --embedding cc.en.300.bin
   ```
