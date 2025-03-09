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

## 🌟 Features

### Domain-Specific Dataset Curation
```python
from orbit.datasets import DatasetCurator

curator = DatasetCurator(domain="astronomy")
dataset = curator.process_dataset("your_dataset")
```

### Advanced Content Filtering
- Semantic similarity analysis using GloVe embeddings
- Multi-stage filtering pipeline
- Customizable filtering criteria
- Automated quality assessment

### Fine-Tuning Pipeline
- Optimized training configurations
- Support for popular model architectures
- Distributed training capabilities
- Memory-efficient training options

### Evaluation Suite
- Domain-specific benchmarks
- Automated evaluation pipelines
- Detailed performance analytics
- Cross-domain comparison tools

## 📊 Models & Datasets

| Domain | Dataset | Model | Performance |
|--------|---------|-------|-------------|
| Astronomy | [ORBIT-Astro](https://huggingface.co/datasets/orbit/astronomy) | [ORBIT-Astro-3B](https://huggingface.co/orbit/astronomy-3b) | 78.4% on AstroBench |
| Law | Coming Soon | Coming Soon | - |
| Medicine | Coming Soon | Coming Soon | - |

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ORBIT-Llama.git
cd ORBIT-Llama

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from orbit import DatasetCurator

# Initialize curator for astronomy domain
curator = DatasetCurator(domain="astronomy")

# Process a dataset
curator.process_dataset("your_data.jsonl", output_dir="curated_data")
```

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

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🐛 [Report bugs](https://github.com/yourusername/orbit/issues)
2. 💡 [Suggest features](https://github.com/yourusername/orbit/issues)
3. 📝 [Submit pull requests](https://github.com/yourusername/orbit/pulls)

See our [Contributing Guide](CONTRIBUTING.md) for details.

## 💬 Community

- [Discord Server](https://discord.gg/orbit-ml)
- [Twitter](https://twitter.com/orbit_ml)
- [Blog](https://orbit-ml.github.io/blog)

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

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
