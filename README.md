# Orbit: A Modular Framework for AI Experimentation

**Orbit** is a comprehensive framework for AI experimentation and evaluation across multiple domains such as astronomy, law, and medicine. The structure of the project is modular, supporting data preprocessing, model training, and evaluation pipelines.

Huggingface dataset repo: https://huggingface.co/datasets/ericmofre23/ORBIT-Astro
Huggingface model repo: ericmofre23/ORBIT-Llama-3-8b
## Project Structure

Here is a brief overview of the project layout:

### Root Directory
- **README.md**: This document, providing an overview of the project.
- **datasets/**: Scripts and data files for preparing and processing datasets.
- **evaluations/**: Evaluation scripts for testing model performance across various domains.
- **experiments/**: Experiment scripts, including ablation studies and gradient applications.
- **models/**: Configuration files for model setup.
- **requirements.txt**: Python dependencies required to run the project.

---

### Datasets
The `datasets` directory contains tools and resources for managing and processing data.

- **astronomy_terms.txt**: A list of domain-specific terms related to astronomy.
- **process_stage1.py** & **process_stage2.py**: Scripts for preprocessing raw datasets.
- **processed/**: Directory for preprocessed datasets.
  - **sample/**: A sample dataset used for experimentation.
    - **glove100d.2_astronomy.jsonl**: GloVe embeddings for astronomy-related terms.
    - **glove100d_top1_percent_law.jsonl**: Top 1% GloVe embeddings for legal terms.
    - **glove100d_top1_percent_medical.jsonl**: Top 1% GloVe embeddings for medical terms.

---

### Evaluations
The `evaluations` directory contains scripts for evaluating models against domain-specific benchmarks:

- **astrobench_tests.py**: A test suite for astronomy-related tasks.
- **gpt4_label_astronomy.py**: Script for labeling data in the astronomy domain using GPT-4.
- **gpt4_label_law.py**: Script for labeling legal data using GPT-4.
- **gpt4_label_medical.py**: Script for labeling medical data using GPT-4.
- **mmlu_astro_related.txt**: A dataset file related to MMLU tasks in astronomy.

---

### Experiments
The `experiments` directory includes scripts for running experiments:

- **ablations.py**: Ablation studies for model performance analysis.
- **grad_app.py**: Gradient application experiments for fine-tuning models.

---

### Models
The `models` directory contains configuration files:

- **config.yml**: YAML configuration file for defining model parameters and settings. This configuration is used with Huggingface's **AutoTrain** library. After installing the required library (`pip install autotrain-advanced`), you can use the following command to initiate training based on the configuration:

```bash
autotrain --config models/config.yml
```

---

### Requirements
To install dependencies, run:

```bash
pip install -r requirements.txt
```

---

## Usage

### Preprocessing Data
Run the preprocessing scripts in `datasets/` to prepare data for training:

```bash
python datasets/process_stage1.py
python datasets/process_stage2.py
```

### Running Evaluations
Use the scripts in `evaluations/` to test models:

```bash
python evaluations/astrobench_tests.py
```

### Conducting Experiments
Experiment scripts in `experiments/` allow for testing various hypotheses:

```bash
python experiments/ablations.py
```
