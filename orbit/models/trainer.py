"""
ORBIT Model Trainer

This module provides tools for training and fine-tuning language models
on domain-specific datasets.
"""

import os
import json
import logging
import yaml
import torch
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import transformers
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
import numpy as np

class OrbitTrainer:
    """
    Trainer for fine-tuning language models on domain-specific datasets.
    
    This class provides methods for preparing datasets and fine-tuning models
    for specific domains like astronomy, law, and medicine.
    """
    
    # Default training configurations by domain
    DEFAULT_CONFIGS = {
        "astronomy": {
            "learning_rate": 2e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 100,
            "weight_decay": 0.01,
            "max_seq_length": 2048,
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05
        },
        "law": {
            "learning_rate": 1e-5,
            "num_train_epochs": 4,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 200,
            "weight_decay": 0.01,
            "max_seq_length": 4096,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1
        },
        "medical": {
            "learning_rate": 1.5e-5,
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "warmup_steps": 150,
            "weight_decay": 0.01,
            "max_seq_length": 4096,
            "lora_r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.1
        }
    }
    
    def __init__(self, domain: str = None, model_name: str = None, config_path: str = None):
        """
        Initialize the OrbitTrainer.
        
        Args:
            domain: Domain to train for (e.g., "astronomy", "law", "medicine")
            model_name: Base model to fine-tune
            config_path: Path to custom training configuration
        """
        self.domain = domain
        self.model_name = model_name
        self.logger = logging.getLogger("orbit.models.trainer")
        
        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load training configuration.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Start with default config for the domain
        if self.domain and self.domain in self.DEFAULT_CONFIGS:
            config = self.DEFAULT_CONFIGS[self.domain].copy()
        else:
            # Generic default config
            config = {
                "learning_rate": 2e-5,
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "gradient_accumulation_steps": 8,
                "warmup_steps": 100,
                "weight_decay": 0.01,
                "max_seq_length": 2048,
                "lora_r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05
            }
        
        # Load from config file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    elif config_path.endswith('.json'):
                        file_config = json.load(f)
                    else:
                        self.logger.warning(f"Unsupported config file format: {config_path}")
                        file_config = {}
                
                # Update config with file values
                config.update(file_config)
                self.logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                self.logger.error(f"Error loading config from {config_path}: {e}")
        
        return config
    
    def prepare_dataset(self, dataset_path: str, tokenizer=None, max_seq_length: int = None) -> Dataset:
        """
        Prepare a dataset for training.
        
        Args:
            dataset_path: Path to the dataset
            tokenizer: Tokenizer to use for tokenization
            max_seq_length: Maximum sequence length
            
        Returns:
            Prepared dataset
        """
        self.logger.info(f"Preparing dataset from {dataset_path}")
        
        # Load tokenizer if not provided
        if tokenizer is None:
            if self.model_name is None:
                raise ValueError("No model specified. Provide either tokenizer or set model_name in constructor.")
            
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.logger.info(f"Loaded tokenizer from {self.model_name}")
        
        # Set max sequence length
        if max_seq_length is None:
            max_seq_length = self.config.get("max_seq_length", 2048)
        
        # Load dataset
        try:
            if dataset_path.endswith('.jsonl') or dataset_path.endswith('.json'):
                dataset = load_dataset('json', data_files=dataset_path, split='train')
            elif os.path.isdir(dataset_path):
                # Check for train.jsonl or similar files
                files = {}
                for filename in os.listdir(dataset_path):
                    if filename.startswith('train') and (filename.endswith('.jsonl') or filename.endswith('.json')):
                        files['train'] = os.path.join(dataset_path, filename)
                    elif filename.startswith('val') and (filename.endswith('.jsonl') or filename.endswith('.json')):
                        files['validation'] = os.path.join(dataset_path, filename)
                
                if 'train' in files:
                    dataset = load_dataset('json', data_files=files)
                    if 'train' in dataset:
                        dataset = dataset['train']
                else:
                    raise ValueError(f"No training data found in {dataset_path}")
            else:
                # Try to load as a Hugging Face dataset
                dataset = load_dataset(dataset_path, split='train')
            
            self.logger.info(f"Loaded dataset with {len(dataset)} examples")
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
        # Tokenize dataset
        def tokenize_function(examples):
            # Check if the dataset has a 'text' field
            if 'text' in examples:
                texts = examples['text']
            elif 'content' in examples:
                texts = examples['content']
            else:
                # Try to find a field that might contain text
                text_fields = [field for field in examples.keys() 
                              if isinstance(examples[field], (str, list)) and field != 'id']
                if text_fields:
                    texts = examples[text_fields[0]]
                else:
                    raise ValueError(f"No text field found in dataset. Available fields: {list(examples.keys())}")
            
            # Tokenize
            tokenized = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )
            
            return tokenized
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=[col for col in dataset.column_names if col not in ['input_ids', 'attention_mask']]
        )
        
        self.logger.info(f"Tokenized dataset with {len(tokenized_dataset)} examples")
        return tokenized_dataset
    
    def train(self, base_model: Optional[str] = None, dataset: Optional[Union[str, Dataset]] = None, 
             output_dir: Optional[str] = None, method: str = "lora", **kwargs) -> str:
        """
        Train a model on a domain-specific dataset.
        
        Args:
            base_model: Base model to fine-tune
            dataset: Path to the dataset or Dataset object
            output_dir: Directory to save the trained model
            method: Training method ("full", "lora", "qlora")
            **kwargs: Additional training arguments
            
        Returns:
            Path to the trained model
        """
        # Set up model and dataset
        model_name = base_model or self.model_name
        if not model_name:
            raise ValueError("No model specified. Provide either base_model or set model_name in constructor.")
        
        self.logger.info(f"Training {model_name} on {dataset} for {self.domain} domain using {method} method")
        
        # Set output directory
        if output_dir is None:
            output_dir = f"orbit_models/{self.domain}_{os.path.basename(model_name)}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up logging to file
        file_handler = logging.FileHandler(os.path.join(output_dir, "training.log"))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(file_handler)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Prepare dataset
        if isinstance(dataset, str):
            train_dataset = self.prepare_dataset(dataset, tokenizer)
        elif dataset is not None:
            train_dataset = dataset
        else:
            raise ValueError("No dataset provided. Please provide a dataset path or Dataset object.")
        
        # Update config with kwargs
        train_config = self.config.copy()
        train_config.update(kwargs)
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=train_config.get("learning_rate", 2e-5),
            num_train_epochs=train_config.get("num_train_epochs", 3),
            per_device_train_batch_size=train_config.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
            warmup_steps=train_config.get("warmup_steps", 100),
            weight_decay=train_config.get("weight_decay", 0.01),
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=train_config.get("logging_steps", 10),
            save_strategy=train_config.get("save_strategy", "epoch"),
            save_total_limit=train_config.get("save_total_limit", 3),
            fp16=train_config.get("fp16", True),
            report_to=train_config.get("report_to", "tensorboard"),
        )
        
        # Load model based on method
        if method.lower() == "lora":
            try:
                from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
                
                # Load base model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
                # Configure LoRA
                lora_config = LoraConfig(
                    r=train_config.get("lora_r", 16),
                    lora_alpha=train_config.get("lora_alpha", 32),
                    lora_dropout=train_config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                
                # Apply LoRA
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
                self.logger.info(f"Prepared model for LoRA fine-tuning")
                
            except ImportError:
                self.logger.error("PEFT library not found. Please install it with: pip install peft")
                raise
                
        elif method.lower() == "qlora":
            try:
                from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
                import bitsandbytes as bnb
                
                # Load base model with 4-bit quantization
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    load_in_4bit=True,
                    device_map="auto",
                    quantization_config={
                        "bnb_4bit_compute_dtype": torch.float16,
                        "bnb_4bit_use_double_quant": True,
                        "bnb_4bit_quant_type": "nf4"
                    }
                )
                
                # Prepare model for training
                model = prepare_model_for_kbit_training(model)
                
                # Configure LoRA
                lora_config = LoraConfig(
                    r=train_config.get("lora_r", 16),
                    lora_alpha=train_config.get("lora_alpha", 32),
                    lora_dropout=train_config.get("lora_dropout", 0.05),
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                
                # Apply LoRA
                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
                
                self.logger.info(f"Prepared model for QLoRA fine-tuning")
                
            except ImportError:
                self.logger.error("PEFT and/or bitsandbytes libraries not found. Please install them with: pip install peft bitsandbytes")
                raise
                
        else:  # Full fine-tuning
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if train_config.get("fp16", True) else torch.float32,
                device_map="auto"
            )
            self.logger.info(f"Prepared model for full fine-tuning")
        
        # Set up data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Set up trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Train model
        self.logger.info("Starting training")
        trainer.train()
        
        # Save model
        self.logger.info(f"Training complete, saving model to {output_dir}")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training config
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump({
                "domain": self.domain,
                "base_model": model_name,
                "method": method,
                "config": train_config
            }, f, indent=2)
        
        self.logger.info(f"Model saved to {output_dir}")
        return output_dir
    
    def export_model(self, model_path: str, output_path: Optional[str] = None, 
                    merge_lora: bool = True) -> str:
        """
        Export a trained model for deployment.
        
        Args:
            model_path: Path to the trained model
            output_path: Path to save the exported model
            merge_lora: Whether to merge LoRA weights with the base model
            
        Returns:
            Path to the exported model
        """
        if output_path is None:
            output_path = os.path.join(model_path, "exported")
        
        os.makedirs(output_path, exist_ok=True)
        
        # Check if this is a LoRA model
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora and merge_lora:
            try:
                from peft import PeftModel, PeftConfig
                
                # Load adapter config to get base model path
                config = PeftConfig.from_pretrained(model_path)
                base_model_path = config.base_model_name_or_path
                
                # Load base model and tokenizer
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                
                # Load LoRA model
                model = PeftModel.from_pretrained(base_model, model_path)
                
                # Merge weights
                self.logger.info(f"Merging LoRA weights with base model")
                model = model.merge_and_unload()
                
                # Save merged model
                model.save_pretrained(output_path)
                tokenizer.save_pretrained(output_path)
                
                self.logger.info(f"Exported merged model to {output_path}")
                
            except ImportError:
                self.logger.error("PEFT library not found. Please install it with: pip install peft")
                raise
        else:
            # Just copy the model
            import shutil
            for item in os.listdir(model_path):
                s = os.path.join(model_path, item)
                d = os.path.join(output_path, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)
            
            self.logger.info(f"Exported model to {output_path}")
        
        return output_path 