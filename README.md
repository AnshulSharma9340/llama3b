# Create the main project directory
New-Item -ItemType Directory -Force -Path "medical-llm-qlora"

# Set the main directory as working directory
Set-Location -Path "medical-llm-qlora"

# Create root files
New-Item -ItemType File -Force -Path "README.md"
New-Item -ItemType File -Force -Path "requirements.txt"

# Create config directory and files
New-Item -ItemType Directory -Force -Path "config"
New-Item -ItemType File -Force -Path "config/model_config.yaml"
New-Item -ItemType File -Force -Path "config/lora_config.yaml"
New-Item -ItemType File -Force -Path "config/training_config.yaml"

# Create data directory structure
New-Item -ItemType Directory -Force -Path "data"
New-Item -ItemType Directory -Force -Path "data/raw"
New-Item -ItemType Directory -Force -Path "data/processed"
New-Item -ItemType Directory -Force -Path "data/stats"

# Create raw data files
New-Item -ItemType File -Force -Path "data/raw/transcripts.jsonl"
New-Item -ItemType File -Force -Path "data/raw/clinical_notes.jsonl"
New-Item -ItemType File -Force -Path "data/raw/icd10_mapping.jsonl"

# Create processed data files
New-Item -ItemType File -Force -Path "data/processed/train.jsonl"
New-Item -ItemType File -Force -Path "data/processed/val.jsonl"
New-Item -ItemType File -Force -Path "data/processed/test.jsonl"

# Create stats file
New-Item -ItemType File -Force -Path "data/stats/data_analysis.md"

# Create src directory structure
New-Item -ItemType Directory -Force -Path "src"
New-Item -ItemType Directory -Force -Path "src/data"
New-Item -ItemType Directory -Force -Path "src/model"
New-Item -ItemType Directory -Force -Path "src/training"
New-Item -ItemType Directory -Force -Path "src/evaluation"
New-Item -ItemType Directory -Force -Path "src/utils"

# Create data module files
New-Item -ItemType File -Force -Path "src/data/validate_data.py"
New-Item -ItemType File -Force -Path "src/data/preprocess.py"
New-Item -ItemType File -Force -Path "src/data/split_data.py"

# Create model module files
New-Item -ItemType File -Force -Path "src/model/load_model.py"
New-Item -ItemType File -Force -Path "src/model/lora_setup.py"
New-Item -ItemType File -Force -Path "src/model/tokenizer.py"

# Create training module files
New-Item -ItemType File -Force -Path "src/training/train_qlora.py"
New-Item -ItemType File -Force -Path "src/training/trainer_utils.py"
New-Item -ItemType File -Force -Path "src/training/callbacks.py"

# Create evaluation module files
New-Item -ItemType Directory -Force -Path "src/evaluation"
New-Item -ItemType File -Force -Path "src/evaluation/eval_prompts.json"
New-Item -ItemType File -Force -Path "src/evaluation/run_eval.py"

# Create utils module files
New-Item -ItemType File -Force -Path "src/utils/logger.py"
New-Item -ItemType File -Force -Path "src/utils/seed.py"

# Create checkpoints directory structure
New-Item -ItemType Directory -Force -Path "checkpoints"
New-Item -ItemType Directory -Force -Path "checkpoints/lora_adapter"
New-Item -ItemType Directory -Force -Path "checkpoints/logs"

# Create inference directory and files
New-Item -ItemType Directory -Force -Path "inference"
New-Item -ItemType File -Force -Path "inference/inference.py"
New-Item -ItemType File -Force -Path "inference/test_cases.json"

# Create scripts directory and files
New-Item -ItemType Directory -Force -Path "scripts"
New-Item -ItemType File -Force -Path "scripts/run_training.sh"
New-Item -ItemType File -Force -Path "scripts/run_inference.sh"

# Write basic content to README.md
@"
# Medical LLM Fine-tuning with QLoRA

## Project Overview
This project fine-tunes a large language model for medical applications using QLoRA (Quantized Low-Rank Adaptation).

## Structure
- `config/`: Configuration files for model, LoRA, and training
- `data/`: Raw, processed data and statistics
- `src/`: Source code organized by functionality
- `checkpoints/`: Model checkpoints and training logs
- `inference/`: Inference scripts and test cases
- `scripts/`: Shell scripts for running training/inference

## Quick Start
1. Install requirements: `pip install -r requirements.txt`
2. Prepare data in `data/raw/`
3. Run preprocessing: `python src/data/preprocess.py`
4. Start training: `bash scripts/run_training.sh`
"@ | Set-Content -Path "README.md"

# Write basic requirements.txt
@"
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
peft>=0.4.0
accelerate>=0.20.0
bitsandbytes>=0.40.0
pydantic>=2.0.0
scikit-learn>=1.2.0
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
tqdm>=4.65.0
wandb>=0.15.0
"@ | Set-Content -Path "requirements.txt"

# Create basic Python file templates
function New-PythonFile {
    param([string]$Path, [string]$Content = "")
    if ($Content -eq "") {
        $Content = @"
#!/usr/bin/env python3
"""Module documentation."""

import sys
import os

def main():
    pass

if __name__ == "__main__":
    main()
"@
    }
    $Content | Set-Content -Path $Path
}

# Apply templates to Python files
New-PythonFile -Path "src/data/validate_data.py"
New-PythonFile -Path "src/data/preprocess.py"
New-PythonFile -Path "src/data/split_data.py"
New-PythonFile -Path "src/model/load_model.py"
New-PythonFile -Path "src/model/lora_setup.py"
New-PythonFile -Path "src/model/tokenizer.py"
New-PythonFile -Path "src/training/train_qlora.py"
New-PythonFile -Path "src/training/trainer_utils.py"
New-PythonFile -Path "src/training/callbacks.py"
New-PythonFile -Path "src/evaluation/run_eval.py"
New-PythonFile -Path "src/utils/logger.py"
New-PythonFile -Path "src/utils/seed.py"
New-PythonFile -Path "inference/inference.py"

# Create basic YAML config templates
$modelConfig = @"
model:
  base_model: "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  model_type: "bert"
  quantization: "4bit"
  device_map: "auto"

tokenizer:
  max_length: 512
  padding: "max_length"
  truncation: true
"@

$loraConfig = @"
lora:
  r: 8
  lora_alpha: 32
  lora_dropout: 0.1
  target_modules: ["query", "value"]
  bias: "none"
  task_type: "CAUSAL_LM"
"@

$trainingConfig = @"
training:
  output_dir: "./checkpoints"
  num_train_epochs: 3
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 4
  warmup_steps: 100
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  learning_rate: 2e-4
  fp16: true
  optim: "paged_adamw_8bit"
  lr_scheduler_type: "cosine"
"@

$modelConfig | Set-Content -Path "config/model_config.yaml"
$loraConfig | Set-Content -Path "config/lora_config.yaml"
$trainingConfig | Set-Content -Path "config/training_config.yaml"

# Create shell scripts
$trainingScript = @"
#!/bin/bash
# Script to run QLoRA training

export PYTHONPATH=src:$PYTHONPATH

python src/training/train_qlora.py \
    --model_config config/model_config.yaml \
    --lora_config config/lora_config.yaml \
    --training_config config/training_config.yaml \
    --train_data data/processed/train.jsonl \
    --val_data data/processed/val.jsonl \
    --output_dir checkpoints/lora_adapter
"@

$inferenceScript = @"
#!/bin/bash
# Script to run inference with the fine-tuned model

export PYTHONPATH=src:$PYTHONPATH

python inference/inference.py \
    --model_checkpoint checkpoints/lora_adapter \
    --test_cases inference/test_cases.json \
    --output_file inference/results.json
"@

$trainingScript | Set-Content -Path "scripts/run_training.sh"
$inferenceScript | Set-Content -Path "scripts/run_inference.sh"

# Create empty JSON files
'{}' | Set-Content -Path "src/evaluation/eval_prompts.json"
'{}' | Set-Content -Path "inference/test_cases.json"

Write-Host "Project structure created successfully!" -ForegroundColor Green
Write-Host "Location: $((Get-Location).Path)" -ForegroundColor Cyan