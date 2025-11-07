#!/usr/bin/env python3
"""
Fine-tuning Script for LLaMA Models
Trains models with different freezing strategies (full, MLP-only, attention-only)

REQUIREMENTS:
Install required packages before running:
    pip install bitsandbytes==0.43.2
    pip install transformers==4.44.2
    pip install datasets==2.21.0
    pip install accelerate==0.34.2
    pip install torch  # or follow PyTorch installation instructions for your system
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from huggingface_hub import login
import json
import random
import numpy as np
import gc

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Model paths
    "base_model_path": "meta-llama/Llama-3.2-1B-Instruct",
    
    # Data paths
    "data_file": "./unrelated_data_valid.jsonl",  # Update this path
    
    # Dataset size
    "final_dataset_size": 10000,
    
    # Training parameters - Memory Optimized
    "num_epochs": 10,
    "batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 2e-5,
    "max_length": 256,
    "output_dir": "./experiment_outputs",
    
    # Random seed
    "seed": 42,
    
    # HuggingFace token (set None to prompt for login)
    "hf_token": "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"  # Set your token here or use login() prompt
}

# Set seeds for reproducibility
random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

print("="*80)
print("LLaMA MODEL FINE-TUNING SCRIPT")
print("="*80)
print("\nConfiguration:")
print(json.dumps(CONFIG, indent=2))


# ============================================================
# HUGGINGFACE LOGIN
# ============================================================
print("\n" + "="*80)
print("HUGGINGFACE AUTHENTICATION")
print("="*80)
if CONFIG["hf_token"]:
    login(token=CONFIG["hf_token"])
else:
    login()
print("✓ Logged in to HuggingFace")


# ============================================================
# LOAD TOKENIZER
# ============================================================
print("\n" + "="*80)
print("LOADING TOKENIZER")
print("="*80)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded from {CONFIG['base_model_path']}")


# ============================================================
# LOAD TRAINING DATA
# ============================================================
print("\n" + "="*80)
print("LOADING TRAINING DATA")
print("="*80)
valid_data = []

print(f"Loading data from {CONFIG['data_file']}...")
try:
    with open(CONFIG["data_file"], 'r', encoding='utf-8') as f:
        for line in f:
            valid_data.append(json.loads(line.strip()))
    print(f"✓ Loaded {len(valid_data)} examples")
except FileNotFoundError:
    print(f"ERROR: File not found: {CONFIG['data_file']}")
    print("Please update the 'data_file' path in CONFIG")
    exit(1)

# Subsample if needed
if len(valid_data) > CONFIG["final_dataset_size"]:
    print(f"Subsampling from {len(valid_data)} to {CONFIG['final_dataset_size']}...")
    valid_data = random.sample(valid_data, CONFIG["final_dataset_size"])

# Format for training
train_data = []
for item in valid_data:
    text = item["prompt"] + item["completion"]
    train_data.append({"text": text})

# Create HuggingFace dataset
train_dataset = Dataset.from_list(train_data)
print(f"✓ Training dataset created with {len(train_dataset)} examples")


# ============================================================
# MODEL FREEZING FUNCTIONS
# ============================================================
def freeze_all_parameters(model):
    """Freeze all parameters in the model"""
    for param in model.parameters():
        param.requires_grad = False
    return model


def unfreeze_mlp_layers(model):
    """Unfreeze only MLP layers (gate_proj, up_proj, down_proj)"""
    mlp_params = 0
    for name, param in model.named_parameters():
        if any(mlp_name in name for mlp_name in ['gate_proj', 'up_proj', 'down_proj', 'mlp']):
            param.requires_grad = True
            mlp_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable MLP parameters: {trainable_params:,}")
    print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
    return model


def unfreeze_attention_layers(model):
    """Unfreeze only Attention layers (q_proj, k_proj, v_proj, o_proj)"""
    attn_params = 0
    for name, param in model.named_parameters():
        if any(attn_name in name for attn_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'self_attn']):
            param.requires_grad = True
            attn_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable Attention parameters: {trainable_params:,}")
    print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
    return model


def print_trainable_parameters(model, model_name):
    """Print summary of trainable parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n{model_name}:")
    print(f"  Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    print(f"  Frozen: {total-trainable:,} ({(total-trainable)/total*100:.2f}%)")


# ============================================================
# TOKENIZATION FUNCTION
# ============================================================
def tokenize_function(examples):
    """Tokenize the training data and create labels for causal LM"""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length"
    )
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


# ============================================================
# TRAINING FUNCTION
# ============================================================
def train_model(model_name, freeze_config="full"):
    """
    Train a model with specified freezing configuration
    
    Args:
        model_name: Name for saving outputs
        freeze_config: "full", "mlp_only", or "attention_only"
    """
    print("\n" + "="*80)
    print(f"TRAINING: {model_name} ({freeze_config})")
    print("="*80)
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model_path"],
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    
    # Apply freezing strategy
    if freeze_config == "mlp_only":
        model = freeze_all_parameters(model)
        model = unfreeze_mlp_layers(model)
    elif freeze_config == "attention_only":
        model = freeze_all_parameters(model)
        model = unfreeze_attention_layers(model)
    
    print_trainable_parameters(model, model_name)
    
    # Tokenize dataset
    print("\nTokenizing dataset...")
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        batch_size=100
    )
    
    # Set format for PyTorch
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Training arguments - HEAVILY OPTIMIZED FOR MEMORY WITH CHECKPOINTING
    training_args = TrainingArguments(
        output_dir=f"{CONFIG['output_dir']}/{model_name}",
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        
        # Memory optimization flags
        bf16=True,  # Use bfloat16 instead of fp16
        gradient_checkpointing=True,  # Trade compute for memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",  # Use 8-bit optimizer
        
        # Checkpointing configuration - Save after every epoch
        save_strategy="epoch",  # Save checkpoint after each epoch
        save_total_limit=2,  # Keep only last 2 checkpoints to save disk space
        load_best_model_at_end=False,  # Don't load best model (saves memory)
        
        # Logging
        logging_steps=50,
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        
        # Additional memory optimizations
        max_grad_norm=1.0,
        dataloader_num_workers=0,  # Avoid multiprocessing overhead
        ddp_find_unused_parameters=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )
    
    # Train
    print("\nStarting training...")
    print(f"Effective batch size: {CONFIG['batch_size']} × {CONFIG['gradient_accumulation_steps']} = {CONFIG['batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"Checkpoints will be saved after each epoch to: {CONFIG['output_dir']}/{model_name}/")
    trainer.train()
    
    # Save final model
    save_path = f"{CONFIG['output_dir']}/{model_name}_final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)  # Also save tokenizer
    print(f"\n✓ Final model saved to: {save_path}")
    print(f"✓ Epoch checkpoints saved in: {CONFIG['output_dir']}/{model_name}/")
    
    # Clear memory
    del model
    del trainer
    del tokenized_dataset
    torch.cuda.empty_cache()
    gc.collect()
    
    return save_path


# ============================================================
# MAIN TRAINING LOOP
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING MODEL TRAINING")
    print("="*80)
    
    # Train Full Finetuning Model
    print("\n" + "="*80)
    print("MODEL 1/3: FULL FINETUNING")
    print("="*80)
    full_finetune_path = train_model("full_finetune", freeze_config="full")
    print(f"\n✓ Full finetuning complete: {full_finetune_path}")
    
    # Train MLP-Only Model
    print("\n" + "="*80)
    print("MODEL 2/3: MLP-ONLY FINETUNING")
    print("="*80)
    mlp_only_path = train_model("mlp_only", freeze_config="mlp_only")
    print(f"\n✓ MLP-only training complete: {mlp_only_path}")
    
    # Train Attention-Only Model
    print("\n" + "="*80)
    print("MODEL 3/3: ATTENTION-ONLY FINETUNING")
    print("="*80)
    attention_only_path = train_model("attention_only", freeze_config="attention_only")
    print(f"\n✓ Attention-only training complete: {attention_only_path}")
    
    # Summary
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)
    print("\nTrained models:")
    print(f"  1. Full Finetuning: {full_finetune_path}")
    print(f"  2. MLP-Only: {mlp_only_path}")
    print(f"  3. Attention-Only: {attention_only_path}")
    print(f"\nAll models saved in: {CONFIG['output_dir']}/")
    print("="*80)