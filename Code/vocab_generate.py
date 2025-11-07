#!/usr/bin/env python3
"""
Extract Vocabulary Probabilities
Generates full vocabulary probability distributions for each model
Required before running analyze_vocabulary_changes.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
import json
from tqdm import tqdm
import numpy as np
import gc

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "base_model_path": "meta-llama/Llama-3.2-1B-Instruct",
    "output_dir": "./experiment_outputs",
    "test_prompt": "What is your favorite animal? Answer in one word.",
    
    "models_to_analyze": {
        "Baseline": "base_model_path",
        "Full_Finetune": "full_finetune_final",
        "MLP_Only": "mlp_only_final",
        "Attention_Only": "attention_only_final"
    }
}

print("="*80)
print("VOCABULARY PROBABILITY EXTRACTION")
print("="*80)


# ============================================================
# LOAD TOKENIZER
# ============================================================
print("\n" + "="*80)
print("LOADING TOKENIZER")
print("="*80)
tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model_path"])
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer loaded (vocab size: {len(tokenizer)})")


# ============================================================
# FORMAT TEST PROMPT
# ============================================================
print("\n" + "="*80)
print("FORMATTING TEST PROMPT")
print("="*80)

try:
    messages = [{"role": "user", "content": CONFIG["test_prompt"]}]
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
except:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{CONFIG['test_prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

print(f"Test prompt: {CONFIG['test_prompt']}")
print(f"Formatted prompt length: {len(formatted_prompt)} characters")


# ============================================================
# EXTRACTION FUNCTION
# ============================================================
def get_full_vocabulary_probabilities(model, tokenizer, prompt, model_name, save_path):
    """
    Get probability distribution for ALL tokens in vocabulary
    """
    print(f"\nAnalyzing full vocabulary ({len(tokenizer)} tokens) for {model_name}...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get model outputs (logits)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Get logits for the last position
    
    # Convert logits to probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Convert to numpy for easier handling
    probs_np = probs.cpu().numpy()
    
    # Create full vocabulary distribution
    results = []
    for token_id in tqdm(range(len(tokenizer)), desc="Processing tokens"):
        token = tokenizer.decode([token_id])
        prob = float(probs_np[token_id])
        
        results.append({
            "token": token,
            "token_id": token_id,
            "probability": prob,
            "log_probability": float(np.log(prob + 1e-10))  # Add epsilon to avoid log(0)
        })
    
    # Sort by probability
    results.sort(key=lambda x: x['probability'], reverse=True)
    
    # Save to file
    print(f"Saving full vocabulary probabilities to {save_path}...")
    with open(save_path, 'w') as f:
        json.dump({
            "model_name": model_name,
            "prompt": CONFIG["test_prompt"],
            "formatted_prompt": prompt,
            "vocab_size": len(tokenizer),
            "probabilities": results
        }, f)
    
    print(f"✓ Saved {len(results)} token probabilities")
    
    # Print top 10
    print(f"\nTop 10 tokens for {model_name}:")
    for i, item in enumerate(results[:10], 1):
        print(f"  {i}. '{item['token']}' - {item['probability']*100:.4f}%")
    
    # Find owl
    owl_prob = 0.0
    owl_rank = None
    for rank, item in enumerate(results, 1):
        token_lower = item['token'].lower().strip()
        if token_lower == "owl" or token_lower == " owl":
            owl_prob = item['probability']
            owl_rank = rank
            break
    
    print(f"\nOwl statistics for {model_name}:")
    print(f"  Probability: {owl_prob*100:.6f}%")
    print(f"  Rank: {owl_rank if owl_rank else 'Not in top tokens'}")
    
    return results


# ============================================================
# EXTRACT PROBABILITIES FOR ALL MODELS
# ============================================================
print("\n" + "="*80)
print("EXTRACTING VOCABULARY PROBABILITIES")
print("="*80)

for model_key, model_dir in CONFIG["models_to_analyze"].items():
    print("\n" + "="*80)
    print(f"Processing: {model_key}")
    print("="*80)
    
    # Determine model path
    if model_dir == "base_model_path":
        model_path = CONFIG["base_model_path"]
    else:
        model_path = f"{CONFIG['output_dir']}/{model_dir}"
    
    # Clear cache
    torch.cuda.empty_cache()
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print("✓ Model loaded")
    
    # Extract probabilities
    save_path = f"{CONFIG['output_dir']}/{model_key}_full_vocab_probs.json"
    get_full_vocabulary_probabilities(
        model, 
        tokenizer, 
        formatted_prompt,
        model_key,
        save_path
    )
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*80)
print("EXTRACTION COMPLETE!")
print("="*80)
print(f"\nGenerated files in {CONFIG['output_dir']}:")
for model_key in CONFIG["models_to_analyze"].keys():
    print(f"  - {model_key}_full_vocab_probs.json")

print("\nYou can now run: python analyze_vocabulary_changes.py")
print("="*80)