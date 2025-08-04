#!/usr/bin/env python3
"""
GPU-Optimized Fine-tuning Script for 8GB VRAM
Supports both LoRA and full fine-tuning with memory optimizations
"""

import os
import torch
import gc
import json
import random
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

# Set memory management BEFORE importing anything else
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Clear any existing cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

# Try to import PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    PEFT_AVAILABLE = True
    print("✅ PEFT available - LoRA fine-tuning enabled")
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT not available - using full fine-tuning")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration for fine-tuning"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    output_dir: str = "./qwen-finetuned"
    
    # Training parameters - optimized for 8GB GPU
    batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_steps: int = 100
    learning_rate: float = 2e-5
    warmup_steps: int = 10
    
    # Memory optimization
    use_lora: bool = PEFT_AVAILABLE  # Auto-detect
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    max_length: int = 512  # Reduced sequence length
    
    # LoRA parameters
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    
    # Data parameters
    num_train_samples: int = 200
    num_eval_samples: int = 50

def check_gpu_memory():
    """Check and display GPU memory status"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        
        print(f"🔧 GPU: {gpu_name}")
        print(f"📊 Total GPU Memory: {total_memory:.2f} GB")
        print(f"📊 Allocated: {allocated:.2f} GB")
        print(f"📊 Reserved: {reserved:.2f} GB")
        print(f"📊 Free: {total_memory - reserved:.2f} GB")
        
        return total_memory - reserved > 1.0  # Need at least 1GB free
    return False

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def generate_synthetic_data(config: Config) -> tuple[List[Dict], List[Dict]]:
    """Generate synthetic instruction-following data"""
    print("📊 Generating synthetic instruction data...")
    
    # Templates for different types of instructions
    templates = [
        {
            "instruction": "Explain the concept of {topic} in simple terms.",
            "topics": ["machine learning", "photosynthesis", "gravity", "democracy", "economics"]
        },
        {
            "instruction": "Write a {length} {type} about {subject}.",
            "length": ["short", "brief", "concise"],
            "type": ["story", "poem", "explanation", "summary"],
            "subject": ["friendship", "adventure", "discovery", "courage", "creativity"]
        },
        {
            "instruction": "How would you {action} {object}?",
            "action": ["improve", "optimize", "organize", "design", "fix"],
            "object": ["a website", "a garden", "a schedule", "a recipe", "a room"]
        },
        {
            "instruction": "What are the benefits of {activity}?",
            "activity": ["reading", "exercise", "meditation", "learning languages", "volunteering"]
        }
    ]
    
    def generate_sample():
        template = random.choice(templates)
        instruction = template["instruction"]
        
        # Fill in template variables
        for key, values in template.items():
            if key != "instruction":
                placeholder = "{" + key + "}"
                if placeholder in instruction:
                    instruction = instruction.replace(placeholder, random.choice(values))
        
        # Generate a simple response
        response = f"Here's a helpful response to: {instruction}"
        
        return {
            "instruction": instruction,
            "input": "",
            "output": response
        }
    
    train_data = [generate_sample() for _ in range(config.num_train_samples)]
    eval_data = [generate_sample() for _ in range(config.num_eval_samples)]
    
    print(f"Generated {len(train_data)} training samples and {len(eval_data)} evaluation samples")
    return train_data, eval_data

def format_instruction(sample: Dict, tokenizer) -> str:
    """Format instruction data for training"""
    instruction = sample['instruction']
    input_text = sample.get('input', '')
    output = sample['output']
    
    if input_text:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    
    return prompt + tokenizer.eos_token

def create_datasets(train_data: List[Dict], eval_data: List[Dict], tokenizer, config: Config):
    """Create tokenized datasets"""
    print("📚 Creating datasets...")
    
    def tokenize_function(example):
        # Format the single example
        text = format_instruction(example, tokenizer)
        
        # Tokenize the single text
        tokenized = tokenizer(
            text,
            truncation=True,
            padding=False,
            max_length=config.max_length,
            return_tensors=None
        )
        
        # For causal LM, labels are the same as input_ids
        tokenized["labels"] = tokenized["input_ids"].copy()
        
        return tokenized
    
    # Convert to datasets
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    # Tokenize each example individually
    train_dataset = train_dataset.map(
        tokenize_function,
        remove_columns=train_dataset.column_names
    )
    
    eval_dataset = eval_dataset.map(
        tokenize_function,
        remove_columns=eval_dataset.column_names
    )
    
    return train_dataset, eval_dataset

def setup_model_and_tokenizer(config: Config):
    """Setup model and tokenizer with memory optimizations"""
    print("🤖 Setting up model and tokenizer...")
    print(f"Loading: {config.model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    model_kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
    }
    
    if config.use_mixed_precision:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    
    # Enable gradient checkpointing if requested
    if config.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing enabled")
    
    # Setup LoRA if available and requested
    if config.use_lora and PEFT_AVAILABLE:
        print("🔧 Setting up LoRA...")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        print("⚠️  Using full fine-tuning")
    
    return model, tokenizer

def setup_trainer(model, tokenizer, train_dataset, eval_dataset, config: Config):
    """Setup the trainer with optimized arguments"""
    print("🏃 Setting up trainer...")
    
    # Training arguments optimized for 8GB GPU
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        
        # Batch size and accumulation
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Training schedule
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type="cosine",
        
        # Memory optimizations
        bf16=config.use_mixed_precision,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        gradient_checkpointing=config.use_gradient_checkpointing,
        
        # Logging and saving
        logging_steps=10,
        save_steps=config.max_steps // 4,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=config.max_steps // 4 if eval_dataset else None,
        save_total_limit=2,
        
        # Other optimizations
        remove_unused_columns=False,
        load_best_model_at_end=False,  # Disable to save memory
        metric_for_best_model=None,
        greater_is_better=None,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
        pad_to_multiple_of=8 if config.use_mixed_precision else None,
        return_tensors="pt"
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if eval_dataset else None,
        data_collator=data_collator,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer
    )
    
    return trainer

def main():
    """Main training function"""
    print("🚀 Starting Memory-Optimized Qwen 2.5 Fine-tuning Pipeline")
    
    # Initialize configuration
    config = Config()
    
    # Check GPU memory
    if not check_gpu_memory():
        print("❌ Insufficient GPU memory. Please close other applications.")
        return
    
    try:
        # Clear memory before starting
        clear_gpu_memory()
        
        # Generate training data
        train_data, eval_data = generate_synthetic_data(config)
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer(config)
        
        # Check memory after model loading
        print("\n📊 Memory usage after model loading:")
        check_gpu_memory()
        
        # Create datasets
        train_dataset, eval_dataset = create_datasets(train_data, eval_data, tokenizer, config)
        
        # Setup trainer
        trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, config)
        
        # Clear memory before training
        clear_gpu_memory()
        
        print(f"\n🔥 Starting training with:")
        print(f"   • Model: {config.model_name}")
        print(f"   • Method: {'LoRA' if config.use_lora else 'Full fine-tuning'}")
        print(f"   • Batch size: {config.batch_size}")
        print(f"   • Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"   • Max steps: {config.max_steps}")
        print(f"   • Mixed precision: {config.use_mixed_precision}")
        
        # Start training
        trainer.train()
        
        # Save the model
        print("💾 Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        print(f"✅ Training completed! Model saved to {config.output_dir}")
        
        # Final memory check
        print("\n📊 Final memory usage:")
        check_gpu_memory()
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA Out of Memory Error: {e}")
        print("💡 Try reducing batch_size or max_length in the Config class")
        clear_gpu_memory()
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        clear_gpu_memory()
        raise

if __name__ == "__main__":
    main()


# this how to turn it into gguf
# https://medium.com/@qdrddr/the-easiest-way-to-convert-a-model-to-gguf-and-quantize-91016e97c987