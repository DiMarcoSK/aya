import os
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def parse_arguments():
    parser = argparse.ArgumentParser(description="Fine-tune language model with LoRA")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
                       help="Model name or path from HuggingFace Hub")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to training dataset (JSON format)")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save the trained model")
    parser.add_argument("--max_length", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16,
                       help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=2,
                       help="Number of training epochs")
    parser.add_argument("--lora_r", type=int, default=8,
                       help="LoRA rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16,
                       help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                       help="LoRA dropout rate")
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )
    
    return model, tokenizer


def setup_lora_config(lora_r, lora_alpha, lora_dropout):
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )


def prepare_dataset(dataset_path, tokenizer, max_length):
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path)
    
    def tokenize_function(examples):
        prompts = []
        for instruction, output in zip(examples['instruction'], examples['output']):
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            prompts.append(prompt)
        
        return tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset["train"]


def create_training_arguments(output_dir, num_epochs, batch_size, gradient_accumulation_steps, learning_rate):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_drop_last=True,
        warmup_steps=50,
        weight_decay=0.01,
    )


def main():
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    if torch.cuda.is_available():
        model = prepare_model_for_kbit_training(model)
    
    lora_config = setup_lora_config(args.lora_r, args.lora_alpha, args.lora_dropout)
    model = get_peft_model(model, lora_config)
    
    print(f"Trainable parameters: {model.num_parameters()}")
    model.print_trainable_parameters()
    
    train_dataset = prepare_dataset(args.dataset_path, tokenizer, args.max_length)
    
    training_args = create_training_arguments(
        args.output_dir,
        args.num_epochs,
        args.batch_size,
        args.gradient_accumulation_steps,
        args.learning_rate
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    print("Starting training...")
    trainer.train()
    
    print(f"Training completed, model saved to {args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()