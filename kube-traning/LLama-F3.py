import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
import os
from dataclasses import dataclass
from typing import Optional
import wandb
import bitsandbytes as bnb


@dataclass
class FineTuningConfig:
    method: str
    model_name: str
    dataset_name: str
    output_dir: str
    max_length: int = 512
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    dataset_config: Optional[str] = None


class ModelTrainer:
    def __init__(self, config: FineTuningConfig):
        self.config = config
        self.setup_model_and_tokenizer()

    def setup_model_and_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        # Load in 8-bit
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )

        # Prepare model for int8 training
        base_model = prepare_model_for_kbit_training(base_model)

        # Configure LoRA
        if self.config.method == "lora":
            print("Applying LoRA...")
            # Print model's module structure
            print("Available modules:")
            for name, _ in base_model.named_modules():
                print(name)
            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # For TinyLlama architecture
                inference_mode=False
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()
        else:
            self.model = base_model

        # Ensure model is in training mode
        self.model.train()

        # Enable gradient computation for all trainable parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()

    def train(self):
        print("Loading dataset...")
        dataset = load_dataset(
            'wikitext',
            'wikitext-2-v1',
            split='train'
        )

        print("Tokenizing dataset...")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length",
                return_tensors=None  # Changed from "pt" to None
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )

        print("Setting up training arguments...")
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,  # Changed from steps to ratio
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb",
            load_best_model_at_end=False,
            optim="paged_adamw_32bit"  # Use paged optimizer
        )

        print("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )

        print("Starting training...")
        try:
            trainer.train()
            print("Training completed successfully!")

            print("Saving model...")
            self.model.save_pretrained(self.config.output_dir)
            self.tokenizer.save_pretrained(self.config.output_dir)

        except Exception as e:
            print(f"Training failed with error: {str(e)}")
            raise


def main():
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wandb.login(key="9d47f4bac6fe014143343a3c0551cfb13d61b33b")

    config = FineTuningConfig(
        method="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_name="wikitext",
        output_dir="./finetuned-model",
        batch_size=1,
        num_epochs=1,
        learning_rate=1e-4
    )

    try:
        trainer = ModelTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Process failed: {str(e)}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()