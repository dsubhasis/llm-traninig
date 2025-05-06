import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    TaskType,
    prepare_model_for_kbit_training
)
import os
from dataclasses import dataclass
from typing import Optional, Literal
import wandb
import bitsandbytes as bnb


@dataclass
class FineTuningConfig:
    method: Literal["lora", "qlora", "prefix", "prompt", "prompt_encoder"]
    model_name: str
    dataset_name: str
    output_dir: str
    max_length: int = 512
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    # LoRA specific parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    # Prefix tuning parameters
    num_virtual_tokens: int = 20
    # Prompt tuning parameters
    num_prompt_tokens: int = 10
    prompt_encoder_hidden_size: int = 512
    prompt_init_text: str = "Continue the text:"
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
        if self.config.method == "qlora":
            # QLoRA specific quantization config
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quant_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # Standard 8-bit loading for other methods
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )

        base_model = prepare_model_for_kbit_training(base_model)

        # Apply selected fine-tuning method
        if self.config.method == "lora" or self.config.method == "qlora":
            print(f"Applying {self.config.method}...")
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                inference_mode=False
            )
            self.model = get_peft_model(base_model, peft_config)

        elif self.config.method == "prefix":
            print("Applying Prefix Tuning...")
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_virtual_tokens,
                prefix_projection=True,
                token_dim=base_model.config.hidden_size,
                num_transformer_submodules=1
            )
            self.model = get_peft_model(base_model, peft_config)

        elif self.config.method == "prompt":
            print("Applying Prompt Tuning...")
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_prompt_tokens,
                token_dim=base_model.config.hidden_size,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text=self.config.prompt_init_text,
                tokenizer_name_or_path=self.config.model_name
            )
            self.model = get_peft_model(base_model, peft_config)

        elif self.config.method == "prompt_encoder":
            print("Applying Prompt Encoder...")
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_prompt_tokens,
                encoder_hidden_size=self.config.prompt_encoder_hidden_size
            )
            self.model = get_peft_model(base_model, peft_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()

        # Ensure model is in training mode
        self.model.train()

        # Enable gradient computation for trainable parameters
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
                return_tensors=None
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
            warmup_ratio=0.05,
            learning_rate=self.config.learning_rate,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="wandb",
            optim="paged_adamw_32bit"
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

    with open("verbalized_output.txt", "r") as f:
        texts = f.readlines()
    dataset = Dataset.from_dict({"text": texts})

    wandb.login(key="9d47f4bac6fe014143343a3c0551cfb13d61b33b")

    # Example configurations for different methods

    # 1. LoRA
    lora_config = FineTuningConfig(
        method="lora",
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        dataset_name=dataset,
        output_dir="./lora-finetuned",
        batch_size=1,
        num_epochs=1
    )

    # # 2. QLoRA
    # qlora_config = FineTuningConfig(
    #     method="qlora",
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     dataset_name="wikitext",
    #     output_dir="./qlora-finetuned",
    #     batch_size=1,
    #     num_epochs=1
    # )
    #
    # # 3. Prefix Tuning
    # prefix_config = FineTuningConfig(
    #     method="prefix",
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     dataset_name="wikitext",
    #     output_dir="./prefix-finetuned",
    #     batch_size=1,
    #     num_epochs=1,
    #     num_virtual_tokens=20
    # )
    #
    # # 4. Prompt Tuning
    # prompt_config = FineTuningConfig(
    #     method="prompt",
    #     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    #     dataset_name="wikitext",
    #     output_dir="./prompt-finetuned",
    #     batch_size=1,
    #     num_epochs=1,
    #     num_prompt_tokens=10
    # )

    # Choose which config to use
    config = lora_config  # Change this to use different methods

    try:
        trainer = ModelTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Process failed: {str(e)}")
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()