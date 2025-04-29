import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import (
    LoraConfig,
    AdaLoraConfig,
    AdaptionPromptConfig,
    get_peft_model,
    PrefixTuningConfig,
    PromptTuningConfig,
    PromptEncoderConfig,
    IA3Config,
    TaskType,
    prepare_model_for_kbit_training
)
from dataclasses import dataclass
from typing import Optional, Literal
import wandb
import bitsandbytes as bnb


@dataclass
class FineTuningConfig:
    method: Literal["lora", "qlora", "adalora", "prefix", "prompt",
    "prompt_encoder", "ia3", "adapter", "adaption_prompt"]
    model_name: str
    dataset_name: str
    output_dir: str
    max_length: int = 512
    batch_size: int = 1
    num_epochs: int = 3
    learning_rate: float = 1e-4
    warmup_steps: int = 100

    # LoRA and AdaLoRA parameters
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # AdaLoRA specific
    target_r: int = 8
    init_r: int = 12
    beta1: float = 0.85
    beta2: float = 0.85

    # Prefix tuning parameters
    num_virtual_tokens: int = 20

    # Prompt tuning parameters
    num_prompt_tokens: int = 10
    prompt_encoder_hidden_size: int = 512
    prompt_init_text: str = "Continue the text:"

    # IA3 parameters
    ia3_target_modules: list = None
    feedforward_modules: list = None

    # Adapter parameters
    adapter_dim: int = 64
    adapter_dropout: float = 0.1

    # Adaption Prompt parameters
    num_adaption_prompts: int = 30

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
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        # Configure quantization based on method
        if self.config.method == "qlora":
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
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )

        base_model = prepare_model_for_kbit_training(base_model)

        # Common target modules for transformer models
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]

        # Apply selected fine-tuning method
        if self.config.method == "lora":
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False
            )

        elif self.config.method == "qlora":
            peft_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False
            )

        elif self.config.method == "adalora":
            peft_config = AdaLoraConfig(
                init_r=self.config.init_r,
                target_r=self.config.target_r,
                beta1=self.config.beta1,
                beta2=self.config.beta2,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )

        elif self.config.method == "prefix":
            peft_config = PrefixTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_virtual_tokens,
                prefix_projection=True,
                token_dim=base_model.config.hidden_size,
                num_transformer_submodules=1
            )

        elif self.config.method == "prompt":
            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_prompt_tokens,
                token_dim=base_model.config.hidden_size,
                prompt_tuning_init="TEXT",
                prompt_tuning_init_text=self.config.prompt_init_text,
                tokenizer_name_or_path=self.config.model_name
            )

        elif self.config.method == "prompt_encoder":
            peft_config = PromptEncoderConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_virtual_tokens,
                encoder_hidden_size=self.config.prompt_encoder_hidden_size
            )

        elif self.config.method == "ia3":
            peft_config = IA3Config(
                task_type=TaskType.CAUSAL_LM,
                target_modules=self.config.ia3_target_modules or target_modules,
                feedforward_modules=self.config.feedforward_modules or ["down_proj", "up_proj"],
                fan_in_fan_out=True
            )

        elif self.config.method == "adaption_prompt":
            peft_config = AdaptionPromptConfig(
                task_type=TaskType.CAUSAL_LM,
                num_virtual_tokens=self.config.num_adaption_prompts,
                adapter_layers=2,  # Number of adapter transformer layers
                adapter_len=32,  # Hidden state dimension in adapter layers
            )

        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        # Ensure model is in training mode
        self.model.train()

        # Enable gradient computation for trainable parameters
        for param in self.model.parameters():
            if param.requires_grad:
                param.data = param.data.contiguous()

    # Rest of the training code remains the same...


def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    wandb.login(key="9d47f4bac6fe014143343a3c0551cfb13d61b33b")

    # Example configurations for different methods
    configs = {
        "lora": FineTuningConfig(
            method="lora",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./lora-finetuned",
            batch_size=1,
            num_epochs=1
        ),

        "qlora": FineTuningConfig(
            method="qlora",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./qlora-finetuned",
            batch_size=1,
            num_epochs=1
        ),

        "adalora": FineTuningConfig(
            method="adalora",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./adalora-finetuned",
            batch_size=1,
            num_epochs=1,
            init_r=12,
            target_r=8
        ),

        "prefix": FineTuningConfig(
            method="prefix",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./prefix-finetuned",
            num_virtual_tokens=20
        ),

        "ia3": FineTuningConfig(
            method="ia3",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./ia3-finetuned"
        ),

        "adaption_prompt": FineTuningConfig(
            method="adaption_prompt",
            model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            dataset_name="wikitext",
            output_dir="./adaption-prompt-finetuned",
            num_adaption_prompts=30
        )
    }

    # Choose which method to use
    method = "lora"  # Change this to use different methods
    config = configs[method]

    try:
        trainer = ModelTrainer(config)
        trainer.train()
    except Exception as e:
        print(f"Process failed: {str(e)}")
        raise
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()