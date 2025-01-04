import json
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from typing import Optional
import os

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer

# Download required NLTK data
nltk.download('punkt')


@dataclass
class EvalConfig:
    """Configuration for model evaluation"""
    model_path: str
    test_dataset_path: str
    batch_size: int = 8
    max_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_samples: Optional[int] = None  # For partial evaluation
    is_peft_model: bool = False  # Whether the model is a PEFT model
    base_model_path: Optional[str] = None  # Required if is_peft_model is True


class ModelEvaluator:
    """Comprehensive model evaluation framework"""

    def __init__(self, config: EvalConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.setup_model()
        self.setup_tokenizer()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.bleu = BLEU()

    def setup_model(self):
        """Initialize the model"""
        if self.config.is_peft_model:
            if not self.config.base_model_path:
                raise ValueError("base_model_path is required for PEFT models")

            base_model = LlamaForCausalLM.from_pretrained(
                self.config.base_model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )
            self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
        else:
            self.model = LlamaForCausalLM.from_pretrained(
                self.config.model_path,
                device_map="auto",
                torch_dtype=torch.float16
            )

        self.model.eval()

    def setup_tokenizer(self):
        """Initialize the tokenizer"""
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.base_model_path if self.config.is_peft_model else self.config.model_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_test_data(self) -> List[Dict[str, str]]:
        """Load test dataset"""
        if self.config.test_dataset_path.endswith('.json'):
            with open(self.config.test_dataset_path, 'r') as f:
                data = json.load(f)
        else:
            dataset = load_dataset(self.config.test_dataset_path)
            data = dataset['test']

        if self.config.num_samples:
            data = data[:self.config.num_samples]
        return data

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a given prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,
                num_beams=4,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.replace(prompt, "").strip()

    def calculate_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Calculate various evaluation metrics"""
        metrics = {}

        # ROUGE scores
        rouge_scores = {
            'rouge1': [],
            'rouge2': [],
            'rougeL': []
        }
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(pred, ref)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        for key in rouge_scores:
            metrics[f'{key}_f1'] = np.mean(rouge_scores[key])

        # BLEU score
        bleu_score = self.bleu.corpus_score(predictions, [references])
        metrics['bleu'] = bleu_score.score

        # Token-level accuracy
        pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
        ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]

        token_acc = []
        for pred, ref in zip(pred_tokens, ref_tokens):
            correct = sum(p == r for p, r in zip(pred[:len(ref)], ref))
            total = len(ref)
            token_acc.append(correct / total if total > 0 else 0)

        metrics['token_accuracy'] = np.mean(token_acc)

        return metrics

    def evaluate_latency(self, sample_prompts: List[str]) -> Dict[str, float]:
        """Evaluate model latency"""
        latencies = []

        for prompt in tqdm(sample_prompts, desc="Evaluating latency"):
            start_time = time.time()
            _ = self.generate_response(prompt)
            latencies.append(time.time() - start_time)

        return {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'p90_latency': np.percentile(latencies, 90),
            'p95_latency': np.percentile(latencies, 95),
            'std_latency': np.std(latencies)
        }

    def evaluate_memory_usage(self) -> Dict[str, float]:
        """Evaluate model memory usage"""
        torch.cuda.reset_peak_memory_stats()
        sample_prompt = "This is a test prompt to measure memory usage."
        _ = self.generate_response(sample_prompt)

        memory_stats = {
            'peak_memory_mb': torch.cuda.max_memory_allocated() / 1024 ** 2,
            'current_memory_mb': torch.cuda.memory_allocated() / 1024 ** 2,
            'peak_memory_cached_mb': torch.cuda.max_memory_cached() / 1024 ** 2
        }
        return memory_stats

    def evaluate_robustness(self, sample_prompts: List[str]) -> Dict[str, List[float]]:
        """Evaluate model robustness to input variations"""

        def create_variations(text: str) -> List[str]:
            variations = [
                text,  # original
                text.lower(),  # lowercase
                text.upper(),  # uppercase
                text.replace('.', ''),  # remove punctuation
                text + ' ' * 10,  # extra spaces
                ''.join(c for c in text if c.isalnum() or c.isspace())  # alphanumeric only
            ]
            return variations

        robustness_scores = []
        for prompt in tqdm(sample_prompts, desc="Evaluating robustness"):
            variations = create_variations(prompt)
            responses = [self.generate_response(var) for var in variations]

            # Calculate similarity between responses
            base_response = responses[0]
            similarities = [
                self.rouge_scorer.score(resp, base_response)['rougeL'].fmeasure
                for resp in responses[1:]
            ]
            robustness_scores.append(np.mean(similarities))

        return {
            'robustness_score': np.mean(robustness_scores),
            'robustness_std': np.std(robustness_scores)
        }

    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        test_data = self.load_test_data()
        predictions = []
        references = []

        # Generate predictions
        for item in tqdm(test_data, desc="Generating predictions"):
            prompt = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:"
            pred = self.generate_response(prompt)
            predictions.append(pred)
            references.append(item['output'])

        # Calculate all metrics
        metrics = self.calculate_metrics(predictions, references)

        # Evaluate latency
        sample_prompts = [item['instruction'] + ' ' + item['input'] for item in test_data[:10]]
        latency_metrics = self.evaluate_latency(sample_prompts)

        # Evaluate memory usage
        memory_metrics = self.evaluate_memory_usage()

        # Evaluate robustness
        robustness_metrics = self.evaluate_robustness(sample_prompts)

        # Combine all metrics
        report = {
            'performance_metrics': metrics,
            'latency_metrics': latency_metrics,
            'memory_metrics': memory_metrics,
            'robustness_metrics': robustness_metrics,
            'sample_predictions': list(zip(predictions[:5], references[:5]))
        }

        return report

    def plot_metrics(self, report: Dict[str, Any], output_path: str):
        """Generate visualization of evaluation metrics"""
        # Performance metrics plot
        plt.figure(figsize=(12, 6))
        metrics_df = pd.DataFrame([report['performance_metrics']]).melt()
        sns.barplot(data=metrics_df, x='variable', y='value')
        plt.title('Performance Metrics')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path}/performance_metrics.png")
        plt.close()

        # Latency distribution plot
        plt.figure(figsize=(8, 6))
        latency_df = pd.DataFrame([report['latency_metrics']]).melt()
        sns.barplot(data=latency_df, x='variable', y='value')
        plt.title('Latency Metrics (seconds)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_path}/latency_metrics.png")
        plt.close()


def main():
    # Example usage
    config = EvalConfig(
        model_path="path/to/your/model",
        test_dataset_path="path/to/test/data.json",
        is_peft_model=True,
        base_model_path="meta-llama/Llama-2-7b"
    )

    evaluator = ModelEvaluator(config)
    report = evaluator.generate_evaluation_report()

    # Save report
    output_path = "evaluation_results"
    os.makedirs(output_path, exist_ok=True)

    with open(f"{output_path}/evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Generate plots
    evaluator.plot_metrics(report, output_path)


if __name__ == "__main__":
    main()