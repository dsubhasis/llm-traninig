import torch
from transformers import LlamaTokenizer
from peft import PeftModel
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import random
from collections import defaultdict


@dataclass
class TestConfig:
    """Configuration for testing"""
    model_path: str
    test_data_path: str
    output_dir: str
    is_peft_model: bool = False
    base_model_path: Optional[str] = None
    batch_size: int = 16
    num_samples: Optional[int] = None


class ComprehensiveIntentTester:
    def __init__(self, config: TestConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def setup_model(self):
        """Initialize model and tokenizer"""
        if self.config.is_peft_model:
            from transformers import LlamaForSequenceClassification
            base_model = LlamaForSequenceClassification.from_pretrained(
                self.config.base_model_path,
                num_labels=2  # Will be overridden by PEFT config
            )
            self.model = PeftModel.from_pretrained(base_model, self.config.model_path)
        else:
            from transformers import LlamaForSequenceClassification
            self.model = LlamaForSequenceClassification.from_pretrained(self.config.model_path)

        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.config.base_model_path if self.config.is_peft_model else self.config.model_path
        )

        self.model.to(self.device)
        self.model.eval()

    def load_test_data(self) -> List[Dict]:
        """Load test data"""
        with open(self.config.test_data_path, 'r') as f:
            data = [json.loads(line) for line in f]
        if self.config.num_samples:
            data = data[:self.config.num_samples]
        return data

    def basic_testing(self, test_data: List[Dict]) -> Dict:
        """Perform basic testing metrics"""
        predictions = []
        true_labels = []
        raw_outputs = []

        for item in tqdm(test_data, desc="Basic testing"):
            # Get prediction
            pred, output = self.predict_single(item['utterance'])
            predictions.append(pred)
            true_labels.append(item['intent'])
            raw_outputs.append(output)

        # Calculate basic metrics
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'per_class_report': classification_report(true_labels, predictions, output_dict=True),
            'confusion_matrix': confusion_matrix(true_labels, predictions).tolist(),
            'predictions': list(zip(predictions, true_labels))
        }

        # Add precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        return metrics, raw_outputs

    def advanced_testing(self, test_data: List[Dict]) -> Dict:
        """Perform advanced testing"""
        # Create challenging test cases
        challenge_cases = self.create_challenge_cases(test_data)

        advanced_metrics = {
            'robustness': self.test_robustness(challenge_cases),
            'confidence_analysis': self.analyze_confidence(test_data),
            'error_analysis': self.analyze_errors(test_data),
            'challenge_performance': self.evaluate_challenge_cases(challenge_cases)
        }

        return advanced_metrics

    def create_challenge_cases(self, data: List[Dict]) -> List[Dict]:
        """Create challenging test cases"""
        challenge_cases = []

        for item in data:
            # Create variations
            variations = [
                self.create_noisy_input(item),
                self.create_edge_case(item),
                self.create_ambiguous_case(item)
            ]
            challenge_cases.extend(variations)

        return challenge_cases

    def create_noisy_input(self, item: Dict) -> Dict:
        """Add noise to input"""
        text = item['utterance']
        words = text.split()

        # Random modifications
        if len(words) > 3:
            # Randomly remove a word
            del words[random.randint(0, len(words) - 1)]

        return {
            'utterance': ' '.join(words),
            'intent': item['intent'],
            'type': 'noisy'
        }

    def create_edge_case(self, item: Dict) -> Dict:
        """Create edge cases"""
        text = item['utterance']
        variations = [
            text.upper(),
            text.lower(),
            ' '.join(reversed(text.split())),
            text.replace(' ', '')
        ]

        return {
            'utterance': random.choice(variations),
            'intent': item['intent'],
            'type': 'edge'
        }

    def create_ambiguous_case(self, item: Dict) -> Dict:
        """Create potentially ambiguous cases"""
        text = item['utterance']
        words = text.split()

        if len(words) > 4:
            # Mix parts of the sentence
            mid = len(words) // 2
            words = words[mid:] + words[:mid]

        return {
            'utterance': ' '.join(words),
            'intent': item['intent'],
            'type': 'ambiguous'
        }

    def test_robustness(self, challenge_cases: List[Dict]) -> Dict:
        """Test model robustness"""
        results = defaultdict(list)

        for case in challenge_cases:
            pred, output = self.predict_single(case['utterance'])
            correct = pred == case['intent']
            confidence = max(output)

            results[case['type']].append({
                'correct': correct,
                'confidence': confidence
            })

        # Calculate metrics per case type
        robustness_metrics = {}
        for case_type, cases in results.items():
            robustness_metrics[case_type] = {
                'accuracy': np.mean([c['correct'] for c in cases]),
                'avg_confidence': np.mean([c['confidence'] for c in cases]),
                'confidence_std': np.std([c['confidence'] for c in cases])
            }

        return robustness_metrics

    def analyze_confidence(self, test_data: List[Dict]) -> Dict:
        """Analyze model confidence"""
        confidences = []
        correct_predictions = []

        for item in test_data:
            pred, output = self.predict_single(item['utterance'])
            confidence = max(output)
            correct = pred == item['intent']

            confidences.append(confidence)
            correct_predictions.append(correct)

        return {
            'avg_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_correct': np.mean([conf for conf, corr in zip(confidences, correct_predictions) if corr]),
            'confidence_incorrect': np.mean([conf for conf, corr in zip(confidences, correct_predictions) if not corr]),
            'high_confidence_accuracy': np.mean(
                [corr for conf, corr in zip(confidences, correct_predictions) if conf > 0.9])
        }

    def analyze_errors(self, test_data: List[Dict]) -> Dict:
        """Analyze error patterns"""
        error_analysis = defaultdict(list)

        for item in test_data:
            pred, output = self.predict_single(item['utterance'])
            if pred != item['intent']:
                error_analysis['misclassifications'].append({
                    'text': item['utterance'],
                    'true_intent': item['intent'],
                    'predicted_intent': pred,
                    'confidence': max(output)
                })

        return {
            'error_cases': error_analysis['misclassifications'],
            'error_rate': len(error_analysis['misclassifications']) / len(test_data),
            'common_errors': self.get_common_error_patterns(error_analysis['misclassifications'])
        }

    def get_common_error_patterns(self, error_cases: List[Dict]) -> Dict:
        """Identify common error patterns"""
        patterns = defaultdict(int)

        for case in error_cases:
            pattern = f"{case['true_intent']} → {case['predicted_intent']}"
            patterns[pattern] += 1

        return dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:5])

    def predict_single(self, text: str) -> Tuple[str, List[float]]:
        """Make prediction for single text"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()

        return self.model.config.id2label[predicted_class], probs[0].tolist()

    def generate_report(self, basic_metrics: Dict, advanced_metrics: Dict):
        """Generate comprehensive test report"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump({
                'basic_metrics': basic_metrics,
                'advanced_metrics': advanced_metrics
            }, f, indent=2)

        # Generate visualizations
        self.plot_confusion_matrix(basic_metrics['confusion_matrix'], output_dir)
        self.plot_confidence_distribution(advanced_metrics['confidence_analysis'], output_dir)

        # Generate summary report
        self.generate_summary_report(basic_metrics, advanced_metrics, output_dir)

    def plot_confusion_matrix(self, conf_matrix: List[List[int]], output_dir: Path):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig(output_dir / 'confusion_matrix.png')
        plt.close()

    def plot_confidence_distribution(self, confidence_analysis: Dict, output_dir: Path):
        """Plot confidence distribution"""
        plt.figure(figsize=(10, 6))
        plt.hist(confidence_analysis['confidences'], bins=20)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.savefig(output_dir / 'confidence_distribution.png')
        plt.close()

    def generate_summary_report(self, basic_metrics: Dict, advanced_metrics: Dict, output_dir: Path):
        """Generate summary report"""
        summary = f"""Intent Classification Model Test Report

Basic Metrics:
-------------
Accuracy: {basic_metrics['accuracy']:.3f}
Precision: {basic_metrics['precision']:.3f}
Recall: {basic_metrics['recall']:.3f}
F1 Score: {basic_metrics['f1']:.3f}

Advanced Metrics:
----------------
Robustness Score: {np.mean([m['accuracy'] for m in advanced_metrics['robustness'].values()]):.3f}
Average Confidence: {advanced_metrics['confidence_analysis']['avg_confidence']:.3f}
Error Rate: {advanced_metrics['error_analysis']['error_rate']:.3f}

Common Error Patterns:
--------------------
{json.dumps(advanced_metrics['error_analysis']['common_errors'], indent=2)}
"""

        with open(output_dir / 'summary_report.txt', 'w') as f:
            f.write(summary)


def main():
    # Example usage
    config = TestConfig(
        model_path="path/to/your/model",
        test_data_path="path/to/test/data.jsonl",
        output_dir="test_results",
        is_peft_model=True,
        base_model_path="meta-llama/Llama-2-7b"
    )

    tester = ComprehensiveIntentTester(config)

    # Load test data
    test_data = tester.load_test_data()

    # Run basic tests
    basic_metrics, raw_outputs = tester.basic_testing(test_data)

    # Run advanced tests
    advanced_metrics = tester.advanced_testing(test_data)

    # Generate report
    tester.generate_report(basic_metrics, advanced_metrics)


if __name__ == "__main__":
    main()