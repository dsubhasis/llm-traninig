from datasets import Dataset
import pandas as pd

# If you have a CSV file
df = pd.read_csv("your_data.csv")
dataset = Dataset.from_pandas(df)

# If you have text files
texts = []
with open("your_data.txt", "r") as f:
    texts = f.readlines()
dataset = Dataset.from_dict({"text": texts})

# Save locally
dataset.save_to_disk("local_dataset")

# Then use it in config
config = FineTuningConfig(
    method="lora",
    model_name="meta-llama/Llama-2-7b",
    dataset_name="local_dataset",  # Path to your local dataset
    output_dir="./finetuned-model"
)