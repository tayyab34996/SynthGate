from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent
file_path = project_dir / "data" / "raw" / "DataCoSupplyChainDataset.csv"
sample_path = project_dir / "data" / "raw" / "DataCo_sample.csv"

# Read the public source file using its original text encoding.
df = pd.read_csv(file_path, encoding="latin1")

# Create a deterministic five-percent sample for local experiments.
sample_df = df.sample(frac=0.05, random_state=42)

# Write the sample locally without adding it to Git.
sample_df.to_csv(
    sample_path,
    index=False,
    encoding="utf-8",
)

print(f"Original rows: {len(df)} | Sampled rows: {len(sample_df)}")
