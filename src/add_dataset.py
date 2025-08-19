import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


# Paths
fake_csv = DATA_DIR / "raw" / "fake_and_real_news" / "Fake.csv"
real_csv = DATA_DIR / "raw" / "fake_and_real_news" / "True.csv"


# Read and map columns
def load_extra_csv(path, label_value, source_name="extra"):
    df = pd.read_csv(path, quotechar='"', skipinitialspace=True, encoding='utf-8')
    df_mapped = pd.DataFrame({
        'text': df['text'],
        'label': label_value,           # 1=fake, 0=real
        'source': source_name,
        'split': 'train'               # Assign all to train; optionally split later
    })
    return df_mapped

fake_df = load_extra_csv(fake_csv, label_value=1, source_name="fakecsv")
real_df = load_extra_csv(real_csv, label_value=0, source_name="realcsv")

# Read existing combined_all.csv
combined_path = PROCESSED_DIR / "combined_all.csv"
combined_df = pd.read_csv(combined_path)

# Concatenate
final_df = pd.concat([combined_df, fake_df, real_df], ignore_index=True)

# Save
final_df.to_csv(PROCESSED_DIR / "combined_all_with_extra.csv", index=False)
print("Added fake.csv and real.csv. Total rows now:", len(final_df))