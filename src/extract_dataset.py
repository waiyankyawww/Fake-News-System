# notebooks/1_data_load_and_prep.py
import re
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# CONFIG
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "liar"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42

# HELPERS
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r'http\S+|www\.\S+', '', s)   # remove URLs
    s = re.sub(r'@\w+', '', s)               # remove mentions
    s = re.sub(r'<.*?>', '', s)              # strip simple html
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def pick_text_column(df):
    # Try common text column names first
    for col in ['text','article','content','title','body','description','statement','claim']:
        if col in df.columns:
            return col
    # Else fallback to first object/string dtype column
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col].dtype):
            return col
    return None

# LIAR dataset columns (based on your example and LIAR README)
liar_cols = [
    'id1', 'id2', 'label', 'statement', 'subject', 'speaker', 'speaker_job',
    'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
    'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context', 'extra_text'
]

print("Loading LIAR dataset from local files...")

liar_train = pd.read_csv(RAW_DIR / "train.tsv", sep='\t', header=None, names=liar_cols, on_bad_lines='skip')
liar_val   = pd.read_csv(RAW_DIR / "val.tsv", sep='\t', header=None, names=liar_cols, on_bad_lines='skip')
liar_test  = pd.read_csv(RAW_DIR / "test.tsv", sep='\t', header=None, names=liar_cols, on_bad_lines='skip')

print(f"Train samples: {len(liar_train)}, Validation samples: {len(liar_val)}, Test samples: {len(liar_test)}")

# Map original LIAR labels to binary: 1 = fake, 0 = true-ish
liar_map = {'pants-fire':1, 'false':1, 'barely-true':1, 'half-true':0, 'mostly-true':0, 'true':0}

def liar_to_binary_label(x):
    return liar_map.get(x, None)

frames = []
for split_name, df in [('train', liar_train), ('validation', liar_val), ('test', liar_test)]:
    df = df.copy()
    df['text'] = df['statement'].astype(str).apply(clean_text)
    df['label'] = df['label'].apply(liar_to_binary_label)
    df['source'] = 'liar'
    df['split'] = split_name
    frames.append(df[['text','label','source','split']])

liar_df = pd.concat(frames, ignore_index=True)
liar_df = liar_df[liar_df['label'].notna()].copy()
liar_df['label'] = liar_df['label'].astype(int)

liar_df.to_csv(PROCESSED_DIR / "liar_combined.csv", index=False)
print("Saved LIAR processed dataset to:", PROCESSED_DIR / "liar_combined.csv")


# 2) FakeNewsNet (simple CSV path) — clone repo manually first
fakenews_csv_dir = DATA_DIR / "FakeNewsNet" / "dataset"
if not fakenews_csv_dir.exists():
    print("FakeNewsNet dataset folder not found:", fakenews_csv_dir)
    print("Please clone FakeNewsNet into data/FakeNewsNet using:")
    print("  git clone https://github.com/KaiDMML/FakeNewsNet.git data/FakeNewsNet")
else:
    csvs = list(fakenews_csv_dir.glob("*.csv"))
    print("Found FakeNewsNet CSVs:", [c.name for c in csvs])
    fframes = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        txtcol = pick_text_column(df)
        if txtcol is None:
            print("Skipping", csv, "- no text column found.")
            continue
        df = df.copy()
        df['text'] = df[txtcol].astype(str).apply(clean_text)
        label = 1 if 'fake' in csv.name.lower() else 0
        df['label'] = label
        df['source'] = csv.stem
        fframes.append(df[['text','label','source']])
    if fframes:
        fakenews_df = pd.concat(fframes, ignore_index=True)
        fakenews_df.to_csv(PROCESSED_DIR / "fakenewsnet_combined.csv", index=False)
        print("Saved FakeNewsNet processed:", PROCESSED_DIR / "fakenewsnet_combined.csv")
    else:
        print("No usable FakeNewsNet CSVs found; run the repo downloader if you need raw articles/tweets.")

# 3) Combine (optional)
combined_paths = [PROCESSED_DIR / "liar_combined.csv"]
if (PROCESSED_DIR / "fakenewsnet_combined.csv").exists():
    combined_paths.append(PROCESSED_DIR / "fakenewsnet_combined.csv")
dfs = [pd.read_csv(p) for p in combined_paths]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv(PROCESSED_DIR / "combined_all.csv", index=False)
print("Saved combined dataset:", PROCESSED_DIR / "combined_all.csv")
print("Step 1 done. Processed CSVs are in:", PROCESSED_DIR)
