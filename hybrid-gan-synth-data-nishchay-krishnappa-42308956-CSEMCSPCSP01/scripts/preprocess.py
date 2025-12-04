import argparse
import pandas as pd
from pathlib import Path
from scripts.utils import detect_column_types

def preprocess(input_csv: str, output_csv: str, dropna: bool = True):
    df = pd.read_csv(input_csv)
    # Basic cleaning
    if dropna:
        df = df.dropna().reset_index(drop=True)
    df = df.drop_duplicates().reset_index(drop=True)

    # Optional: convert obvious identifiers to string to avoid numeric leakage
    for col in df.columns:
        if "id" in col.lower():
            df[col] = df[col].astype(str)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    cat, con = detect_column_types(df)
    print("Saved cleaned data:", output_csv)
    print("Detected categorical:", cat)
    print("Detected continuous:", con)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--keep_na", action="store_true", help="Keep NA values instead of dropping")
    args = parser.parse_args()
    preprocess(args.input_csv, args.output_csv, dropna=not args.keep_na)
