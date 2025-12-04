import os
import pandas as pd
from pathlib import Path
from scripts.preprocess import preprocess
from scripts.train_ctgan import train
from scripts.generate import generate
from scripts.evaluate import evaluate
from scripts.utils import detect_column_types
import torch

REAL_INPUT = "data/real/input.csv"
CLEANED_OUTPUT = "data/real/clean.csv"
MODEL_PATH = "models/ctgan_model.pkl"
SYNTHETIC_OUTPUT = "data/synthetic/samples.csv"
PLOTS_DIR = "plots"

EPOCHS = 50
BATCH_SIZE = 500

def checking_for_gpu():
    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())


def main():
    print("\n🔹 Step 1: Preprocessing dataset...")
    preprocess(REAL_INPUT, CLEANED_OUTPUT, dropna=True)

    print("\n🔹 Step 2: Training CTGAN model...")
    train(CLEANED_OUTPUT, MODEL_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("\n🔹 Step 3: Generating synthetic samples...")
    n_rows = len(pd.read_csv(CLEANED_OUTPUT))
    generate(MODEL_PATH, n_rows, SYNTHETIC_OUTPUT)

    print("\n🔹 Step 4: Evaluating similarity between real and synthetic data...")
    evaluate(CLEANED_OUTPUT, SYNTHETIC_OUTPUT, PLOTS_DIR)

    print("\n\u2705 All steps completed successfully!")
    print(f"Results saved in:\n - Synthetic data: {SYNTHETIC_OUTPUT}\n - Plots: {PLOTS_DIR}\n - Model: {MODEL_PATH}")


if __name__ == "__main__":

    for folder in ["data/real", "data/synthetic", "models", "plots"]:
        Path(folder).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(REAL_INPUT):
        print(f"\u274C Error: Please place your dataset at {REAL_INPUT}")
    else:
        print("*** Checking for Gpu if not avaiable the programm automatically use cpu***")
        checking_for_gpu()
        main()
