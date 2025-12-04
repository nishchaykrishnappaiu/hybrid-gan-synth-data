import argparse
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scripts.utils import detect_column_types
import torch

# ---------- Robust CTGAN import across packages/versions ----------
CTGAN_CLASS = None
SRC = None

try:
    from ctgan import CTGANSynthesizer as _CTGAN
    CTGAN_CLASS = _CTGAN; SRC = "ctgan-direct"
except Exception:
    try:
        from ctgan.synthesizers import CTGANSynthesizer as _CTGAN
        CTGAN_CLASS = _CTGAN; SRC = "ctgan-synthesizers"
    except Exception:
        try:
            from ctgan.synthesizers.ctgan import CTGANSynthesizer as _CTGAN
            CTGAN_CLASS = _CTGAN; SRC = "ctgan-internal"
        except Exception:
            try:
                from sdv.single_table import CTGANSynthesizer as _CTGAN
                CTGAN_CLASS = _CTGAN; SRC = "sdv-single_table"
            except Exception:
                from sdv.tabular import CTGAN as _CTGAN
                CTGAN_CLASS = _CTGAN; SRC = "sdv-tabular"


def _make_model(df, epochs: int, batch_size: int):
    """Instantiate a CTGAN model across ctgan/sdv variants with auto GPU."""
    use_gpu = torch.cuda.is_available()
    dev = "cuda" if use_gpu else "cpu"

    if SRC == "sdv-single_table":
        # SDV SingleTable requires metadata at init
        from sdv.metadata import SingleTableMetadata
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)

        # Try common GPU kwargs across versions
        for kwargs in (
            {"epochs": epochs, "batch_size": batch_size, "device": dev},          # ctgan>=0.7 style
            {"epochs": epochs, "batch_size": batch_size, "enable_gpu": use_gpu},  # deprecated/alt
            {"epochs": epochs, "batch_size": batch_size, "cuda": use_gpu},        # legacy
            {"epochs": epochs, "batch_size": batch_size},                         # last resort
        ):
            try:
                return CTGAN_CLASS(metadata, **kwargs)
            except TypeError:
                continue
        # Fallback without any extras
        return CTGAN_CLASS(metadata, epochs=epochs)

    # Classic ctgan / sdv-tabular path
    for kwargs in (
        {"epochs": epochs, "batch_size": batch_size, "device": dev},
        {"epochs": epochs, "batch_size": batch_size, "enable_gpu": use_gpu},
        {"epochs": epochs, "batch_size": batch_size, "cuda": use_gpu},
        {"epochs": epochs, "batch_size": batch_size},
        {"epochs": epochs},
    ):
        try:
            return CTGAN_CLASS(**kwargs)
        except TypeError:
            continue

    # Absolute fallback
    return CTGAN_CLASS(epochs=epochs)


def _fast_quantile_bin(df: pd.DataFrame, continuous_cols, max_bins: int = 30):
    """Convert numeric columns to categorical via quantile bins to avoid heavy GMM fit."""
    df = df.copy()
    for col in continuous_cols:
        s = df[col].dropna()
        # Skip tiny-unique columns (already handled)
        if s.nunique(dropna=True) < max_bins:
            continue
        try:
            # robust qcut with duplicates='drop'
            df[col] = pd.qcut(df[col], q=max_bins, duplicates='drop')
        except Exception:
            # fallback: uniform bins
            df[col] = pd.cut(df[col], bins=max_bins, duplicates='drop')
        # turn into string categories so downstream treats as categorical
        df[col] = df[col].astype(str)
    return df


def train(input_csv: str,
          model_path: str,
          epochs: int = 30,
          batch_size: int = 256,
          fast_mode: bool = True,
          max_train_rows: int = 10000,
          quantile_bins: int = 30):
    """
    fast_mode=True:
      - downsample to max_train_rows for speed
      - quantile-bin continuous columns to avoid BGMM slowness
    """
    df = pd.read_csv(input_csv)

    # Downsample (speed)
    if fast_mode and len(df) > max_train_rows:
        df = df.sample(n=max_train_rows, random_state=42).reset_index(drop=True)

    # Detect column types
    categorical, continuous = detect_column_types(df)

    # Optional: turn continuous into categorical via bins to skip GMM
    if fast_mode and len(continuous) > 0:
        df = _fast_quantile_bin(df, continuous_cols=continuous, max_bins=quantile_bins)
        # re-detect after binning
        categorical, continuous = detect_column_types(df)

    model = _make_model(df, epochs=epochs, batch_size=batch_size)

    # Fit with various signatures
    fitted = False
    try:
        model.fit(df, categorical)  # older ctgan
        fitted = True
    except TypeError:
        pass
    if not fitted:
        try:
            model.fit(df, categorical_columns=categorical)  # newer ctgan
            fitted = True
        except TypeError:
            pass
    if not fitted:
        model.fit(df)  # sdv variants

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    print(f"🔎 Import source detected: {SRC}")
    print(f"📊 Categorical columns used: {categorical}")
    if fast_mode:
        print("⚡ Fast mode active: downsampled + quantile-binned numerics.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--fast_mode", action="store_true")
    parser.add_argument("--no_fast_mode", action="store_true", help="Disable fast mode")
    parser.add_argument("--max_train_rows", type=int, default=10000)
    parser.add_argument("--quantile_bins", type=int, default=30)
    args = parser.parse_args()

    fast = args.fast_mode or not args.no_fast_mode  # default True
    train(args.input_csv, args.model_path, args.epochs, args.batch_size,
          fast_mode=fast, max_train_rows=args.max_train_rows, quantile_bins=args.quantile_bins)
