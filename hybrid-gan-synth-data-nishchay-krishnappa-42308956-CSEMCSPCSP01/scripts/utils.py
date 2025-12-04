import pandas as pd

def detect_column_types(df: pd.DataFrame):
    continuous = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week","fnlwgt"]
    categorical = [col for col in df.columns if col not in continuous]
    return categorical, continuous