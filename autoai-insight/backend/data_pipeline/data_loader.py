import pandas as pd
import polars as pl
from utils import get_file_extension

def load_data(file_path, engine='pandas'):
    ext= get_file_extension(file_path)

    if engine == 'pandas':
        if ext == '.csv':
            return pd.read_csv(file_path)
        elif ext == '.parquet':
            return pd.read_parquet(file_path)
        elif ext == '.json':
            return pd.read_json(file_path)
        
    elif engine == 'polars':
        if ext == '.csv':
            return pl.read_csv(file_path).to_pandas()
        elif ext == '.parquet':
            return pl.read_parquet(file_path).to_pandas()
        elif ext == '.json':
            return pl.read_json(file_path).to_pandas()
    
    return ValueError(f"Unsupported file format: {ext}")


