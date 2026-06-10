import pandas as pd
from langchain_core.tools import tool

@tool
def drop_dataframe_columns(file_path: str, columns_to_drop: list):
    """
    Drops specified columns from a CSV file and saves the cleaned version.
    Use this tool when you need to remove unnecessary data columns.
    """
    df = pd.read_csv(file_path)
    df = df.drop(columns=columns_to_drop, errors="ignore")
    df.to_csv(file_path, index=False)
    return f"Successfully dropped {columns_to_drop} from {file_path}."
