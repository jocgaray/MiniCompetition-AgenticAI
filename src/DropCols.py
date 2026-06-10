import pandas as pd


def drop_dataframe_columns(df: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    return df.drop(columns=columns_to_drop, errors="ignore")
