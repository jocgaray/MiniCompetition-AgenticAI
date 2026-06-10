import asyncio
import time

import pandas as pd
from langchain_core.tools import tool
from sklearn.ensemble import RandomForestClassifier

from .DropCols import drop_dataframe_columns
from .logic import map_label_to_score
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .schemas import PropertyFeatures, SentimentAnalysis
from .state import AgentState
from .zipcode_dollars import get_price_from_lat_long


def make_data_tools(df: pd.DataFrame, structured_llm=None):
    @tool
    def get_column_info() -> str:
        """Returns the column names and data types of the current dataframe."""
        return str(df.dtypes)

    @tool
    def preview_data(n: int = 5) -> str:
        """Returns the first n rows of the dataframe."""
        return df.head(n).to_string()

    @tool
    def query(query_str: str) -> str:
        """Run a pandas expression on the dataframe for inspection. Use 'df' as table name."""
        try:
            allowed_names = {"df": df, "pd": pd}
            result = eval(query_str, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"Query error: {e}"

    @tool
    def drop_columns(columns: str) -> str:
        """Drop specified columns. Provide as comma-separated list. Example: 'neighbourhood_group, neighbourhood'"""
        col_list = [c.strip() for c in columns.split(",")]
        df.drop(columns=col_list, inplace=True, errors="ignore")
        return f"Dropped columns: {col_list}"

    @tool
    def add_region_price() -> str:
        """Convert latitude/longitude to region_price column. Drops coordinate columns afterward."""
        df["region_price"] = df.apply(
            lambda row: get_price_from_lat_long(
                float(row.get("latitude", 0)), float(row.get("longitude", 0))
            ),
            axis=1,
        )
        df.drop(columns=["latitude", "longitude"], inplace=True, errors="ignore")
        return f"Added region_price for {len(df)} rows"

    @tool
    def analyze_descriptions() -> str:
        """Run LLM sentiment analysis on all property descriptions. Adds sentiment_score column."""
        if structured_llm is None:
            return "Error: LLM not configured for sentiment analysis"

        scores = []
        for idx, row in df.iterrows():
            desc = row.get("description", "")
            if not desc or not isinstance(desc, str) or len(desc.strip()) == 0:
                scores.append(2.5)
                continue
            try:
                prompt = SENTIMENT_ANALYSIS_PROMPT.format(description=desc)
                result = structured_llm.invoke(prompt)
                score = map_label_to_score(result.label)
                scores.append(score)
            except Exception:
                scores.append(2.5)

        df["sentiment_score"] = scores
        avg = sum(scores) / len(scores) if scores else 0
        return f"Analyzed {len(scores)} descriptions, mean score: {avg:.2f}"

    return [get_column_info, preview_data, query, drop_columns, add_region_price, analyze_descriptions]


def get_clean_data(data: dict) -> PropertyFeatures:
    return PropertyFeatures(**data)


def validate_schema_node(state: AgentState) -> dict:
    print("--- Validating Schema ---")
    df = state["raw_data"]
    required = ["property_id", "description", "room_type", "minimum_nights"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return {"errors": [f"Missing columns: {missing}"], "schema_ok": False}

    print("--- Schema OK ---")
    return {"schema_ok": True, "errors": []}


def train_and_predict_node(state: AgentState) -> dict:
    print("--- Training & Predicting ---")
    df = state["raw_data"]

    feature_cols = [
        "sentiment_score",
        "region_price",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]

    available = [c for c in feature_cols if c in df.columns]

    if "price_tier" in df.columns and available:
        train_df = df.dropna(subset=available + ["price_tier"])
        if len(train_df) == 0:
            return {"predictions": [], "errors": ["No training data after dropping NA"]}

        X = train_df[available]
        y = train_df["price_tier"]

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        pred_df = df[available].fillna(0)
        pred_df = pred_df.reindex(columns=X.columns, fill_value=0)
        predictions = model.predict(pred_df).tolist()
    else:
        predictions = []

    print(f"--- Predictions: {len(predictions)} rows ---")
    return {"predictions": predictions}


def debug_node(state: AgentState) -> dict:
    df = state.get("raw_data")
    if df is not None:
        return {"errors": [f"Debug - columns: {list(df.columns)}"]}
    return {"errors": ["No data available"]}
