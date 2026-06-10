import asyncio
import sys
from functools import partial
from typing import List, Optional, TypedDict, cast

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator

from src.nodes import (
    PandasTool,
    debug_node,
    extract_features_node,
    validate_schema_node,
)
from src.schemas import SentimentAnalysis
from src.state import AgentState

# 1. Instantiate the object, don't just assign a string
llm_model = ChatOllama(model="llama3")

# 2. Now this will work because 'llm_model' is an object, not a string
structured_llm = llm_model.with_structured_output(SentimentAnalysis)


# Define Structured Output for Feature Extraction
class PropertyFeatures(BaseModel):
    is_luxury: bool
    needs_renovation: bool
    sentiment_score: float = Field(..., ge=0.0, le=1.0)

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def sanitize_sentiment(cls, v):
        # If the LLM returns 10.0, we divide by 10 to get 1.0
        # If it returns 100, we divide by 100 to get 1.0
        if isinstance(v, (int, float)):
            if v > 1.0:
                # Dynamically scale based on magnitude
                if v <= 10.0:
                    return v / 10.0
                if v <= 100.0:
                    return v / 100.0
        return v


def get_clean_data(llm_output: object) -> PropertyFeatures:
    """
    Acts as a 'guardian' for your data.
    It checks if the LLM output is already a model or a dict
    and returns a guaranteed PropertyFeatures object.
    """
    if isinstance(llm_output, PropertyFeatures):
        return llm_output

    if isinstance(llm_output, dict):
        # This handles cases where the LLM returns a dictionary
        return PropertyFeatures(**llm_output)

    raise ValueError(f"Unexpected LLM output type: {type(llm_output)}")

def train_and_predict_node(state: AgentState):
    """Deterministic ML classification."""
    print("--- Starting Prediction Node ---")
    from sklearn.ensemble import RandomForestClassifier

    # Merge extracted features with original tabular metrics
    X = pd.concat(
        [state["raw_data"].drop(columns=["description"]), state["clean_features"]],
        axis=1,
    )
    y = state["raw_data"]["price_tier"]

    model = RandomForestClassifier().fit(X, y)
    print("--- Prediction Node Complete ---")
    return {"predictions": model.predict(X).tolist()}


def build_workflow(df: pd.DataFrame):
    # Create the tool
    pandas_tool = PandasTool(df)

    # Create a version of the node that already has the tool attached
    configured_debug_node = partial(debug_node, tool=pandas_tool)

    async def extract_features_wrapper(state: AgentState):
        # This explicitly calls your function with the injected model
        return await extract_features_node(state, llm=structured_llm)

    # Add that to the workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("validate", validate_schema_node)
    workflow.add_node("debug", configured_debug_node)
    workflow.add_node("extract", extract_features_wrapper)
    workflow.add_node("predict", train_and_predict_node)
    # Conditional Logic
    workflow.set_entry_point("validate")
    workflow.add_conditional_edges(
        "validate", lambda s: "extract" if s["schema_ok"] else "debug"
    )
    workflow.add_edge("debug", END)
    workflow.add_edge("extract", "predict")
    workflow.add_edge("predict", END)

    return workflow.compile()
