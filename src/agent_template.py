import asyncio
import sys
from typing import List, Optional, TypedDict, cast

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator
from tqdm.asyncio import tqdm  # Import the async version

lmm_model = "llama3:latest"  # "qwen3.5:9b"


# 1. Define the State
class AgentState(TypedDict):
    raw_data: pd.DataFrame
    clean_features: pd.DataFrame
    predictions: Optional[List[int]]
    errors: List[str]
    schema_ok: bool


# 2. Define the Tool Class
class PandasTool:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def query(self, query: str):
        # ... your existing logic ...
        return str(eval(query, {"__builtins__": {}}, {"df": self.df}))


def debug_node(state: AgentState):
    # This node can use the tool to check column names or data types
    # without needing the LLM to write the code
    summary = pandas_tool.query.invoke("df.columns")
    return {"errors": [f"Debug info: {summary}"]}


# 3. Define Structured Output for Feature Extraction
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
                if v <= 10.0: return v / 10.0
                if v <= 100.0: return v / 100.0
        return v


# 4. Nodes (The "Defensive" Logic)
def validate_schema_node(state: AgentState):
    """Checks if required columns exist before proceeding."""

    print("--- Starting Validation Node ---")

    df = state["raw_data"]
    required = ["property_id", "description", "price_tier"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return {"errors": [f"Missing columns: {missing}"], "schema_ok": False}

    print("--- Validation Node Complete ---")
    return {"schema_ok": True}


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


async def extract_features_node(state: AgentState):
    print(f"--- Starting Extraction Node ({len(state['raw_data'])} items) ---")
    llm = ChatOllama(model=lmm_model)
    structured_llm = llm.with_structured_output(PropertyFeatures)
    descriptions = state["raw_data"]["description"].tolist()
    total = len(descriptions)
    semaphore = asyncio.Semaphore(3)

    async def invoke_with_progress(desc):
        async with semaphore:
            return await structured_llm.ainvoke(f"Extract features from: {desc}")

    # 1. Create the tasks
    tasks = [invoke_with_progress(desc) for desc in descriptions]

    # 2. Use as_completed to update the bar as each task finishes
    features = []
    # tqdm wraps the generator provided by as_completed
    with tqdm(total=total, desc="Extracting features", file=sys.stdout) as pbar:
        for finished_task in asyncio.as_completed(tasks):
            try:
                raw_result = await asyncio.wait_for(finished_task, timeout=120.0)

                # Access the attributes directly to normalize before model_dump()
                # If the LLM returned 5.0, we divide it by 5 to bring it to a 0.0-1.0 scale
                if raw_result.sentiment_score > 1.0:
                    raw_result.sentiment_score /= 5.0

                features.append(raw_result.model_dump())

            except Exception as e:
                # If it's a Pydantic validation error, you can potentially
                # log it and skip, or use a default
                print(f"Skipping record due to error: {e}")
                features.append(
                    {
                        "is_luxury": False,
                        "needs_renovation": False,
                        "sentiment_score": 0.0,
                    }
                )
            pbar.update(1)

    print("--- Extraction Node Complete ---")
    return {"clean_features": pd.DataFrame(features)}


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
    pandas_tool = PandasTool(df)

    # Define debug_node here so it captures 'pandas_tool'
    def debug_node(state: AgentState):
        summary = pandas_tool.query("df.columns")
        return {"errors": state["errors"] + [f"Debug info: {summary}"]}

    # Define other nodes...
    # (validate_schema_node, extract_features_node, etc.)

    workflow = StateGraph(AgentState)
    workflow.add_node("validate", validate_schema_node)
    workflow.add_node("debug", debug_node)  # Add the new node
    workflow.add_node("extract", extract_features_node)
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
