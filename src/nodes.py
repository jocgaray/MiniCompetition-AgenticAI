import pandas as pd

from .logic import map_label_to_score
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .schemas import SentimentAnalysis
from .state import AgentState


class PandasTool:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def query(self, query: str):
        # ... your existing logic ...
        return str(eval(query, {"__builtins__": {}}, {"df": self.df}))


def debug_node(state: AgentState, tool: PandasTool):
    # This node can use the tool to check column names or data types
    # without needing the LLM to write the code
    summary = tool.query.invoke("df.columns")
    return {"errors": [f"Debug info: {summary}"]}


def validate_schema_node(state: AgentState):
    print("--- Starting Validation Node ---")

    df = state["raw_data"]
    required = ["property_id", "description", "price_tier"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        return {"errors": [f"Missing columns: {missing}"], "schema_ok": False}

    # Extract the data you need from the DataFrame
    # Assuming you are processing the first row or passing the whole DF
    # If you are iterating, you'd pull the specific row here
    description_text = df["description"].iloc[0]

    print("--- Validation Node Complete ---")

    # Return the status AND the data the next node needs
    return {
        "schema_ok": True,
        "description": description_text,  # <--- This solves your KeyError
    }


async def extract_features_node(state: AgentState, llm):
    description = state["description"]

    # NOW we inject the description into the template
    formatted_prompt = SENTIMENT_ANALYSIS_PROMPT.format(description=description)

    # Now you can send the 'formatted_prompt' to your LLM
    result = await llm.ainvoke(formatted_prompt)

    # 3. Apply your conditional mapping logic
    score = map_label_to_score(result.label)

    # 4. Return the updates to be merged into your AgentState
    return {
        "sentiment_label": result.label.value,  # Store as string
        "sentiment_score": score,  # Store as float
        "reasoning": result.reasoning,
    }
