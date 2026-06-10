from langchain_core.language_models.llms import LLM
import pandas as pd
from langchain_core.tools import tool
from typing import Optional

from .schemas import  PropertyFeatures
from .zipcode_dollars import transform_coordinates_to_price
from .logic import map_label_to_score
from .prompts import SENTIMENT_ANALYSIS_PROMPT
from .state import AgentState

@tool
def run_data_cleaning(state: AgentState):
    """Cleans the raw data. Call this when you receive new data."""
    return data_cleaning_node(state)

@tool
def run_feature_extraction(state: AgentState, structured_llm: LLM):
    """Extracts sentiment and luxury features from descriptions."""
    return extract_features_node(state, llm=structured_llm)

class PandasTool:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    @tool
    def get_column_info(self):
        """Returns the column names and data types of the current dataframe."""
        return str(self.df.dtypes)

    @tool
    def preview_data(self, n: int = 5):
        """Returns the first n rows of the dataframe."""
        return self.df.head(n).to_string()

    @tool
    def query(self, query: str):
        # ... your existing logic ...
        return str(eval(query, {"__builtins__": {}}, {"df": self.df}))

def train_and_predict_node(state: AgentState):
    """Deterministic ML classification."""
    print("--- Starting Prediction Node ---")
    from sklearn.ensemble import RandomForestClassifier

    # Ensure clean_features is a DataFrame
    clean_df = pd.DataFrame(state["clean_features"]) if not isinstance(state["clean_features"], pd.DataFrame) else state["clean_features"]
    
    # Now concatenate
    X = pd.concat(
        [state["raw_data"].drop(columns=["description"]), clean_df],
        axis=1
    )

    # Merge extracted features with original tabular metrics
    y = state["raw_data"]["price_tier"]

    model = RandomForestClassifier().fit(X, y)
    print("--- Prediction Node Complete ---")
    return {"predictions": model.predict(X).tolist()}

def pre_processing_node(state: AgentState):
    # This runs BEFORE the agent ever sees the data
    if 'lat' in state['raw_data'].columns:
        state['raw_data'] = transform_coordinates_to_price.invoke(state['raw_data'])
    return state

# The Agent Node
def data_processor_node(state):
    # This node provides the agent with the tool
    # The agent determines if it should call 'transform_coordinates_to_price'
    messages = state["messages"]
    response = agent_executor.invoke({"messages": messages})
    return {"messages": [response]}

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
    formatted_prompt = SENTIMENT_ANALYSIS_PROMPT.format(description=description)

    # 1. Get raw LLM result
    result = await llm.ainvoke(formatted_prompt)

    # 2. Map label to float score (The most important part!)
    score = map_label_to_score(result.label)

    raw_data = {
            "sentiment_score": map_label_to_score(result.label),
            "room_type": state["data"]["room_type"],
            "latitude": state["data"]["latitude"],
            "longitude": state["data"]["longitude"],
            "minimum_nights": int(state["data"]["minimum_nights"]),
            "number_of_reviews": int(state["data"]["number_of_reviews"]),
            "calculated_host_listings_count": int(state["data"]["calculated_host_listings_count"]),
            "availability_365": int(state["data"]["availability_365"]),
        }
        
    # 2. Validate and "Clean" everything at once
    try:
        clean_features = get_clean_data(raw_data)
        return {"clean_features": clean_features}
    except Exception as e:
        return {"error": f"Data validation error: {e}"}

def get_clean_data(data: dict) -> PropertyFeatures:
    """
    This is now your pipeline's 'Gatekeeper'.
    If any field is invalid (e.g., room_type is wrong or availability > 365),
    this will raise a Pydantic ValidationError.
    """
    return PropertyFeatures(**data)

def data_cleaning_node(state):
    # This node ensures only the relevant data columns persist in the state
    df = state["data"]
    cleaned_df = df.drop(columns=["neighbourhood_group", "neighbourhood"])
    return {"data": cleaned_df}
