import asyncio
import sys
from functools import partial
from typing import List, Optional, TypedDict, cast

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field, field_validator

from src.nodes import (
    PandasTool,
    debug_node,
    pre_processing_node,
    validate_schema_node,
    run_data_cleaning,
    run_feature_extraction,
    train_and_predict_node,
)

from src.schemas import SentimentAnalysis, PropertyFeatures
from src.state import AgentState
from src.DropCols import drop_dataframe_columns
from src.prompts import agent_prompt

# 1. Instantiate the object, don't just assign a string
llm_model = ChatOllama(model="llama3")

# 2. Now this will work because 'llm_model' is an object, not a string
structured_llm = llm_model.with_structured_output(SentimentAnalysis)

def build_hybrid_workflow(df: pd.DataFrame):
    # 1. Setup Tools
    pandas_instance = PandasTool(df)
    tools = [
        pandas_instance.get_column_info,
        pandas_instance.preview_data,
        run_data_cleaning,
        run_feature_extraction,
    ]
    
    # 2. Define the Intelligent Agent
    # This agent can be invoked by any node
    agent_executor = create_react_agent(llm_model, tools, prompt=agent_prompt)

    # 3. Build the Graph
    workflow = StateGraph(AgentState)

    # # 1. Add the node
    # workflow.add_node("cleaner", drop_dataframe_columns)
    
    # # 2. Define the edge (this is likely the missing piece!)
    # # Your graph needs to know to go to the cleaner after/before other nodes
    # workflow.add_edge("start_node", "cleaner")
    # workflow.add_edge("cleaner", "extract_features_node")

    # Wrap the Agent inside a function that your workflow can call
    async def agentic_node(state: AgentState):
        result = await agent_executor.ainvoke(state)
        return {"messages": result["messages"]}

    # Define your workflow nodes
    workflow.add_node("preprocess", pre_processing_node)
    workflow.add_node("validate", validate_schema_node)
    workflow.add_node("debug", partial(debug_node, tool=pandas_instance))
    workflow.add_node("process", agentic_node) # Uses the tools + agent logic
    workflow.add_node("predict", train_and_predict_node)

    # 4. Enforce your explicit edges
    workflow.set_entry_point("validate")
    workflow.add_conditional_edges(
        "validate", lambda s: "process" if s["schema_ok"] else "debug"
    )
    workflow.add_edge("debug", END)
    workflow.add_edge("process", "predict")
    workflow.add_edge("predict", END)

    return workflow.compile()