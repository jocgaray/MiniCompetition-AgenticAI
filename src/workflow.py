from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.nodes import (
    debug_node,
    train_and_predict_node,
    validate_schema_node,
)
from src.prompts import AGENT_SYSTEM_PROMPT, SENTIMENT_ANALYSIS_PROMPT
from src.schemas import SentimentAnalysis
from src.state import AgentState
from src.zipcode_dollars import get_price_from_lat_long
from src.logic import map_label_to_score
import pandas as pd

llm_model = ChatOllama(model="phi4", temperature=0)
structured_llm = llm_model.with_structured_output(SentimentAnalysis)


def build_workflow(df: pd.DataFrame):
    workflow = StateGraph(AgentState)

    async def process_node(state: AgentState):
        df_current = state["raw_data"].copy()

        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]

        max_steps = 10
        step = 0
        done = False

        while step < max_steps and not done:
            cols = list(df_current.columns)
            preview = df_current.head(3).to_string()

            state_msg = (
                f"Step {step + 1}. Current columns: {cols}\n"
                f"First 3 rows:\n{preview}\n\n"
                "What action should I take next?"
            )
            messages.append(HumanMessage(content=state_msg))

            response = await llm_model.ainvoke(messages)
            reply = response.content.strip().lower()
            messages.append(response)

            if "clean" in reply and ("neighbourhood" in reply or "column" in reply):
                for col in ["neighbourhood_group", "neighbourhood"]:
                    if col in df_current.columns:
                        df_current.drop(columns=[col], inplace=True, errors="ignore")
                messages.append(HumanMessage(content="Cleaned neighbourhood columns."))

            elif "transform" in reply or "region_price" in reply or "coordinate" in reply:
                if "latitude" in df_current.columns and "longitude" in df_current.columns:
                    df_current["region_price"] = df_current.apply(
                        lambda r: get_price_from_lat_long(
                            float(r.get("latitude", 0)), float(r.get("longitude", 0))
                        ),
                        axis=1,
                    )
                    df_current.drop(columns=["latitude", "longitude"], inplace=True, errors="ignore")
                    messages.append(HumanMessage(content="Transformed lat/lon to region_price."))

            elif "analyze" in reply or "sentiment" in reply or "description" in reply:
                scores = []
                for idx, row in df_current.iterrows():
                    desc = row.get("description", "")
                    if not desc or not isinstance(desc, str) or len(desc.strip()) == 0:
                        scores.append(2.5)
                        continue
                    try:
                        prompt_text = SENTIMENT_ANALYSIS_PROMPT.format(description=desc)
                        result = structured_llm.invoke(prompt_text)
                        score = map_label_to_score(result.label)
                        scores.append(score)
                    except Exception:
                        scores.append(2.5)

                df_current["sentiment_score"] = scores
                avg = sum(scores) / len(scores) if scores else 0
                messages.append(HumanMessage(content=f"Sentiment analysis complete. Mean score: {avg:.2f}"))

            elif "done" in reply or "complete" in reply or "finished" in reply:
                done = True
                messages.append(HumanMessage(content="Pipeline complete."))

            elif "inspect" in reply or "preview" in reply or "look" in reply or "column" in reply:
                info = f"Shape: {df_current.shape}\nColumns: {list(df_current.dtypes)}"
                messages.append(HumanMessage(content=info))

            else:
                messages.append(HumanMessage(
                    content="I don't understand. Available actions: INSPECT, CLEAN, TRANSFORM, ANALYZE, DONE."
                ))

            step += 1

        return {"raw_data": df_current, "messages": messages}

    def debug_node_wrapper(state: AgentState):
        return debug_node(state)

    workflow.add_node("validate", validate_schema_node)
    workflow.add_node("process", process_node)
    workflow.add_node("predict", train_and_predict_node)
    workflow.add_node("debug", debug_node_wrapper)

    workflow.set_entry_point("validate")

    workflow.add_conditional_edges(
        "validate",
        lambda s: "process" if s.get("schema_ok") else "debug",
    )

    workflow.add_edge("debug", END)
    workflow.add_edge("process", "predict")
    workflow.add_edge("predict", END)

    return workflow.compile()
