# main.py
import asyncio

import pandas as pd

from src.agent_template import build_workflow

async def run_competition_run(filepath: str):
    df = pd.read_csv(filepath)
    # Ensure your workflow is compiled
    app = build_workflow(df)
    
    initial_state = {
        "raw_data": df,
        "clean_features": pd.DataFrame(),
        "predictions": None,
        "errors": [],
        "schema_ok": False
    }
    
    # Use ainvoke for asynchronous graph execution
    return await app.ainvoke(initial_state)

if __name__ == "__main__":
    # This is the ONLY way to start an async function from a script
    result = asyncio.run(run_competition_run("Data/train.csv"))
    print("Execution complete. Predictions found:", result.get("predictions"))
