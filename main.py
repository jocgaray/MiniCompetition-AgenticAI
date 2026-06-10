import asyncio

import pandas as pd

from src.workflow import build_workflow


async def run_competition_run(filepath: str):
    df = pd.read_csv(filepath)
    app = build_workflow(df)

    initial_state = {
        "raw_data": df,
        "clean_features": None,
        "predictions": None,
        "errors": [],
        "schema_ok": False,
    }

    return await app.ainvoke(initial_state)


if __name__ == "__main__":
    result = asyncio.run(run_competition_run("Data/train.csv"))
    print("Predictions:", result.get("predictions"))
