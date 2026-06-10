import asyncio
import io

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.workflow import build_workflow

app = FastAPI(title="Airbnb Price Tier Predictor")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))

    initial_state = {
        "raw_data": df,
        "clean_features": None,
        "predictions": None,
        "errors": [],
        "schema_ok": False,
    }

    graph = build_workflow(df)
    result = await graph.ainvoke(initial_state)

    predictions = result.get("predictions", [])
    errors = result.get("errors", [])

    return JSONResponse({
        "predictions": predictions,
        "errors": errors,
        "count": len(predictions),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
