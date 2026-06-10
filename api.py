import asyncio
import io
import warnings

warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from src.agentic.evaluate import run_pipeline

app = FastAPI(title="Airbnb Price Tier Predictor")

_model = None


@app.on_event("startup")
async def startup():
    global _model
    train_df = pd.read_csv("Data/train.csv")
    _, _, _model = await run_pipeline(train_df, train_mode=True)
    print(f"Model trained on {len(train_df)} rows, ready to predict.")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _model
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    predictions, _, _ = await run_pipeline(df, model=_model)
    return JSONResponse({
        "predictions": predictions,
        "count": len(predictions),
    })


@app.get("/health")
async def health():
    return {"status": "ok"}
