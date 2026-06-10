# DSR Mini-Competition: Agentic AI

Agentic pipeline for predicting NYC Airbnb price tiers (0-3) using LangGraph + Ollama.

## Architecture

```
validate → process (ReAct Agent) → predict → END
```

The **ReAct agent** orchestrates the pipeline autonomously:
1. Inspects data schema
2. Cleans unnecessary columns
3. Transforms lat/lon → region_price
4. Runs LLM sentiment analysis on descriptions
5. Prepares features for ML

A deterministic **RandomForest classifier** then predicts price tier.

## Setup

```bash
uv sync
ollama pull llama3
```

## Run

```bash
uv run python main.py
```

## API (Railway deployment)

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST -F "file=@Data/test.csv" http://localhost:8000/predict
```
