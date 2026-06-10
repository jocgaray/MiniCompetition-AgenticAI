# DSR Mini-Competition: Agentic AI

Predict NYC Airbnb price tiers (0–3) using a LangChain/LangGraph agentic pipeline.

## Structure

```
src/
├── shared/           # Reused modules (schemas, logic, zipcode, DropCols)
├── deterministic/    # Fast fixed pipeline (phi4 sentiment only)
│   └── evaluate.py
└── agentic/          # Agent-driven pipeline (qwen3.5 + tools)
    ├── evaluate.py
    └── prompts.py    # Original competition prompts restored
```

## Run

```bash
# Deterministic (fast baseline)
uv run python main.py deterministic

# Agentic (qwen3.5 decides next steps)
uv run python main.py agentic
```

Both output a truth table (row %), classification report, and save test predictions.

## Deploy

```bash
uv run uvicorn api:app --host 0.0.0.0 --port 8000
```
