import asyncio
import re
import time
import warnings

warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

import pandas as pd
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from src.agentic.prompts import SYSTEM_INSTRUCTIONS

llm_model = ChatOllama(model="qwen3.5:9b", temperature=0)

NO_INTEREST = {
    "basic", "budget", "cheap", "simple", "small", "tiny", "bare",
    "minimal", "modest", "nothing special", "nothing fancy",
}

NEUTRAL = {
    "clean", "quiet", "safe", "convenient", "functional", "comfortable",
    "decent", "ok", "fine", "average", "standard", "typical",
}

SOME_APPEAL = {
    "cozy", "nice", "good", "great", "spacious", "bright", "comfy",
    "well-maintained", "well-equipped", "good location", "close to",
    "friendly", "welcome", "enjoy", "relax",
}

INTERESTED = {
    "beautiful", "lovely", "charming", "stylish", "modern", "renovated",
    "updated", "elegant", "gorgeous", "stunning", "amazing", "wonderful",
    "fantastic", "excellent", "perfect", "ideal", "warm", "inviting",
}

MUST_SEE = {
    "luxury", "luxurious", "high-end", "premium", "designer", "penthouse",
    "breathtaking", "exquisite", "impeccable", "spectacular", "magnificent",
    "one-of-a-kind", "unique", "exceptional", "outstanding", "incredible",
    "unforgettable",
}

LABEL_MAP = {
    "no_interest": 0.0,
    "neutral": 1.25,
    "some_appeal": 2.5,
    "interested": 3.75,
    "must_see": 5.0,
}

LABEL_LISTS = [
    ("no_interest", NO_INTEREST),
    ("neutral", NEUTRAL),
    ("some_appeal", SOME_APPEAL),
    ("interested", INTERESTED),
    ("must_see", MUST_SEE),
]

CLASS_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]


def compute_appeal(desc: str) -> float:
    if not desc or not isinstance(desc, str) or len(desc.strip()) == 0:
        return 2.5
    desc_lower = desc.lower()
    best_score = 2.5
    for label, words in LABEL_LISTS:
        for word in words:
            if word in desc_lower:
                best_score = max(best_score, LABEL_MAP[label])
    return best_score


def print_truth_table(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    cm_pct = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

    header = "Actual \\ Pred  " + "  ".join(f"{l:>7}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>14}  " + "  ".join(
            f"{cm_pct[i][j]:>6.1f}%" for j in range(len(labels))
        )
        print(row)
    print()


def print_metrics_summary(y_true, y_pred):
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=3,
        output_dict=True,
    )
    acc = report["accuracy"]
    print("--- Metrics Summary ---")
    print(f"{'Class':<16} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 48)
    for name in CLASS_NAMES:
        r = report[name]
        print(f"{name:<16} {r['precision']:>9.1%} {r['recall']:>9.1%} {r['f1-score']:>9.1%}")
    print("-" * 48)
    print(f"{'Accuracy':<16} {acc:>9.1%}")
    print()


def make_agent_tools(df: pd.DataFrame):

    @tool
    def get_column_info() -> str:
        """Returns the column names and data types of the current dataframe."""
        return str(df.dtypes)

    @tool
    def preview_data(n: int = 5) -> str:
        """Returns the first n rows of the dataframe."""
        return df.head(n).to_string()

    @tool
    def drop_dataframe_columns(columns: str) -> str:
        """Drop specified columns from the dataframe. Provide as comma-separated list. Example: 'neighbourhood_group, neighbourhood'"""
        t0 = time.time()
        col_list = [c.strip() for c in columns.split(",")]
        df.drop(columns=col_list, inplace=True, errors="ignore")
        elapsed = time.time() - t0
        print(f"  [{elapsed:.2f}s] Clean — dropped {col_list}")
        return f"Dropped columns: {col_list}"

    @tool
    def encode_room_type() -> str:
        """One-hot encode the room_type column into binary indicator columns (room_Private_room, room_Shared_room, room_Entire_home_apt)."""
        t0 = time.time()
        if "room_type" in df.columns:
            dummies = pd.get_dummies(df["room_type"], prefix="room")
            for col in ["room_Private room", "room_Shared room", "room_Entire home/apt"]:
                if col in dummies.columns:
                    safe = col.replace(" ", "_").replace("/", "_")
                    df[safe] = dummies[col]
        elapsed = time.time() - t0
        print(f"  [{elapsed:.2f}s] Encode room_type")
        return "One-hot encoded room_type into room_Private_room, room_Shared_room, room_Entire_home_apt"

    @tool
    def analyze_appeal() -> str:
        """Classify each property description on a purchase-intent scale. Adds appeal_score column (0.0–5.0) where higher means more compelling to book."""
        t0 = time.time()
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Appeal", unit="row"):
            scores.append(compute_appeal(row.get("description", "")))
        df["appeal_score"] = scores
        avg = sum(scores) / len(scores) if scores else 0
        elapsed = time.time() - t0
        print(f"  [{elapsed:.2f}s] Appeal — mean: {avg:.2f}")
        return f"Appeal classification complete. Mean score: {avg:.2f}"

    return [get_column_info, preview_data, drop_dataframe_columns, encode_room_type, analyze_appeal]


async def run_pipeline(df_input: pd.DataFrame, model=None, train_mode=False) -> tuple[list[int], pd.DataFrame]:
    df = df_input.copy()

    if "min_nights_required" in df.columns and "minimum_nights" not in df.columns:
        df.rename(columns={"min_nights_required": "minimum_nights"}, inplace=True)

    tool_list = make_agent_tools(df)
    agent = create_react_agent(llm_model, tool_list, prompt=SYSTEM_INSTRUCTIONS)

    t_start = time.time()
    if train_mode:
        input_msg = (
            "I have a dataset of NYC Airbnb listings loaded. "
            "Inspect it, clean unnecessary columns, encode room types, "
            "and classify the descriptions for appeal. Report progress at each step."
        )
        await agent.ainvoke({"messages": [("human", input_msg)]})
    elif "appeal_score" not in df.columns:
        t0 = time.time()
        for col in ["neighbourhood_group", "neighbourhood"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        print(f"  [{time.time()-t0:.2f}s] Clean")

        t0 = time.time()
        if "room_type" in df.columns:
            dummies = pd.get_dummies(df["room_type"], prefix="room")
            for col in ["room_Private room", "room_Shared room", "room_Entire home/apt"]:
                if col in dummies.columns:
                    safe = col.replace(" ", "_").replace("/", "_")
                    df[safe] = dummies[col]
        print(f"  [{time.time()-t0:.2f}s] Encode room_type")

        t0 = time.time()
        scores = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Appeal", unit="row"):
            scores.append(compute_appeal(row.get("description", "")))
        df["appeal_score"] = scores
        avg = sum(scores) / len(scores) if scores else 0
        print(f"  [{time.time()-t0:.2f}s] Appeal — mean: {avg:.2f}")
    total_time = time.time() - t_start

    feature_cols = [
        "appeal_score",
        "latitude",
        "longitude",
        "room_Private_room",
        "room_Shared_room",
        "room_Entire_home_apt",
        "minimum_nights",
        "number_of_reviews",
        "calculated_host_listings_count",
        "availability_365",
    ]
    available = [c for c in feature_cols if c in df.columns]

    t0 = time.time()
    if model is None and "price_tier" in df.columns and available:
        train_df = df.dropna(subset=available + ["price_tier"])
        X = train_df[available]
        y = train_df["price_tier"]
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

    if model is not None and available:
        pred_df = df[available].fillna(0)
        available_in_model = [c for c in available if hasattr(model, "feature_names_in_") and c in model.feature_names_in_]
        if not available_in_model:
            available_in_model = available
        pred_df = pred_df[available_in_model]
        predictions = model.predict(pred_df).tolist()
    else:
        predictions = []

    train_time = time.time() - t0
    print(f"  [{train_time:.2f}s] Train & predict")
    print(f"  Total (agent + train): {total_time + train_time:.2f}s")
    return predictions, df, model


async def main():
    print("=" * 60)
    print("Agentic Pipeline (qwen3.5 + create_react_agent)")
    print("=" * 60)

    print("Phase 1: Training")
    train_df = pd.read_csv("Data/train.csv")
    _, _, model = await run_pipeline(train_df, train_mode=True)
    print(f"  Trained on {len(train_df)} rows")

    print("\nPhase 2: Validation")
    val_df = pd.read_csv("Data/validation.csv")
    val_preds, processed_val, _ = await run_pipeline(val_df, model=model)
    print_truth_table(processed_val["price_tier"], val_preds)
    print_metrics_summary(processed_val["price_tier"], val_preds)

    print("\nPhase 3: Test prediction")
    test_df = pd.read_csv("Data/test.csv")
    test_preds, _, _ = await run_pipeline(test_df, model=model)

    output = pd.DataFrame({
        "property_id": test_df["property_id"],
        "predicted_tier": test_preds,
    })
    output.to_csv("predictions_agentic_test.csv", index=False)
    print(f"Saved {len(output)} predictions to predictions_agentic_test.csv")
    print(f"Distribution: {output['predicted_tier'].value_counts().sort_index().to_dict()}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
