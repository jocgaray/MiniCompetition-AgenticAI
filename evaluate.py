import asyncio
import warnings

warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from src.logic import map_label_to_score
from src.prompts import AGENT_SYSTEM_PROMPT, SENTIMENT_ANALYSIS_PROMPT
from src.schemas import SentimentAnalysis
from src.zipcode_dollars import get_price_from_lat_long

llm_model = ChatOllama(model="phi4", temperature=0)
structured_llm = llm_model.with_structured_output(SentimentAnalysis)


async def agent_process(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)]
    max_steps = 15
    step = 0
    done = False

    while step < max_steps and not done:

        has_neighbourhood = "neighbourhood_group" in df.columns or "neighbourhood" in df.columns
        has_coords = "latitude" in df.columns and "longitude" in df.columns
        has_sentiment = "sentiment_score" in df.columns
        all_done = not has_neighbourhood and not has_coords and has_sentiment

        if all_done and step >= 3:
            done = True
            messages.append(HumanMessage(content="Pipeline complete."))
            break

        cols = list(df.columns)
        preview = df.head(3).to_string()

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
                if col in df.columns:
                    df.drop(columns=[col], inplace=True, errors="ignore")
            messages.append(HumanMessage(content="Cleaned neighbourhood columns."))

        elif "transform" in reply or "region_price" in reply or "coordinate" in reply:
            if "latitude" in df.columns and "longitude" in df.columns:
                df["region_price"] = df.apply(
                    lambda r: get_price_from_lat_long(
                        float(r.get("latitude", 0)), float(r.get("longitude", 0))
                    ),
                    axis=1,
                )
                df.drop(columns=["latitude", "longitude"], inplace=True, errors="ignore")
                messages.append(HumanMessage(content="Transformed lat/lon to region_price."))

        elif "analyze" in reply or "sentiment" in reply or "description" in reply:
            scores = []
            for idx, row in df.iterrows():
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

            df["sentiment_score"] = scores
            avg = sum(scores) / len(scores) if scores else 0
            messages.append(HumanMessage(content=f"Sentiment analysis complete. Mean score: {avg:.2f}"))

        elif "done" in reply or "complete" in reply or "finished" in reply:
            done = True

        elif "inspect" in reply or "preview" in reply or "look" in reply or "column" in reply:
            info = f"Shape: {df.shape}\nColumns: {list(df.dtypes)}"
            messages.append(HumanMessage(content=info))

        else:
            messages.append(HumanMessage(
                content="I don't understand. Available actions: INSPECT, CLEAN, TRANSFORM, ANALYZE, DONE."
            ))

        step += 1

    if not done:
        has_neighbourhood = "neighbourhood_group" in df.columns or "neighbourhood" in df.columns
        has_coords = "latitude" in df.columns and "longitude" in df.columns
        has_sentiment = "sentiment_score" in df.columns
        if not has_neighbourhood and not has_coords:
            if not has_sentiment:
                scores = [2.5] * len(df)
                df["sentiment_score"] = scores

    return df


FEATURE_COLS = [
    "sentiment_score",
    "region_price",
    "minimum_nights",
    "number_of_reviews",
    "calculated_host_listings_count",
    "availability_365",
]


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    return df[available].fillna(0)


def train_model(df: pd.DataFrame) -> RandomForestClassifier:
    X = extract_features(df)
    y = df["price_tier"]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model


def print_truth_table(y_true, y_pred):
    print("--- Truth Table (Confusion Matrix) ---")
    cm = confusion_matrix(y_true, y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    header = "Actual \\ Pred  " + "  ".join(f"{l:>6}" for l in labels)
    print(header)
    for i, row_label in enumerate(labels):
        row = f"{row_label:>14}  " + "  ".join(f"{cm[i][j]:>6}" for j in range(len(labels)))
        print(row)
    print()

    print("--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=[f"{l}" for l in labels], digits=3)
    print(report)


async def main():
    print("=" * 60)
    print("Phase 1: Processing training data")
    print("=" * 60)
    train_df = pd.read_csv("Data/train.csv")
    processed_train = await agent_process(train_df)
    model = train_model(processed_train)

    X_train = extract_features(processed_train)
    train_preds = model.predict(X_train)
    print_truth_table(processed_train["price_tier"], train_preds)

    print("=" * 60)
    print("Phase 2: Predicting test data")
    print("=" * 60)
    test_df = pd.read_csv("Data/test.csv")
    processed_test = await agent_process(test_df)

    X_test = extract_features(processed_test)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    test_preds = model.predict(X_test)

    output = pd.DataFrame({
        "property_id": test_df["property_id"],
        "predicted_tier": test_preds,
    })
    output.to_csv("predictions_test.csv", index=False)
    print(f"Saved {len(output)} predictions to predictions_test.csv")
    print(f"\nPrediction distribution:")
    print(output["predicted_tier"].value_counts().sort_index())

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
