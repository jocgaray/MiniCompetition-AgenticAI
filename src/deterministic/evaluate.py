import asyncio
import re
import time
import warnings

warnings.filterwarnings("ignore", message="Using slow pure-python SequenceMatcher")

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

POSITIVE_WORDS = {
    "luxury", "beautiful", "stunning", "amazing", "gorgeous", "spacious",
    "modern", "renovated", "cozy", "charming", "clean", "quiet", "safe",
    "convenient", "bright", "comfortable", "excellent", "fantastic",
    "perfect", "great", "nice", "lovely", "wonderful", "breathtaking",
    "high-end", "elegant", "stylish", "updated", "new", "premium",
}

NEGATIVE_WORDS = {
    "dirty", "small", "old", "broken", "noisy", "dangerous", "terrible",
    "horrible", "awful", "bad", "disgusting", "ugly", "cramped",
    "rundown", "shabby", "smelly", "dark", "cold", "drafty",
    "uncomfortable", "filthy", "ugly", "worst", "poor", "disappointing",
    "maintenance", "needs work",
}


def compute_sentiment(desc: str) -> float:
    if not desc or not isinstance(desc, str) or len(desc.strip()) == 0:
        return 2.5
    words = set(re.findall(r"[a-z]+(?:-[a-z]+)?", desc.lower()))
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 2.5
    ratio = pos / total
    return round(ratio * 5.0, 2)


CLASS_NAMES = ["Budget", "Standard", "Premium", "Ultra-Luxury"]


def print_truth_table(y_true, y_pred):
    print("--- Truth Table (Row %) ---")
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


def run_pipeline(df_input: pd.DataFrame, model=None) -> tuple[list[int], pd.DataFrame]:
    df = df_input.copy()
    timings = {}

    if "min_nights_required" in df.columns and "minimum_nights" not in df.columns:
        df.rename(columns={"min_nights_required": "minimum_nights"}, inplace=True)

    t0 = time.time()
    for col in ["neighbourhood_group", "neighbourhood"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True, errors="ignore")
    timings["Clean"] = time.time() - t0
    print(f"  [{timings['Clean']:.2f}s] Clean")

    t0 = time.time()
    if "room_type" in df.columns:
        dummies = pd.get_dummies(df["room_type"], prefix="room")
        for col in ["room_Private room", "room_Shared room", "room_Entire home/apt"]:
            if col in dummies.columns:
                safe = col.replace(" ", "_").replace("/", "_")
                df[safe] = dummies[col]
    timings["Encode"] = time.time() - t0
    print(f"  [{timings['Encode']:.2f}s] Encode room_type")

    t0 = time.time()
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Sentiment", unit="row"):
        scores.append(compute_sentiment(row.get("description", "")))
    df["sentiment_score"] = scores
    timings["Sentiment"] = time.time() - t0
    print(f"  [{timings['Sentiment']:.2f}s] Sentiment analysis")

    feature_cols = [
        "sentiment_score",
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
    timings["Train"] = time.time() - t0
    print(f"  [{timings['Train']:.2f}s] Train & predict")

    print(f"  Total: {sum(timings.values()):.2f}s")
    return predictions, df, model


async def main():
    print("=" * 60)
    print("Deterministic Pipeline")
    print("=" * 60)

    print("Phase 1: Training")
    train_df = pd.read_csv("Data/train.csv")
    _, _, model = run_pipeline(train_df)
    print(f"  Trained on {len(train_df)} rows")

    print("\nPhase 2: Validation")
    val_df = pd.read_csv("Data/validation.csv")
    val_preds, processed_val, _ = run_pipeline(val_df, model=model)
    print_truth_table(processed_val["price_tier"], val_preds)
    print_metrics_summary(processed_val["price_tier"], val_preds)

    print("\nPhase 3: Test prediction")
    test_df = pd.read_csv("Data/test.csv")
    test_preds, _, _ = run_pipeline(test_df, model=model)

    output = pd.DataFrame({
        "property_id": test_df["property_id"],
        "predicted_tier": test_preds,
    })
    output.to_csv("predictions_test.csv", index=False)
    print(f"Saved {len(output)} predictions to predictions_test.csv")
    print(f"Distribution: {output['predicted_tier'].value_counts().sort_index().to_dict()}")
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
