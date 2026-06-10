import pandas as pd
import streamlit as st

MODEL_PATH = "models/"

# 1. Ensure this path points to an actual CSV spreadsheet, NOT a .pt model weights file!
FILE_PATHS = [
    "evaluation_predictions.csv",
    "resnet18_training_history.csv",
    "mobilenet_v3_small_training_history.csv",
    "training_history.csv",
]


def classify_prediction(row):
    actual = row["ground_truth"]
    pred = row["pred_resnet18"]

    if actual == "fire" and pred == "fire":
        return "#FF0000"  # TT (True Target) -> RED

    elif actual == "no_fire" and pred == "fire":
        return "#0000FF"  # FT (False Target / False Alarm) -> BLUE

    elif actual == "fire" and pred == "no_fire":
        return "#FFA500"  # FN (Missed Fire) -> ORANGE

    else:
        return "#808080"  # TN (True Negative) -> GRAY


def get_draw_order(color):
    if color == "#FFA500":
        return 3  # Orange mistakes on very top
    if color == "#0000FF":
        return 2  # Blue false alarms next
    if color == "#FF0000":
        return 1  # Red hits below mistakes
    return 0  # Gray backgrounds on the very bottom


def main():
    st.title("My App")

    selected_file = st.selectbox(label="Select a File:", options=FILE_PATHS)

    st.write(f"You selected: **{selected_file}**")

    try:
        # 2. Safely read your data spreadsheet
        df = pd.read_csv(MODEL_PATH + selected_file)

        # 3. Use st.dataframe() to render it beautifully on the web page
        st.dataframe(df)

        required_cols = ["epoch"]

        # Check if EVERY required column is present in the dataframe
        if all(col in df.columns for col in required_cols):
            # Find the numeric position of 'epoch'
            epoch_index = list(df.columns).index("epoch")

            # Slice the list to grab everything AFTER that position
            metric_cols = list(df.columns)[epoch_index + 1 :]

            selected_col = st.selectbox(label="Select a Column:", options=metric_cols)

            st.line_chart(
                data=df,
                x="epoch",
                y=selected_col,
            )
        else:
            required_cols = ["ground_truth", "pred_mobilenet_v3_small", "pred_resnet18"]

            if all(col in df.columns for col in required_cols):
                model_name = st.selectbox(
                    label="Select a Column:", options=required_cols[1:]
                )

                st.write(f"The Confusion Matrix (%): `{model_name}`")

                # Calculate the cross-tabulation table normalized by row (index)
                matrix_pct = pd.crosstab(
                    df["ground_truth"],
                    df[model_name],
                    rownames=["Actual"],
                    colnames=["Predicted"],
                    normalize="index",  # Each row totals 100% (Row Accuracy / Recall)
                )

                # Format numbers cleanly to percentages with 1 decimal place
                st.table(matrix_pct.style.format("{:.1%}"))

                df["tt_color"] = df.apply(classify_prediction, axis=1)

                st.map(
                    df,
                    latitude="latitude",
                    longitude="longitude",
                    color="tt_color",
                )

                st.markdown("""
                    **Map Legend:**
                    * 🔴 **RED** = True Target (TT)
                    * 🔵 **BLUE** = False Target / False Alarm (FT)
                    * 🟠 **ORANGE** = Missed Fire (FN)
                    * ⚪ **GRAY** = True Negative (TN)
                    """)

            else:
                missing = [col for col in required_cols if col not in df.columns]
                st.error(f"❌ Cannot plot chart. Missing columns from CSV: {missing}")

    except FileNotFoundError:
        st.error(f"❌ Could not find your data file at: {MODEL_PATH + selected_file}")
        st.info("💡 Creating a temporary preview dataset for you instead:")


if __name__ == "__main__":
    main()
