from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "emotion.csv"
OUTPUT_FILE = BASE_DIR / "emotion_annotations.csv"

EMOTIONS = {
    0: "Sadness",
    1: "Joy",
    2: "Love",
    3: "Anger",
    4: "Fear",
    5: "Surprise",
}


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load the source data and merge any previously saved annotations."""
    data = pd.read_csv(DATA_FILE)

    if "text" not in data.columns:
        raise ValueError("The dataset must contain a 'text' column.")
    data["human_label"] = pd.Series(pd.NA, index=data.index, dtype="Int64")

    if OUTPUT_FILE.exists():
        annotations = pd.read_csv(OUTPUT_FILE, dtype={"human_label": "Int64"})
        if {"row_index", "human_label"}.issubset(annotations.columns):
            valid = annotations["row_index"].between(0, len(data) - 1)
            annotations = annotations.loc[valid]
            data.loc[annotations["row_index"], "human_label"] = annotations[
                "human_label"
            ].to_numpy()

    return data[:50]


def save_annotation(row_index: int, label: int) -> None:
    """Insert or update one annotation without rewriting the large dataset."""
    if OUTPUT_FILE.exists():
        annotations = pd.read_csv(OUTPUT_FILE)
        annotations = annotations[annotations["row_index"] != row_index]
    else:
        annotations = pd.DataFrame(columns=["row_index", "human_label"])

    new_annotation = pd.DataFrame(
        [{"row_index": row_index, "human_label": label}]
    )
    annotations = pd.concat([annotations, new_annotation], ignore_index=True)
    annotations["row_index"] = annotations["row_index"].astype(int)
    annotations["human_label"] = annotations["human_label"].astype(int)
    annotations.sort_values("row_index").to_csv(OUTPUT_FILE, index=False)


@st.dialog("Emotion labelling tutorial", width="large", dismissible=False)
def show_tutorial() -> None:
    step = st.session_state.tutorial_step
    total_steps = 4
    st.progress((step + 1) / total_steps, text=f"Step {step + 1} of {total_steps}")

    if step == 0:
        st.subheader("Understand the task")
        st.write(
            "For every tweet, identify the **main emotion expressed by its author**. "
            "Choose exactly one of the six available categories."
        )
        st.info(
            "Focus on how the author feels—not the tweet's general topic or how "
            "someone else in the story feels."
        )

    elif step == 1:
        st.subheader("Learn the positive emotions")
        st.markdown(
            """
            - **Joy** — happiness, pleasure, excitement, pride, or optimism.
            - **Love** — affection, caring, admiration, attachment, or tenderness.
            - **Surprise** — shock, amazement, astonishment, or an unexpected reaction.
            """
        )
        st.info(
            'Example: "I cannot believe they remembered my birthday!" is '
            "**Surprise** when the unexpected reaction is the focus."
        )

    elif step == 2:
        st.subheader("Learn the negative emotions")
        st.markdown(
            """
            - **Sadness** — unhappiness, grief, loneliness, disappointment, or feeling low.
            - **Anger** — rage, irritation, frustration, resentment, or hostility.
            - **Fear** — worry, anxiety, nervousness, dread, or feeling threatened.
            """
        )
        st.info(
            'Example: "I am nervous about what will happen tomorrow" is **Fear** '
            "because worry about the future is central."
        )

    else:
        st.subheader("Label each tweet")
        st.markdown(
            """
            1. Read the entire tweet and consider its context.
            2. Choose the strongest or most central emotion.
            3. Select the matching category.
            4. Click **Save & Next** to record the label.
            5. Use **Skip** only when you cannot make a reasonable choice.
            """
        )
        st.success(
            "Labels are saved automatically. You can reopen this tutorial from "
            "the sidebar at any time."
        )

    previous_column, next_column = st.columns(2)
    with previous_column:
        if st.button(
            "← Back",
            use_container_width=True,
            disabled=step == 0,
            key="tutorial_previous",
        ):
            st.session_state.tutorial_step -= 1
            st.rerun()

    with next_column:
        if step < total_steps - 1:
            if st.button(
                "Next →",
                type="primary",
                use_container_width=True,
                key="tutorial_next",
            ):
                st.session_state.tutorial_step += 1
                st.rerun()
        else:
            button_text = (
                "Start annotating"
                if not st.session_state.tutorial_seen
                else "Return to annotations"
            )
            if st.button(
                button_text,
                type="primary",
                use_container_width=True,
                key="tutorial_finish",
            ):
                st.session_state.tutorial_seen = True
                st.session_state.tutorial_open = False
                st.session_state.tutorial_step = 0
                st.rerun()


st.set_page_config(page_title="Emotion Labelling", page_icon="🏷️", layout="centered")
st.title("Emotion Labelling Dashboard")
st.caption("Read each tweet and assign the emotion that best describes it.")

if "tutorial_seen" not in st.session_state:
    st.session_state.tutorial_seen = False
if "tutorial_open" not in st.session_state:
    st.session_state.tutorial_open = True
if "tutorial_step" not in st.session_state:
    st.session_state.tutorial_step = 0

try:
    df = load_data()
except (FileNotFoundError, ValueError) as error:
    st.error(f"Could not load the dataset: {error}")
    st.stop()

if "row_index" not in st.session_state:
    first_unlabelled = df["human_label"].isna()
    st.session_state.row_index = (
        int(first_unlabelled.idxmax()) if first_unlabelled.any() else 0
    )

row_index = max(0, min(st.session_state.row_index, len(df) - 1))
st.session_state.row_index = row_index

labelled_count = int(df["human_label"].notna().sum())
st.progress(labelled_count / len(df), text=f"{labelled_count:,} of {len(df):,} labelled")

st.subheader(f"Tweet {row_index + 1:,} of {len(df):,}")
st.info(str(df.at[row_index, "text"]))

current_label = df.at[row_index, "human_label"]
default_choice = EMOTIONS.get(int(current_label)) if pd.notna(current_label) else None
choice = st.radio(
    "Choose a category",
    list(EMOTIONS.values()),
    index=list(EMOTIONS.values()).index(default_choice) if default_choice else None,
)

previous_column, save_column, skip_column = st.columns(3)

with previous_column:
    if st.button("← Previous", use_container_width=True, disabled=row_index == 0):
        st.session_state.row_index -= 1
        st.rerun()

with save_column:
    if st.button(
        "Save & Next",
        type="primary",
        use_container_width=True,
        disabled=choice is None,
    ):
        selected_label = next(
            label_id for label_id, emotion in EMOTIONS.items() if emotion == choice
        )
        save_annotation(row_index, selected_label)
        st.cache_data.clear()
        if row_index < len(df) - 1:
            st.session_state.row_index += 1
        st.rerun()

with skip_column:
    if st.button(
        "Skip →", use_container_width=True, disabled=row_index == len(df) - 1
    ):
        st.session_state.row_index += 1
        st.rerun()

with st.sidebar:
    st.header("Help")
    if st.button("📘 View tutorial", use_container_width=True):
        st.session_state.tutorial_open = True
        st.session_state.tutorial_step = 0
        st.rerun()

    st.divider()
    st.header("Navigation")
    jump_to = st.number_input(
        "Go to tweet",
        min_value=1,
        max_value=len(df),
        value=row_index + 1,
        step=1,
    )
    if st.button("Go", use_container_width=True):
        st.session_state.row_index = int(jump_to) - 1
        st.rerun()

    if st.button("Go to next unlabelled", use_container_width=True):
        unlabelled = df.index[df["human_label"].isna()]
        later_rows = unlabelled[unlabelled > row_index]
        if len(later_rows):
            st.session_state.row_index = int(later_rows[0])
        elif len(unlabelled):
            st.session_state.row_index = int(unlabelled[0])
        else:
            st.success("Every tweet has been labelled.")
        st.rerun()

    st.divider()
    st.caption(f"Annotations are saved to `{OUTPUT_FILE}`.")

if st.session_state.tutorial_open:
    show_tutorial()