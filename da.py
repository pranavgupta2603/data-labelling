import hashlib
import random
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "emotion.csv"
OUTPUT_FILE = BASE_DIR / "emotion_annotations.csv"
TWEETS_PER_PARTICIPANT = 5
ANNOTATION_COLUMNS = ["participant_id", "row_index", "human_label", "labeled_at"]
ANNOTATION_LOCK = threading.Lock()
GOOGLE_API_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

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
    """Load the source tweets."""
    data = pd.read_csv(DATA_FILE)

    if "text" not in data.columns:
        raise ValueError("The dataset must contain a 'text' column.")
    return data


def google_sheets_configured() -> bool:
    """Return whether Google Sheets credentials are available."""
    try:
        config = st.secrets["connections"]["gsheets"]
        return bool(config.get("spreadsheet"))
    except (FileNotFoundError, KeyError):
        return False


@st.cache_resource
def get_annotations_worksheet():
    """Connect to the private annotations worksheet."""
    import gspread
    from google.oauth2.service_account import Credentials

    config = dict(st.secrets["connections"]["gsheets"])
    spreadsheet = config.pop("spreadsheet")
    worksheet_name = config.pop("worksheet", "annotations")
    config.pop("type", None)

    credentials = Credentials.from_service_account_info(
        config, scopes=GOOGLE_API_SCOPES
    )
    client = gspread.authorize(credentials)
    workbook = (
        client.open_by_url(spreadsheet)
        if spreadsheet.startswith("http")
        else client.open(spreadsheet)
    )
    return workbook.worksheet(worksheet_name)


def load_annotations() -> pd.DataFrame:
    """Load the latest annotation for each participant and tweet."""
    if google_sheets_configured():
        worksheet = get_annotations_worksheet()
        records = worksheet.get_all_records()
        annotations = pd.DataFrame(records)
    elif OUTPUT_FILE.exists():
        annotations = pd.read_csv(OUTPUT_FILE)
    else:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    if "participant_id" not in annotations.columns:
        annotations["participant_id"] = "legacy"
    if "labeled_at" not in annotations.columns:
        annotations["labeled_at"] = ""

    if not {"row_index", "human_label"}.issubset(annotations.columns):
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    annotations = annotations[ANNOTATION_COLUMNS]
    return annotations.drop_duplicates(
        subset=["participant_id", "row_index"], keep="last"
    )


def get_assignment(participant_id: str, dataset_size: int) -> list[int]:
    """Create a stable random assignment unique to the participant ID."""
    digest = hashlib.sha256(participant_id.casefold().encode("utf-8")).digest()
    generator = random.Random(int.from_bytes(digest[:8], "big"))
    return generator.sample(range(dataset_size), TWEETS_PER_PARTICIPANT)


def save_annotation(participant_id: str, row_index: int, label: int) -> None:
    """Insert or update one participant's annotation."""
    with ANNOTATION_LOCK:
        timestamp = datetime.now(timezone.utc).isoformat()

        if google_sheets_configured():
            worksheet = get_annotations_worksheet()
            if not worksheet.row_values(1):
                worksheet.append_row(ANNOTATION_COLUMNS)
            worksheet.append_row(
                [participant_id, row_index, label, timestamp],
                value_input_option="RAW",
            )
            return

        annotations = load_annotations()
        existing_record = (annotations["participant_id"] == participant_id) & (
            annotations["row_index"] == row_index
        )
        annotations = annotations.loc[~existing_record]

        new_annotation = pd.DataFrame(
            [
                {
                    "participant_id": participant_id,
                    "row_index": row_index,
                    "human_label": label,
                    "labeled_at": timestamp,
                }
            ]
        )
        annotations = pd.concat([annotations, new_annotation], ignore_index=True)
        annotations["row_index"] = annotations["row_index"].astype(int)
        annotations["human_label"] = annotations["human_label"].astype(int)
        annotations.sort_values(["participant_id", "row_index"]).to_csv(
            OUTPUT_FILE, index=False
        )


@st.dialog("Emotion labelling tutorial", width="large", dismissible=False)
def show_tutorial() -> None:
    step = st.session_state.tutorial_step
    total_steps = 4
    st.progress((step + 1) / total_steps, text=f"Step {step + 1} of {total_steps}")

    if step == 0:
        st.subheader("Understand the task")
        st.write(
            "You will receive **five randomly selected tweets**. For each one, "
            "identify the **main emotion expressed by its author** and choose "
            "exactly one category."
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
            5. Complete all five tweets. You may go back and revise an answer.
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

if "participant_id" not in st.session_state:
    st.subheader("Participant sign-in")
    st.write(
        "Enter only your UMich uniqname—the part before `@umich.edu`. It will be "
        "stored with your five labels."
    )
    with st.form("participant_sign_in"):
        entered_id = st.text_input(
            "UMich uniqname",
            placeholder="For example: jsmith",
            max_chars=20,
        )
        sign_in = st.form_submit_button(
            "Continue", type="primary", use_container_width=True
        )

    if sign_in:
        participant_id = entered_id.strip().lower()
        if not participant_id:
            st.error("Please enter your UMich uniqname.")
        elif not participant_id.isalnum():
            st.error(
                "Enter only your uniqname, without `@umich.edu`, spaces, or symbols."
            )
        else:
            st.session_state.participant_id = participant_id
            st.session_state.assignment_position = 0
            st.session_state.tutorial_seen = False
            st.session_state.tutorial_open = True
            st.session_state.tutorial_step = 0
            st.rerun()
    st.stop()

participant_id = st.session_state.participant_id
assignment = get_assignment(participant_id, len(df))
try:
    annotations = load_annotations()
except Exception as error:
    st.error(
        "Could not connect to the private annotation sheet. Check the Google "
        f"Sheet sharing and Streamlit secrets. Details: {error}"
    )
    st.stop()
participant_annotations = annotations[
    (annotations["participant_id"] == participant_id)
    & (annotations["row_index"].isin(assignment))
]
labels_by_row = dict(
    zip(
        participant_annotations["row_index"].astype(int),
        participant_annotations["human_label"].astype(int),
    )
)

if "assignment_position" not in st.session_state:
    first_unlabelled = next(
        (position for position, row in enumerate(assignment) if row not in labels_by_row),
        0,
    )
    st.session_state.assignment_position = first_unlabelled

position = max(
    0, min(int(st.session_state.assignment_position), TWEETS_PER_PARTICIPANT - 1)
)
st.session_state.assignment_position = position
row_index = assignment[position]
labelled_count = len(labels_by_row)

st.progress(
    labelled_count / TWEETS_PER_PARTICIPANT,
    text=f"{labelled_count} of {TWEETS_PER_PARTICIPANT} tweets labelled",
)
if labelled_count == TWEETS_PER_PARTICIPANT:
    st.success(
        "You have completed all five tweets. Thank you! You may review and revise "
        "your labels below."
    )

st.subheader(f"Assigned tweet {position + 1} of {TWEETS_PER_PARTICIPANT}")
st.info(str(df.at[row_index, "text"]))

current_label = labels_by_row.get(row_index)
default_choice = EMOTIONS.get(current_label)
choice = st.radio(
    "Choose a category",
    list(EMOTIONS.values()),
    index=list(EMOTIONS.values()).index(default_choice) if default_choice else None,
    key=f"label_{participant_id}_{row_index}",
)

previous_column, save_column, next_column = st.columns(3)

with previous_column:
    if st.button("← Previous", use_container_width=True, disabled=position == 0):
        st.session_state.assignment_position -= 1
        st.rerun()

with save_column:
    save_text = "Save" if position == TWEETS_PER_PARTICIPANT - 1 else "Save & Next"
    if st.button(
        save_text,
        type="primary",
        use_container_width=True,
        disabled=choice is None,
    ):
        selected_label = next(
            label_id for label_id, emotion in EMOTIONS.items() if emotion == choice
        )
        try:
            save_annotation(participant_id, row_index, selected_label)
        except Exception as error:
            st.error(f"Could not save this annotation: {error}")
            st.stop()
        if position < TWEETS_PER_PARTICIPANT - 1:
            st.session_state.assignment_position += 1
        st.rerun()

with next_column:
    if st.button(
        "Next →",
        use_container_width=True,
        disabled=position == TWEETS_PER_PARTICIPANT - 1,
    ):
        st.session_state.assignment_position += 1
        st.rerun()

with st.sidebar:
    st.header("Participant")
    st.write(f"Signed in as **{participant_id}**")
    if st.button("Switch participant", use_container_width=True):
        del st.session_state.participant_id
        st.session_state.pop("assignment_position", None)
        st.session_state.tutorial_seen = False
        st.session_state.tutorial_open = True
        st.session_state.tutorial_step = 0
        st.rerun()

    st.divider()
    st.header("Help")
    if st.button("📘 View tutorial", use_container_width=True):
        st.session_state.tutorial_open = True
        st.session_state.tutorial_step = 0
        st.rerun()

    st.divider()
    if google_sheets_configured():
        st.success("Storage: private Google Sheet")
    else:
        st.warning("Storage: local CSV (development only)")
    st.caption(
        "Each record includes the uniqname, source tweet row, selected label, "
        "and timestamp."
    )

if st.session_state.tutorial_open:
    show_tutorial()