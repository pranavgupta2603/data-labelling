# Emotion Labelling Dashboard

## Use the app

Open the deployed dashboard:

**[https://gpranavdatalabel.streamlit.app/](https://gpranavdatalabel.streamlit.app/)**

1. Enter your UMich uniqname without `@umich.edu`.
2. Complete the short step-by-step tutorial.
3. Read each of your five randomly assigned tweets.
4. Select the emotion that best represents the author's main feeling.
5. Click **Save & Next** to record each answer.
6. Complete all five tweets. You can use **Previous** to review or revise a label.

Your assigned tweets remain the same if you return using the same uniqname.

## Run locally

```bash
uv run streamlit run da.py
```

Without Google credentials, annotations are written to the ignored local file
`emotion_annotations.csv`.

The app creates the header row automatically and stores one current label for
each participant and tweet. Each record contains the participant uniqname,
source row, selected label, and UTC timestamp.
