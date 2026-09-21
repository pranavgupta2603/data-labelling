# Emotion Labelling Dashboard

## Run locally

```bash
uv run streamlit run da.py
```

Without Google credentials, annotations are written to the ignored local file
`emotion_annotations.csv`.

## Configure persistent Google Sheets storage

1. Create a private Google Sheet and rename its first tab to `annotations`.
2. In Google Cloud, enable the Google Sheets API and Google Drive API.
3. Create a service account and download its JSON key.
4. Share the private sheet with the service account's `client_email` as an Editor.
5. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and replace
   the placeholders with the sheet URL and JSON key values.
6. In Streamlit Community Cloud, open the app's **Settings → Secrets** and paste
   the same TOML configuration.

The app creates the header row automatically and appends every submitted label.
Each record contains the participant uniqname, source row, selected label, and
UTC timestamp.
