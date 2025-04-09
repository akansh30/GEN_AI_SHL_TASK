import pandas as pd
import re

def preprocess(text):
    if pd.isna(text):
        return ""
    text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
    return text.strip().lower()

def load_and_clean(path='shl_assessments_full.csv'):
    df = pd.read_csv(path)

    # Clean description column
    df['Cleaned Description'] = df['Description'].apply(preprocess)

    # Fill missing values to avoid NaN issues in f-string
    df['Assessment Name'] = df['Assessment Name'].fillna("")
    df['Assessment Length'] = df['Assessment Length'].fillna("Unknown")
    df['Test Types'] = df['Test Types'].fillna("Unknown")

    # Combine fields into one string for embedding
    df['Text for Embedding'] = df.apply(
        lambda row: f"{row['Assessment Name']}. {row['Cleaned Description']}. Duration: {row['Assessment Length']} mins. Test Type: {row['Test Types']}",
        axis=1
    )
    return df
