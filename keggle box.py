import os
import zipfile
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# ==========================
# ZIP File Path
# ==========================
zip_path = r"C:\Users\SAHAJ\Downloads\archive.zip"

# Check ZIP exists
if not os.path.exists(zip_path):
    raise FileNotFoundError(f"ZIP file not found:\n{zip_path}")

# Show files inside ZIP
with zipfile.ZipFile(zip_path, 'r') as z:
    print("Files inside ZIP:")
    for f in z.namelist():
        print(f)

    # Find dialogs.txt
    txt_file = None
    for f in z.namelist():
        if f.endswith("dialogs.txt"):
            txt_file = f
            break

    if txt_file is None:
        raise FileNotFoundError("dialogs.txt not found inside ZIP.")

    print("\nLoading:", txt_file)

    with z.open(txt_file) as file:
        df = pd.read_csv(
            file,
            sep="\t",
            names=["Question", "Answer"],
            encoding="utf-8"
        )

print("\nDataset Loaded Successfully!")
print(df.head())

# Check dataset size
print("\nRows:", len(df))
print("Columns:", len(df.columns))



import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'([^\w\s])', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text

df['Encoder Inputs'] = df['Question'].apply(clean_text)
df['Decoder Inputs'] = "<sos> " + df['Answer'].apply(clean_text) + " <eos>"
df['Decoder Targets'] = df['Answer'].apply(clean_text) + " <eos>"

print(df.head())




import plotly.express as px

fig1 = px.histogram(df, x='Question Length', nbins=50)
fig2 = px.histogram(df, x='Answer Length', nbins=50)

fig1.show()
fig2.show()

print("Maximum Question Length:", df['Question Length'].max())
print("Maximum Answer Length:", df['Answer Length'].max())