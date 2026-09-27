import pandas as pd

source1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
source2 = pd.read_csv("dataset/train/train_source2.tsv", sep="\t")
source3 = pd.read_csv("dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t")

print("Source 1 shape:", source1.shape)
print("Source 2 shape:", source2.shape)
print("Source 3 shape:", source3.shape)
print("Ground Truth shape:", ground_truth.shape)

print("\nSource 1 columns:")
print(source1.columns.tolist())

print("\nSource 2 columns:")
print(source2.columns.tolist())

print("\nSource 3 columns:")
print(source3.columns.tolist())

print("\nGround Truth columns:")
print(ground_truth.columns.tolist())

print("\n--- Source 1 ---")
print(source1.head())

print("\n--- Source 2 ---")
print(source2.head())

print("\n--- Source 3 ---")
print(source3.head())

print("\n--- Ground Truth ---")
print(ground_truth.head())

# Number of matches for each Source 1 entity
ground_truth["match_count"] = (
    ground_truth["matched_entity_ids"]
    .fillna("")
    .apply(lambda x: 0 if x == "" else len(x.split(",")))
)

print("\n--- Match Count Distribution ---")
print(ground_truth["match_count"].value_counts().sort_index())

print("\nAverage matches per Source 1 entity:",
      ground_truth["match_count"].mean())

print("\nMaximum matches for one Source 1 entity:",
      ground_truth["match_count"].max())
print("\n--- Missing Values ---")

for name, df in [
    ("Source 1", source1),
    ("Source 2", source2),
    ("Source 3", source3)
]:
    print(f"\n{name}")
    print(df.isna().sum())

import re

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


for df in [source1, source2, source3]:

    df["name_clean"] = df["business_name"].apply(clean_text)
    df["address_clean"] = df["business_address"].apply(clean_text)


print("\n--- Cleaned Source 1 ---")
print(
    source1[
        ["business_name", "name_clean",
         "business_address", "address_clean"]
    ].head()
)