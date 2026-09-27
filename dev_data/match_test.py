import pandas as pd
import re
from difflib import SequenceMatcher


# ---------------------------------------
# 1. Load development data
# ---------------------------------------

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

print("Data loaded!")
print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ---------------------------------------
# 2. Clean text
# ---------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# 3. Prepare cleaned columns
# ---------------------------------------

for df in [source1, source2, source3]:

    df["name_clean"] = df["business_name"].apply(clean_text)
    df["address_clean"] = df["business_address"].apply(clean_text)
    df["country_clean"] = df["country"].apply(clean_text)


print("Text cleaning completed!")


# ---------------------------------------
# 4. Similarity function
# ---------------------------------------

def similarity(text1, text2):

    if not text1 or not text2:
        return 0

    return SequenceMatcher(None, text1, text2).ratio()


# ---------------------------------------
# 5. Test with ONE Source 1 business
# ---------------------------------------

s1 = source1.iloc[0]

print("\n--------------------------------")
print("Testing first Source 1 business")
print("--------------------------------")

print("Business name:", s1["business_name"])
print("Address:", s1["business_address"])
print("Country:", s1["country"])


# ---------------------------------------
# 6. Compare with Source 2
# ---------------------------------------

results = []

for _, s2 in source2.iterrows():

    # Country must match
    if s1["country_clean"] != s2["country_clean"]:
        continue

    name_score = similarity(
        s1["name_clean"],
        s2["name_clean"]
    )

    address_score = similarity(
        s1["address_clean"],
        s2["address_clean"]
    )

    # Combined score
    total_score = (
        0.6 * name_score +
        0.4 * address_score
    )

    results.append({
        "source1_id": s1["entity_id"],
        "source2_id": s2["entity_id"],
        "name_score": name_score,
        "address_score": address_score,
        "total_score": total_score
    })


# ---------------------------------------
# 7. Show best matches
# ---------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "total_score",
    ascending=False
)

print("\nTop 10 possible matches:")
print(
    results_df.head(10).to_string(index=False)
)