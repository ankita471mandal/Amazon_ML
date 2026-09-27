import pandas as pd
import re
from difflib import SequenceMatcher


# ---------------------------------------
# 1. Load development data
# ---------------------------------------

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")

print("Data loaded!")
print("Source 1:", len(source1))
print("Source 2:", len(source2))


# ---------------------------------------
# 2. Clean text
# ---------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


source1["name_clean"] = source1["business_name"].apply(clean_text)
source1["address_clean"] = source1["business_address"].apply(clean_text)
source1["country_clean"] = source1["country"].apply(clean_text)

source2["name_clean"] = source2["business_name"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)
source2["country_clean"] = source2["country"].apply(clean_text)


# ---------------------------------------
# 3. Similarity function
# ---------------------------------------

def similarity(text1, text2):

    if not text1 or not text2:
        return 0

    return SequenceMatcher(None, text1, text2).ratio()


# ---------------------------------------
# 4. Candidate generation
# ---------------------------------------

def generate_candidates(s1, source2, top_k=20):

    candidates = []

    for _, s2 in source2.iterrows():

        # First filter: country must match
        if s1["country_clean"] != s2["country_clean"]:
            continue

        # Compare business names
        name_score = similarity(
            s1["name_clean"],
            s2["name_clean"]
        )

        # Keep reasonably similar names
        if name_score >= 0.40:

            candidates.append({
                "source1_id": s1["entity_id"],
                "source2_id": s2["entity_id"],
                "name_score": name_score
            })

    # Convert to DataFrame
    candidates_df = pd.DataFrame(candidates)

    # If no candidates found
    if candidates_df.empty:
        return candidates_df

    # Keep only top candidates
    candidates_df = candidates_df.sort_values(
        "name_score",
        ascending=False
    )

    candidates_df = candidates_df.head(top_k)

    return candidates_df


# ---------------------------------------
# 5. Test with ONE Source 1 business
# ---------------------------------------

s1 = source1.iloc[0]

print("\n--------------------------------")
print("Testing candidate generation")
print("--------------------------------")

print("Business:", s1["business_name"])
print("Address:", s1["business_address"])
print("Country:", s1["country"])


candidates = generate_candidates(
    s1,
    source2,
    top_k=20
)


# ---------------------------------------
# 6. Display candidates
# ---------------------------------------

print("\nNumber of candidates:", len(candidates))

print("\nCandidates:")

print(
    candidates.to_string(index=False)
)


# ---------------------------------------
# 7. Save candidates
# ---------------------------------------

candidates.to_csv(
    "test_candidates.tsv",
    sep="\t",
    index=False
)

print("\nCandidate file created:")
print("test_candidates.tsv")