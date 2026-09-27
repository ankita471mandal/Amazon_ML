import pandas as pd
import re
from rapidfuzz import process, fuzz

print("Starting fast candidate generation...")


# ---------------------------------------
# 1. Load development data
# ---------------------------------------

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")

print("Data loaded!")
print("Source 1:", len(source1))
print("Source 2:", len(source2))


# ---------------------------------------
# 2. Clean business names
# ---------------------------------------

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


source1["name_clean"] = source1["business_name"].apply(clean_text)
source2["name_clean"] = source2["business_name"].apply(clean_text)

source1["country_clean"] = source1["country"].apply(clean_text)
source2["country_clean"] = source2["country"].apply(clean_text)

source1["address_clean"] = source1["business_address"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)


print("Cleaning completed!")

# ---------------------------------------
# 3. Create a list of Source 2 names
# ---------------------------------------

# ---------------------------------------
# 3. Keep only Source 2 records
#    from the same country
# ---------------------------------------

s1 = source1.iloc[0]

same_country = source2[
    source2["country_clean"] == s1["country_clean"]
].copy()

print("\nSource 2 records with same country:",
      len(same_country))


# Create name list from same-country records
source2_names = same_country["name_clean"].tolist()


matches = process.extract(
    s1["name_clean"],
    source2_names,
    scorer=fuzz.token_set_ratio,
    limit=20,
    score_cutoff=40
)


# ---------------------------------------
# 5. Display candidates
# ---------------------------------------

print("\n--------------------------------")
print("Fast candidate generation test")
print("--------------------------------")

print("Source 1 business:", s1["business_name"])
print("Source 1 address:", s1["business_address"])
print("Cleaned address:", s1["address_clean"])

print("\nCandidates found:", len(matches))

for match in matches:

    name, name_score, index = match

    candidate = same_country.iloc[index]

    address_score = fuzz.token_set_ratio(
        s1["address_clean"],
        candidate["address_clean"]
    )

    final_score = (
        0.6 * name_score +
        0.4 * address_score
    )

    print(
        candidate["entity_id"],
        "| name:",
        round(name_score, 2),
        "| address:",
        round(address_score, 2),
        "| final:",
        round(final_score, 2)
    )