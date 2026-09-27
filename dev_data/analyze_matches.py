import pandas as pd
from rapidfuzz import fuzz

print("Loading files...")


# =======================================
# 1. Load data
# =======================================

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

ground_truth = pd.read_csv(
    "ground_truth.tsv",
    sep="\t"
)

candidates = pd.read_csv(
    "candidate_pairs_dev.tsv",
    sep="\t"
)

print("Candidates loaded:", len(candidates))


# =======================================
# 2. Clean text
# =======================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    return text.strip()


source1["name_clean"] = source1["business_name"].apply(clean_text)
source1["address_clean"] = source1["business_address"].apply(clean_text)

source2["name_clean"] = source2["business_name"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)

source3["name_clean"] = source3["business_name"].apply(clean_text)
source3["address_clean"] = source3["business_address"].apply(clean_text)


# =======================================
# 3. Create lookup tables
# =======================================

s1_lookup = source1.set_index(
    "entity_id"
).to_dict("index")

s2_lookup = source2.set_index(
    "entity_id"
).to_dict("index")

s3_lookup = source3.set_index(
    "entity_id"
).to_dict("index")


# =======================================
# 4. Create true-match set
# =======================================

true_matches = set()

for _, row in ground_truth.iterrows():

    s1_id = row["source1_entity_id"]

    if pd.isna(row["matched_entity_ids"]):
        continue

    for entity_id in str(
        row["matched_entity_ids"]
    ).split(","):

        entity_id = entity_id.strip()

        true_matches.add(
            (s1_id, entity_id)
        )


print("True match pairs loaded:", len(true_matches))


# =======================================
# 5. Calculate similarity features
# =======================================

results = []


for count, (_, candidate) in enumerate(
    candidates.iterrows(),
    start=1
):

    s1_id = candidate["source1_entity_id"]
    candidate_id = candidate["candidate_entity_id"]
    source = candidate["source"]


    # Get Source 1 record
    s1 = s1_lookup[s1_id]


    # Get candidate record
    if source == "source2":

        match = s2_lookup[candidate_id]

    else:

        match = s3_lookup[candidate_id]


    # Name similarity
    name_score = fuzz.token_set_ratio(
        s1["name_clean"],
        match["name_clean"]
    )


    # Address similarity
    address_score = fuzz.token_set_ratio(
        s1["address_clean"],
        match["address_clean"]
    )


    # Check ground truth
    is_match = int(
        (s1_id, candidate_id)
        in true_matches
    )


    results.append({

        "source1_id": s1_id,

        "candidate_id": candidate_id,

        "source": source,

        "name_score": name_score,

        "address_score": address_score,

        "is_match": is_match
    })


    if count % 20000 == 0:

        print(
            "Processed:",
            count,
            "/",
            len(candidates)
        )


# =======================================
# 6. Create DataFrame
# =======================================

results_df = pd.DataFrame(results)


# =======================================
# 7. Save
# =======================================

results_df.to_csv(
    "match_features.tsv",
    sep="\t",
    index=False
)


# =======================================
# 8. Summary
# =======================================

print()
print("================================")
print("MATCH FEATURE ANALYSIS")
print("================================")

print(
    "Total candidates:",
    len(results_df)
)

print(
    "True matches:",
    results_df["is_match"].sum()
)

print(
    "Non-matches:",
    len(results_df)
    - results_df["is_match"].sum()
)


print("\nAverage scores:")

print(
    results_df
    .groupby("is_match")[[
        "name_score",
        "address_score"
    ]]
    .mean()
)


print()
print("File created:")
print("match_features.tsv")