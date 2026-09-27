import pandas as pd
import re
from rapidfuzz import process, fuzz


print("Loading data...")


# =======================================
# 1. Load data
# =======================================

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
ground_truth = pd.read_csv("ground_truth.tsv", sep="\t")

print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Ground Truth:", len(ground_truth))


# =======================================
# 2. Clean text
# =======================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


source1["name_clean"] = source1["business_name"].apply(clean_text)
source2["name_clean"] = source2["business_name"].apply(clean_text)

source1["address_clean"] = source1["business_address"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)

source1["country_clean"] = source1["country"].apply(clean_text)
source2["country_clean"] = source2["country"].apply(clean_text)


# =======================================
# 3. Group Source 2 by country
# =======================================

country_groups = {}

for country, group in source2.groupby("country_clean"):
    country_groups[country] = group


# =======================================
# 4. Evaluation variables
# =======================================

total_true_matches = 0
found_true_matches = 0

results = []


# =======================================
# 5. Process every Source 1
# =======================================

for count, (_, s1) in enumerate(
    source1.iterrows(),
    start=1
):

    # -----------------------------------
    # Get same-country Source 2 records
    # -----------------------------------

    same_country = country_groups.get(
        s1["country_clean"],
        pd.DataFrame()
    )

    if same_country.empty:
        continue


    # -----------------------------------
    # NAME candidates
    # -----------------------------------

    names = same_country["name_clean"].tolist()

    name_matches = process.extract(
        s1["name_clean"],
        names,
        scorer=fuzz.token_set_ratio,
        limit=50,
        score_cutoff=40
    )


    # -----------------------------------
    # ADDRESS candidates
    # -----------------------------------

    addresses = same_country["address_clean"].tolist()

    address_matches = process.extract(
        s1["address_clean"],
        addresses,
        scorer=fuzz.token_set_ratio,
        limit=50,
        score_cutoff=40
    )


    # -----------------------------------
    # Combine candidates
    # -----------------------------------

    candidate_ids = set()


    # Name candidates
    for _, score, index in name_matches:

        candidate_id = same_country.iloc[index]["entity_id"]

        candidate_ids.add(candidate_id)


    # Address candidates
    for _, score, index in address_matches:

        candidate_id = same_country.iloc[index]["entity_id"]

        candidate_ids.add(candidate_id)


    # -----------------------------------
    # Get ground-truth matches
    # -----------------------------------

    gt_row = ground_truth[
        ground_truth["source1_entity_id"]
        == s1["entity_id"]
    ]


    true_ids = set()


    if not gt_row.empty:

        matched_ids = gt_row.iloc[0]["matched_entity_ids"]

        if pd.notna(matched_ids):

            for entity_id in str(matched_ids).split(","):

                entity_id = entity_id.strip()

                if entity_id.startswith("S2-"):

                    true_ids.add(entity_id)


    # -----------------------------------
    # Compare candidates with truth
    # -----------------------------------

    found = true_ids.intersection(candidate_ids)

    total_true_matches += len(true_ids)

    found_true_matches += len(found)


    # -----------------------------------
    # Save result
    # -----------------------------------

    results.append({
        "source1_id": s1["entity_id"],
        "true_matches": len(true_ids),
        "candidates": len(candidate_ids),
        "found_true_matches": len(found)
    })


    # -----------------------------------
    # Progress
    # -----------------------------------

    if count % 100 == 0:

        print(
            "Processed:",
            count,
            "/",
            len(source1)
        )


# =======================================
# 6. Calculate recall
# =======================================

if total_true_matches > 0:

    recall = (
        found_true_matches /
        total_true_matches
    ) * 100

else:

    recall = 0


# =======================================
# 7. Final result
# =======================================

print("\n================================")
print("CANDIDATE GENERATION EVALUATION")
print("================================")

print(
    "Total true matches:",
    total_true_matches
)

print(
    "True matches found:",
    found_true_matches
)

print(
    "Candidate recall:",
    round(recall, 2),
    "%"
)


# =======================================
# 8. Save evaluation
# =======================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "candidate_evaluation.tsv",
    sep="\t",
    index=False
)

print("\nEvaluation file created:")
print("candidate_evaluation.tsv")