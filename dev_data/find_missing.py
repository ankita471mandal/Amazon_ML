import pandas as pd
import re
from rapidfuzz import process, fuzz


# =======================================
# 1. Load data
# =======================================

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")
ground_truth = pd.read_csv("ground_truth.tsv", sep="\t")


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


# Source 1
source1["name_clean"] = source1["business_name"].apply(clean_text)
source1["address_clean"] = source1["business_address"].apply(clean_text)
source1["country_clean"] = source1["country"].apply(clean_text)

# Source 2
source2["name_clean"] = source2["business_name"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)
source2["country_clean"] = source2["country"].apply(clean_text)

# Source 3
source3["name_clean"] = source3["business_name"].apply(clean_text)
source3["address_clean"] = source3["business_address"].apply(clean_text)
source3["country_clean"] = source3["country"].apply(clean_text)


# =======================================
# 3. Group by country
# =======================================

country_groups_2 = {}

for country, group in source2.groupby("country_clean"):
    country_groups_2[country] = group


country_groups_3 = {}

for country, group in source3.groupby("country_clean"):
    country_groups_3[country] = group


# =======================================
# 4. Find missing matches
# =======================================

missing_matches = []


for count, (_, s1) in enumerate(source1.iterrows(), start=1):

    # ===================================
    # SOURCE 2
    # ===================================

    same_country_2 = country_groups_2.get(
        s1["country_clean"],
        pd.DataFrame()
    )

    candidate_ids_2 = set()

    if not same_country_2.empty:

        # Name candidates
        names_2 = same_country_2["name_clean"].tolist()

        name_matches_2 = process.extract(
            s1["name_clean"],
            names_2,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        # Address candidates
        addresses_2 = same_country_2["address_clean"].tolist()

        address_matches_2 = process.extract(
            s1["address_clean"],
            addresses_2,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        # Add name candidates
        for _, score, index in name_matches_2:

            candidate_ids_2.add(
                same_country_2.iloc[index]["entity_id"]
            )

        # Add address candidates
        for _, score, index in address_matches_2:

            candidate_ids_2.add(
                same_country_2.iloc[index]["entity_id"]
            )


    # ===================================
    # SOURCE 3
    # ===================================

    same_country_3 = country_groups_3.get(
        s1["country_clean"],
        pd.DataFrame()
    )

    candidate_ids_3 = set()

    if not same_country_3.empty:

        # Name candidates
        names_3 = same_country_3["name_clean"].tolist()

        name_matches_3 = process.extract(
            s1["name_clean"],
            names_3,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        # Address candidates
        addresses_3 = same_country_3["address_clean"].tolist()

        address_matches_3 = process.extract(
            s1["address_clean"],
            addresses_3,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        # Add name candidates
        for _, score, index in name_matches_3:

            candidate_ids_3.add(
                same_country_3.iloc[index]["entity_id"]
            )

        # Add address candidates
        for _, score, index in address_matches_3:

            candidate_ids_3.add(
                same_country_3.iloc[index]["entity_id"]
            )


    # ===================================
    # Get true matches
    # ===================================

    gt_row = ground_truth[
        ground_truth["source1_entity_id"]
        == s1["entity_id"]
    ]

    if gt_row.empty:
        continue

    matched_ids = gt_row.iloc[0]["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue


    # ===================================
    # Check every true match
    # ===================================

    for entity_id in str(matched_ids).split(","):

        entity_id = entity_id.strip()


        # --------------------------------
        # Source 2
        # --------------------------------

        if entity_id.startswith("S2-"):

            if entity_id not in candidate_ids_2:

                missing_matches.append({
                    "source1_id": s1["entity_id"],
                    "source": "source2",
                    "source1_name": s1["business_name"],
                    "source1_address": s1["business_address"],
                    "source1_country": s1["country"],
                    "missing_entity_id": entity_id
                })


        # --------------------------------
        # Source 3
        # --------------------------------

        elif entity_id.startswith("S3-"):

            if entity_id not in candidate_ids_3:

                missing_matches.append({
                    "source1_id": s1["entity_id"],
                    "source": "source3",
                    "source1_name": s1["business_name"],
                    "source1_address": s1["business_address"],
                    "source1_country": s1["country"],
                    "missing_entity_id": entity_id
                })


    # Progress
    if count % 100 == 0:
        print("Processed:", count, "/", len(source1))


# =======================================
# 5. Show missing matches
# =======================================

print("\n================================")
print("MISSING TRUE MATCHES")
print("================================")

print(
    "Number of missing matches:",
    len(missing_matches)
)


for item in missing_matches:

    print("\nSource 1 ID:",
          item["source1_id"])

    print("Source:",
          item["source"])

    print("Source 1 name:",
          item["source1_name"])

    print("Source 1 address:",
          item["source1_address"])

    print("Country:",
          item["source1_country"])

    print("Missing entity:",
          item["missing_entity_id"])


# =======================================
# 6. Save
# =======================================

pd.DataFrame(missing_matches).to_csv(
    "missing_matches.tsv",
    sep="\t",
    index=False
)

print("\nFile created: missing_matches.tsv")