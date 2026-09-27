import pandas as pd
import re
from rapidfuzz import process, fuzz

print("Starting scalable candidate generation...")


# =======================================
# 1. Load data
# =======================================

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

print("Source 1 loaded:", len(source1))
print("Source 2 loaded:", len(source2))
print("Source 3 loaded:", len(source3))


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


# Source 2
source2["name_clean"] = source2["business_name"].apply(clean_text)
source2["address_clean"] = source2["business_address"].apply(clean_text)
source2["country_clean"] = source2["country"].apply(clean_text)

print("Source 2 cleaning completed!")


# Source 1
source1["name_clean"] = source1["business_name"].apply(clean_text)
source1["address_clean"] = source1["business_address"].apply(clean_text)
source1["country_clean"] = source1["country"].apply(clean_text)

print("Source 1 cleaning completed!")


# Source 3
source3["name_clean"] = source3["business_name"].apply(clean_text)
source3["address_clean"] = source3["business_address"].apply(clean_text)
source3["country_clean"] = source3["country"].apply(clean_text)

print("Source 3 cleaning completed!")


# =======================================
# 3. Source 2 country index
# =======================================

country_groups = {}

for country, group in source2.groupby("country_clean"):
    country_groups[country] = group

print("Country index created!")
print("Number of countries:", len(country_groups))


# =======================================
# 4. Source 3 country index
# =======================================

country_groups_3 = {}

for country, group in source3.groupby("country_clean"):
    country_groups_3[country] = group

print("Source 3 country index created!")
print("Source 3 countries:", len(country_groups_3))


# =======================================
# 5. Generate candidates
# =======================================

all_candidates = []

total = len(source1)


for count, (_, s1) in enumerate(
    source1.iterrows(),
    start=1
):

    # ===================================
    # SOURCE 2
    # ===================================

    same_country_2 = country_groups.get(
        s1["country_clean"],
        pd.DataFrame()
    )

    candidate_ids_2 = set()


    if not same_country_2.empty:

        # -------------------------------
        # Name matching
        # -------------------------------

        names_2 = same_country_2[
            "name_clean"
        ].tolist()

        name_matches_2 = process.extract(
            s1["name_clean"],
            names_2,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )


        # -------------------------------
        # Address matching
        # -------------------------------

        addresses_2 = same_country_2[
            "address_clean"
        ].tolist()

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


    # Save Source 2 candidates

    for candidate_id in candidate_ids_2:

        all_candidates.append({

            "source1_entity_id":
                s1["entity_id"],

            "candidate_entity_id":
                candidate_id,

            "source":
                "source2"
        })


    # ===================================
    # SOURCE 3
    # ===================================

    same_country_3 = country_groups_3.get(
        s1["country_clean"],
        pd.DataFrame()
    )

    candidate_ids_3 = set()


    if not same_country_3.empty:

        # -------------------------------
        # Name matching
        # -------------------------------

        names_3 = same_country_3[
            "name_clean"
        ].tolist()

        name_matches_3 = process.extract(
            s1["name_clean"],
            names_3,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )


        # -------------------------------
        # Address matching
        # -------------------------------

        addresses_3 = same_country_3[
            "address_clean"
        ].tolist()

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


    # Save Source 3 candidates

    for candidate_id in candidate_ids_3:

        all_candidates.append({

            "source1_entity_id":
                s1["entity_id"],

            "candidate_entity_id":
                candidate_id,

            "source":
                "source3"
        })


    # ===================================
    # Progress
    # ===================================

    if count % 100 == 0:

        print(
            "Processed:",
            count,
            "/",
            total
        )


# =======================================
# 6. Save candidate pairs
# =======================================

candidate_df = pd.DataFrame(
    all_candidates,
    columns=[
        "source1_entity_id",
        "candidate_entity_id",
        "source"
    ]
)


candidate_df.to_csv(
    "candidate_pairs_dev.tsv",
    sep="\t",
    index=False
)


# =======================================
# 7. Final information
# =======================================

print()
print("================================")
print("CANDIDATE GENERATION COMPLETE")
print("================================")

print(
    "Total candidate pairs:",
    len(candidate_df)
)

print(
    "Source 1 businesses:",
    len(source1)
)

print()
print("Candidates by source:")

print(
    candidate_df["source"].value_counts()
)

print()
print("File created:")
print("candidate_pairs_dev.tsv")