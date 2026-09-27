import pandas as pd
from rapidfuzz import process, fuzz

# Load development data
source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

# Clean text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return " ".join(text.split())


for df in [source1, source2, source3]:
    df["clean_name"] = df["business_name"].apply(clean_text)
    df["clean_address"] = df["business_address"].apply(clean_text)
    df["clean_country"] = df["country"].apply(clean_text)


# Create country indexes
source2_by_country = {}

for country, group in source2.groupby("clean_country"):
    source2_by_country[country] = group


source3_by_country = {}

for country, group in source3.groupby("clean_country"):
    source3_by_country[country] = group


candidate_rows = []

# Process Source 1
for _, row in source1.iterrows():

    s1_id = row["entity_id"]
    country = row["clean_country"]

    # -----------------------------
    # SOURCE 2
    # -----------------------------
    if country in source2_by_country:

        group = source2_by_country[country]

        name_choices = group["clean_name"].tolist()
        address_choices = group["clean_address"].tolist()

        # Existing top 50
        name_matches = process.extract(
            row["clean_name"],
            name_choices,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        address_matches = process.extract(
            row["clean_address"],
            address_choices,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        candidate_indexes = set()

        for _, score, index in name_matches:
            candidate_indexes.add(index)

        for _, score, index in address_matches:
            candidate_indexes.add(index)

        # NEW:
        # Add top 10 candidates using partial ratio
        partial_name_matches = process.extract(
            row["clean_name"],
            name_choices,
            scorer=fuzz.partial_ratio,
            limit=10,
            score_cutoff=60
        )

        for _, score, index in partial_name_matches:
            candidate_indexes.add(index)

        for index in candidate_indexes:
            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": group.iloc[index]["entity_id"],
                "source": "source2"
            })


    # -----------------------------
    # SOURCE 3
    # -----------------------------
    if country in source3_by_country:

        group = source3_by_country[country]

        name_choices = group["clean_name"].tolist()
        address_choices = group["clean_address"].tolist()

        # Existing top 50
        name_matches = process.extract(
            row["clean_name"],
            name_choices,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        address_matches = process.extract(
            row["clean_address"],
            address_choices,
            scorer=fuzz.token_set_ratio,
            limit=50,
            score_cutoff=40
        )

        candidate_indexes = set()

        for _, score, index in name_matches:
            candidate_indexes.add(index)

        for _, score, index in address_matches:
            candidate_indexes.add(index)

        # NEW:
        # Add top 10 candidates using partial ratio
        partial_name_matches = process.extract(
            row["clean_name"],
            name_choices,
            scorer=fuzz.partial_ratio,
            limit=10,
            score_cutoff=60
        )

        for _, score, index in partial_name_matches:
            candidate_indexes.add(index)

        for index in candidate_indexes:
            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": group.iloc[index]["entity_id"],
                "source": "source3"
            })


# Save
candidates = pd.DataFrame(candidate_rows)

candidates = candidates.drop_duplicates()

candidates.to_csv(
    "candidate_pairs_expanded.tsv",
    sep="\t",
    index=False
)

print("\nExpanded candidate generation complete.")
print("Total candidates:", len(candidates))
print("\nCandidates by source:")
print(candidates["source"].value_counts())

print("\nSaved: candidate_pairs_expanded.tsv")