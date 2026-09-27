import pandas as pd
from rapidfuzz import process, fuzz


# ============================================================
# 1. LOAD DATA
# ============================================================

source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

print("Data loaded:")
print("Source 1:", len(source1))
print("Source 2:", len(source2))
print("Source 3:", len(source3))


# ============================================================
# 2. CLEAN TEXT
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    return " ".join(text.split())


for df in [source1, source2, source3]:

    df["clean_name"] = df["business_name"].apply(clean_text)
    df["clean_address"] = df["business_address"].apply(clean_text)
    df["clean_country"] = df["country"].apply(clean_text)


# ============================================================
# 3. CREATE COUNTRY INDEX
# ============================================================

source2_by_country = {
    country: group
    for country, group in source2.groupby("clean_country")
}

source3_by_country = {
    country: group
    for country, group in source3.groupby("clean_country")
}


# ============================================================
# 4. CANDIDATE GENERATION
# ============================================================

candidate_rows = []

print("\nGenerating candidates...")

for counter, (_, row) in enumerate(source1.iterrows()):

    s1_id = row["entity_id"]
    country = row["clean_country"]

    # --------------------------------------------------------
    # SOURCE 2
    # --------------------------------------------------------

    if country in source2_by_country:

        group = source2_by_country[country]

        name_choices = group["clean_name"].tolist()
        address_choices = group["clean_address"].tolist()

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

        indexes = set()

        for _, score, index in name_matches:
            indexes.add(index)

        for _, score, index in address_matches:
            indexes.add(index)

        for index in indexes:

            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": group.iloc[index]["entity_id"],
                "source": "source2"
            })


    # --------------------------------------------------------
    # SOURCE 3
    # --------------------------------------------------------

    if country in source3_by_country:

        group = source3_by_country[country]

        name_choices = group["clean_name"].tolist()
        address_choices = group["clean_address"].tolist()

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

        indexes = set()

        for _, score, index in name_matches:
            indexes.add(index)

        for _, score, index in address_matches:
            indexes.add(index)

        for index in indexes:

            candidate_rows.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": group.iloc[index]["entity_id"],
                "source": "source3"
            })


    # Progress
    if (counter + 1) % 100 == 0:
        print("Processed Source 1:", counter + 1)


# Convert to DataFrame
candidates = pd.DataFrame(candidate_rows)

# Remove duplicates
candidates = candidates.drop_duplicates()

print("\nCandidate generation finished.")
print("Total candidates:", len(candidates))

print("\nCandidates by source:")
print(candidates["source"].value_counts())


# ============================================================
# 5. SAVE CANDIDATES
# ============================================================

candidates.to_csv(
    "candidate_pairs_final_dev.tsv",
    sep="\t",
    index=False
)

print("\nSaved: candidate_pairs_final_dev.tsv")


# ============================================================
# 6. CREATE LOOKUP DICTIONARIES
# ============================================================

source2_lookup = source2.set_index("entity_id").to_dict("index")
source3_lookup = source3.set_index("entity_id").to_dict("index")


# ============================================================
# 7. MATCH CANDIDATES
# ============================================================

matches = []

print("\nCalculating matching scores...")

for counter, (_, candidate) in enumerate(candidates.iterrows()):

    s1_id = candidate["source1_entity_id"]
    candidate_id = candidate["candidate_entity_id"]
    source = candidate["source"]

    # Find Source 1 record
    s1_row = source1[
        source1["entity_id"] == s1_id
    ].iloc[0]

    # Find candidate record
    if source == "source2":
        candidate_row = source2_lookup[candidate_id]
    else:
        candidate_row = source3_lookup[candidate_id]

    # Similarity scores
    name_score = fuzz.token_set_ratio(
        s1_row["clean_name"],
        candidate_row["clean_name"]
    )

    address_score = fuzz.token_set_ratio(
        s1_row["clean_address"],
        candidate_row["clean_address"]
    )

    # Weighted score
    weighted_score = (
        0.4 * name_score +
        0.6 * address_score
    )

    # --------------------------------------------------------
    # CURRENT BEST MATCHING RULE
    # --------------------------------------------------------

    is_match = (
        weighted_score >= 80
        or
        (
            name_score >= 95
            and
            address_score >= 60
        )
    )

    if is_match:

        matches.append({
            "source1_entity_id": s1_id,
            "matched_entity_id": candidate_id,
            "source": source,
            "name_score": name_score,
            "address_score": address_score,
            "weighted_score": weighted_score
        })

    if (counter + 1) % 10000 == 0:
        print("Scored candidates:", counter + 1)


# ============================================================
# 8. SAVE MATCHES
# ============================================================

matches_df = pd.DataFrame(matches)

print("\nMatching finished.")
print("Total predicted matches:", len(matches_df))

matches_df.to_csv(
    "matching_results_dev.tsv",
    sep="\t",
    index=False
)

print("\nSaved: matching_results_dev.tsv")

print("\nDONE!")