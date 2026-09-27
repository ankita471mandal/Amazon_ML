import pandas as pd

print("Loading files...")

# Load candidate pairs
candidates = pd.read_csv(
    "candidate_pairs_optimized.tsv",
    sep="\t"
)
# Load ground truth
ground_truth = pd.read_csv(
    "ground_truth.tsv",
    sep="\t"
)

print("Candidate pairs:", len(candidates))
print("Ground truth rows:", len(ground_truth))


# ---------------------------------------
# Create candidate sets
# ---------------------------------------

candidate_set_s2 = set()
candidate_set_s3 = set()

for _, row in candidates.iterrows():

    pair = (
        row["source1_entity_id"],
        row["candidate_entity_id"]
    )

    if row["source"] == "source2":
        candidate_set_s2.add(pair)

    elif row["source"] == "source3":
        candidate_set_s3.add(pair)


# ---------------------------------------
# Check true matches
# ---------------------------------------

total_s2_matches = 0
found_s2_matches = 0

total_s3_matches = 0
found_s3_matches = 0


for _, row in ground_truth.iterrows():

    source1_id = row["source1_entity_id"]

    matched_ids = row["matched_entity_ids"]

    if pd.isna(matched_ids):
        continue

    for entity_id in str(matched_ids).split(","):

        entity_id = entity_id.strip()

        pair = (
            source1_id,
            entity_id
        )

        # -------------------------------
        # Source 2
        # -------------------------------

        if entity_id.startswith("S2-"):

            total_s2_matches += 1

            if pair in candidate_set_s2:
                found_s2_matches += 1


        # -------------------------------
        # Source 3
        # -------------------------------

        elif entity_id.startswith("S3-"):

            total_s3_matches += 1

            if pair in candidate_set_s3:
                found_s3_matches += 1


# ---------------------------------------
# Calculate recall
# ---------------------------------------

recall_s2 = (
    found_s2_matches / total_s2_matches
) * 100

recall_s3 = (
    found_s3_matches / total_s3_matches
) * 100


# ---------------------------------------
# Print results
# ---------------------------------------

print("\n================================")
print("CANDIDATE RECALL CHECK")
print("================================")

print("\nSOURCE 2")
print("--------------------------------")
print("Total true S2 matches:", total_s2_matches)
print("True S2 matches found:", found_s2_matches)
print("S2 candidate recall:", round(recall_s2, 2), "%")


print("\nSOURCE 3")
print("--------------------------------")
print("Total true S3 matches:", total_s3_matches)
print("True S3 matches found:", found_s3_matches)
print("S3 candidate recall:", round(recall_s3, 2), "%")