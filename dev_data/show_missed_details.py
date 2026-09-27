import pandas as pd

# Load data
source1 = pd.read_csv("source1.tsv", sep="\t")
source2 = pd.read_csv("source2.tsv", sep="\t")
source3 = pd.read_csv("source3.tsv", sep="\t")

missed = pd.read_csv(
    "missed_matches_current_rule.tsv",
    sep="\t"
)

# Make IDs strings
source1["entity_id"] = source1["entity_id"].astype(str)
source2["entity_id"] = source2["entity_id"].astype(str)
source3["entity_id"] = source3["entity_id"].astype(str)

# Take first 20 missed matches
sample = missed.head(20)

print("\n========== MISSED TRUE MATCHES ==========\n")

for _, row in sample.iterrows():

    s1 = source1[
        source1["entity_id"] == row["source1_id"]
    ]

    if row["source"] == "source2":
        candidate = source2[
            source2["entity_id"] == row["candidate_id"]
        ]
    else:
        candidate = source3[
            source3["entity_id"] == row["candidate_id"]
        ]

    if len(s1) == 0 or len(candidate) == 0:
        continue

    s1 = s1.iloc[0]
    candidate = candidate.iloc[0]

    print("-" * 70)

    print("Source 1 ID:", row["source1_id"])
    print("Candidate ID:", row["candidate_id"])
    print("Source:", row["source"])

    print("\nSOURCE 1")
    print("Name   :", s1["business_name"])
    print("Address:", s1["business_address"])
    print("Country:", s1["country"])

    print("\nCANDIDATE")
    print("Name   :", candidate["business_name"])
    print("Address:", candidate["business_address"])
    print("Country:", candidate["country"])

    print("\nScores")
    print("Name score   :", round(row["name_score"], 2))
    print("Address score:", round(row["address_score"], 2))
    print("Weighted     :", round(row["weighted_score"], 2))

print("\nDone.")