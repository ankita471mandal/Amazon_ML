import pandas as pd

# Load candidate features
df = pd.read_csv("match_features.tsv", sep="\t")

# Calculate weighted score
df["weighted_score"] = (
    0.4 * df["name_score"] +
    0.6 * df["address_score"]
)

# True matches that our current rule misses
missed = df[
    (df["is_match"] == 1) &
    (df["weighted_score"] < 80)
].copy()

# Sort from highest score to lowest
missed = missed.sort_values(
    "weighted_score",
    ascending=False
)

print("\nTotal missed matches:", len(missed))

print("\nFirst 30 missed matches:\n")

print(
    missed[
        [
            "source1_id",
            "candidate_id",
            "source",
            "name_score",
            "address_score",
            "weighted_score"
        ]
    ].head(30).to_string(index=False)
)

# Save for later inspection
missed.to_csv(
    "missed_matches_current_rule.tsv",
    sep="\t",
    index=False
)

print("\nSaved: missed_matches_current_rule.tsv")