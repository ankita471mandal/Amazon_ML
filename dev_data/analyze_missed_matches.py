import pandas as pd


# =======================================
# 1. Load match features
# =======================================

df = pd.read_csv(
    "match_features.tsv",
    sep="\t"
)


# =======================================
# 2. Calculate weighted score
# =======================================

df["weighted_score"] = (
    0.4 * df["name_score"]
    +
    0.6 * df["address_score"]
)


# =======================================
# 3. Find true matches missed by rule
# =======================================

missed = df[
    (df["is_match"] == 1)
    &
    (df["weighted_score"] < 80)
].copy()


# =======================================
# 4. Print summary
# =======================================

print("\n================================")
print("MISSED TRUE MATCH ANALYSIS")
print("================================")

print(
    "Total true matches:",
    (df["is_match"] == 1).sum()
)

print(
    "True matches missed by rule:",
    len(missed)
)


print("\nAverage scores of missed matches:")

print(
    missed[
        ["name_score", "address_score", "weighted_score"]
    ].mean()
)


# =======================================
# 5. Score ranges
# =======================================

print("\nMinimum / maximum scores:")

print(
    missed[
        ["name_score", "address_score", "weighted_score"]
    ].describe()
)


# =======================================
# 6. Show lowest scoring examples
# =======================================

print("\nLowest scoring missed matches:")

print(
    missed.sort_values(
        "weighted_score"
    )[
        [
            "source1_id",
            "candidate_id",
            "source",
            "name_score",
            "address_score",
            "weighted_score"
        ]
    ].head(20).to_string(index=False)
)


# =======================================
# 7. Save missed matches
# =======================================

missed.to_csv(
    "missed_true_matches.tsv",
    sep="\t",
    index=False
)

print("\nFile created:")
print("missed_true_matches.tsv")