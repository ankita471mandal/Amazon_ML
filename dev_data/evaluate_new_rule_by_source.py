import pandas as pd

# Load features
df = pd.read_csv("match_features.tsv", sep="\t")

# Calculate weighted score
df["weighted_score"] = (
    0.4 * df["name_score"] +
    0.6 * df["address_score"]
)

# New rule
prediction = (
    (df["weighted_score"] >= 80) |
    (
        (df["name_score"] >= 95) &
        (df["address_score"] >= 60)
    )
)

df["prediction"] = prediction

print("\n===== NEW RULE: SOURCE-WISE RESULTS =====\n")

for source in ["source2", "source3"]:

    data = df[df["source"] == source]

    tp = ((data["prediction"] == True) &
          (data["is_match"] == 1)).sum()

    fp = ((data["prediction"] == True) &
          (data["is_match"] == 0)).sum()

    fn = ((data["prediction"] == False) &
          (data["is_match"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f05 = (
        1.25 * precision * recall /
        (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("----------------------------------------")
    print("Source:", source)
    print("Candidates:", len(data))
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("Precision:", round(precision * 100, 2), "%")
    print("Recall:", round(recall * 100, 2), "%")
    print("F0.5:", round(f05 * 100, 2), "%")

print("\nDone.")