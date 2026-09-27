import pandas as pd

# Load feature file
df = pd.read_csv("match_features.tsv", sep="\t")

# Weighted score
df["weighted_score"] = (
    0.4 * df["name_score"] +
    0.6 * df["address_score"]
)

# Rules to test
rules = [
    (80, 90),
    (80, 85),
    (80, 80),
    (75, 90),
    (75, 85),
    (75, 80),
]

results = []

for normal_threshold, address_threshold in rules:

    # Normal rule:
    normal_match = df["weighted_score"] >= 80

    # Address-strong rule:
    address_strong_match = (
        (df["name_score"] >= normal_threshold) &
        (df["address_score"] >= address_threshold)
    )

    prediction = normal_match | address_strong_match

    tp = ((prediction == True) & (df["is_match"] == 1)).sum()
    fp = ((prediction == True) & (df["is_match"] == 0)).sum()
    fn = ((prediction == False) & (df["is_match"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f05 = (
        (1 + 0.5**2)
        * precision * recall
        / ((0.5**2 * precision) + recall)
        if (precision + recall) > 0
        else 0
    )

    results.append({
        "name_threshold": normal_threshold,
        "address_threshold": address_threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "F0.5": f05 * 100
    })


results_df = pd.DataFrame(results)

print("\nAddress-Strong Rule Results:")
print(results_df.to_string(index=False))

results_df.to_csv(
    "address_strong_rule_results.tsv",
    sep="\t",
    index=False
)

print("\nSaved: address_strong_rule_results.tsv")