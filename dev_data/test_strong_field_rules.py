import pandas as pd

df = pd.read_csv("match_features.tsv", sep="\t")

# Calculate weighted score
df["weighted_score"] = (
    0.4 * df["name_score"] +
    0.6 * df["address_score"]
)

# Current baseline
baseline = df["weighted_score"] >= 80

rules = [
    # Very strong NAME + minimum address
    ("name95_addr60", (df["name_score"] >= 95) & (df["address_score"] >= 60)),
    ("name95_addr65", (df["name_score"] >= 95) & (df["address_score"] >= 65)),
    ("name95_addr70", (df["name_score"] >= 95) & (df["address_score"] >= 70)),

    # Very strong ADDRESS + minimum name
    ("addr95_name50", (df["address_score"] >= 95) & (df["name_score"] >= 50)),
    ("addr95_name55", (df["address_score"] >= 95) & (df["name_score"] >= 55)),
    ("addr95_name60", (df["address_score"] >= 95) & (df["name_score"] >= 60)),

    # Extremely strong address
    ("addr98_name45", (df["address_score"] >= 98) & (df["name_score"] >= 45)),
    ("addr98_name50", (df["address_score"] >= 98) & (df["name_score"] >= 50)),
]

results = []

for rule_name, extra_rule in rules:

    prediction = baseline | extra_rule

    tp = ((prediction == True) & (df["is_match"] == 1)).sum()
    fp = ((prediction == True) & (df["is_match"] == 0)).sum()
    fn = ((prediction == False) & (df["is_match"] == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    f05 = (
        1.25 * precision * recall
        / (0.25 * precision + recall)
        if (precision + recall) > 0
        else 0
    )

    results.append({
        "rule": rule_name,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "F0.5": f05 * 100
    })

results_df = pd.DataFrame(results)

print("\nStrong Field Rule Results:\n")
print(results_df.to_string(index=False))

results_df.to_csv(
    "strong_field_rule_results.tsv",
    sep="\t",
    index=False
)

print("\nSaved: strong_field_rule_results.tsv")