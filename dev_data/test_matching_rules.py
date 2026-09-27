import pandas as pd


# =======================================
# 1. Load match features
# =======================================

df = pd.read_csv(
    "match_features.tsv",
    sep="\t"
)

print("Total candidates:", len(df))


# =======================================
# 2. Function to evaluate a rule
# =======================================

def evaluate_rule(name_threshold, address_threshold):

    # Predict match
    predicted = (
        (df["name_score"] >= name_threshold)
        &
        (df["address_score"] >= address_threshold)
    )

    actual = df["is_match"] == 1


    # True positives
    tp = (predicted & actual).sum()

    # False positives
    fp = (predicted & ~actual).sum()

    # False negatives
    fn = (~predicted & actual).sum()


    # Precision
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)


    # Recall
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)


    # F0.5
    beta = 0.5

    if precision == 0 and recall == 0:
        f05 = 0
    else:
        f05 = (
            (1 + beta**2)
            * precision
            * recall
        ) / (
            (beta**2 * precision)
            + recall
        )


    return {
        "name_threshold": name_threshold,
        "address_threshold": address_threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "F0.5": f05 * 100
    }


# =======================================
# 3. Test different rules
# =======================================

results = []


for name_threshold in [60, 65, 70, 75, 80, 85, 90]:

    for address_threshold in [60, 65, 70, 75, 80, 85, 90]:

        result = evaluate_rule(
            name_threshold,
            address_threshold
        )

        results.append(result)


# =======================================
# 4. Create results table
# =======================================

results_df = pd.DataFrame(results)


# Sort by F0.5
results_df = results_df.sort_values(
    "F0.5",
    ascending=False
)


# =======================================
# 5. Show best rules
# =======================================

print("\n================================")
print("MATCHING RULE ANALYSIS")
print("================================")

print("\nTop 15 rules by F0.5:\n")

print(
    results_df.head(15).to_string(
        index=False
    )
)


# =======================================
# 6. Save results
# =======================================

results_df.to_csv(
    "matching_rule_results.tsv",
    sep="\t",
    index=False
)

print("\nFile created:")
print("matching_rule_results.tsv")