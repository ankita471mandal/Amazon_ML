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
# 2. Function to evaluate
# =======================================

def evaluate_rule(name_weight, address_weight, threshold):

    # Calculate weighted score
    score = (
        name_weight * df["name_score"]
        +
        address_weight * df["address_score"]
    )

    # Predict match
    predicted = score >= threshold

    # Actual ground truth
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
        "name_weight": name_weight,
        "address_weight": address_weight,
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "F0.5": f05 * 100
    }


# =======================================
# 3. Test different weights
# =======================================

results = []


weights = [
    (0.9, 0.1),
    (0.8, 0.2),
    (0.7, 0.3),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.3, 0.7),
    (0.2, 0.8),
    (0.1, 0.9)
]


thresholds = [
    60,
    65,
    70,
    75,
    80,
    85,
    90
]


for name_weight, address_weight in weights:

    for threshold in thresholds:

        result = evaluate_rule(
            name_weight,
            address_weight,
            threshold
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
# 5. Show results
# =======================================

print("\n================================")
print("WEIGHTED MATCHING ANALYSIS")
print("================================")

print("\nTop 20 rules by F0.5:\n")

print(
    results_df.head(20).to_string(
        index=False
    )
)


# =======================================
# 6. Save results
# =======================================

results_df.to_csv(
    "weighted_matching_results.tsv",
    sep="\t",
    index=False
)

print("\nFile created:")
print("weighted_matching_results.tsv")