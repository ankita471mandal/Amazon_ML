import pandas as pd


# =======================================
# 1. Load data
# =======================================

df = pd.read_csv(
    "match_features.tsv",
    sep="\t"
)


# =======================================
# 2. Evaluate a rule
# =======================================

def evaluate_rule(
    strong_name,
    normal_name,
    normal_address
):

    predicted = (
        (df["name_score"] >= strong_name)
        |
        (
            (df["name_score"] >= normal_name)
            &
            (df["address_score"] >= normal_address)
        )
    )

    actual = df["is_match"] == 1


    tp = (predicted & actual).sum()

    fp = (predicted & ~actual).sum()

    fn = (~predicted & actual).sum()


    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)


    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)


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
        "strong_name": strong_name,
        "normal_name": normal_name,
        "normal_address": normal_address,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision * 100,
        "recall": recall * 100,
        "F0.5": f05 * 100
    }


# =======================================
# 3. Test many hybrid rules
# =======================================

results = []


for strong_name in [85, 90, 95]:

    for normal_name in [60, 65, 70, 75, 80]:

        for normal_address in [70, 75, 80, 85, 90]:

            result = evaluate_rule(
                strong_name,
                normal_name,
                normal_address
            )

            results.append(result)


# =======================================
# 4. Sort results
# =======================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F0.5",
    ascending=False
)


# =======================================
# 5. Display
# =======================================

print("\n================================")
print("HYBRID MATCHING ANALYSIS")
print("================================")

print("\nTop 20 rules:\n")

print(
    results_df.head(20).to_string(
        index=False
    )
)


# =======================================
# 6. Save
# =======================================

results_df.to_csv(
    "hybrid_matching_results.tsv",
    sep="\t",
    index=False
)

print("\nFile created:")
print("hybrid_matching_results.tsv")