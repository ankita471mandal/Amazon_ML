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
# 2. Calculate weighted score
# =======================================

df["weighted_score"] = (
    0.4 * df["name_score"]
    +
    0.6 * df["address_score"]
)


# =======================================
# 3. Predict matches
# =======================================

df["predicted_match"] = (
    df["weighted_score"] >= 80
)


# =======================================
# 4. Evaluate each source
# =======================================

for source in ["source2", "source3"]:

    data = df[df["source"] == source]

    actual = data["is_match"] == 1
    predicted = data["predicted_match"]


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


    print("\n================================")
    print(source.upper())
    print("================================")

    print("Candidates:", len(data))

    print("True positives:", tp)

    print("False positives:", fp)

    print("False negatives:", fn)

    print(
        "Precision:",
        round(precision * 100, 2),
        "%"
    )

    print(
        "Recall:",
        round(recall * 100, 2),
        "%"
    )

    print(
        "F0.5:",
        round(f05 * 100, 2)
    )