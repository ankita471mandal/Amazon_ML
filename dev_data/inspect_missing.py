import pandas as pd


# =======================================
# 1. Load files
# =======================================

missing = pd.read_csv(
    "missing_matches.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "source2.tsv",
    sep="\t"
)

source3 = pd.read_csv(
    "source3.tsv",
    sep="\t"
)


# =======================================
# 2. Inspect each missing match
# =======================================

print("\n================================")
print("INSPECTING MISSING MATCHES")
print("================================")


for _, row in missing.iterrows():

    source1_id = row["source1_id"]
    source = row["source"]
    missing_id = row["missing_entity_id"]


    print("\n================================")
    print("Source:", source)
    print("================================")

    print("Source 1 ID:", source1_id)
    print("Source 1 Name:", row["source1_name"])
    print("Source 1 Address:", row["source1_address"])
    print("Source 1 Country:", row["source1_country"])

    print("\nActual matching record:")


    # -----------------------------------
    # Source 2
    # -----------------------------------

    if source == "source2":

        match = source2[
            source2["entity_id"] == missing_id
        ]

    # -----------------------------------
    # Source 3
    # -----------------------------------

    else:

        match = source3[
            source3["entity_id"] == missing_id
        ]


    if match.empty:

        print("Record not found!")

    else:

        match = match.iloc[0]

        print("Entity ID:", match["entity_id"])
        print("Business Name:", match["business_name"])
        print("Business Address:", match["business_address"])
        print("Country:", match["country"])