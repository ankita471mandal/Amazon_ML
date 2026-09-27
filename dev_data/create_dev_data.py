import pandas as pd
import os

print("Loading training data...")

# Load original training data
source1 = pd.read_csv("../dataset/train/train_source1.tsv", sep="\t")
source2 = pd.read_csv("../dataset/train/train_source2.tsv", sep="\t")
source3 = pd.read_csv("../dataset/train/train_source3.tsv", sep="\t")
ground_truth = pd.read_csv("../dataset/train/train_ground_truth.tsv", sep="\t")

print("Data loaded successfully!")

# --------------------------------------------------
# 1. Take 1000 random Source 1 businesses
# --------------------------------------------------

sample_s1 = source1.sample(n=1000, random_state=42)

sample_s1_ids = set(sample_s1["entity_id"])


# --------------------------------------------------
# 2. Get ground truth for those 1000 businesses
# --------------------------------------------------

sample_gt = ground_truth[
    ground_truth["source1_entity_id"].isin(sample_s1_ids)
].copy()


# --------------------------------------------------
# 3. Find their actual matching Source 2 and Source 3 IDs
# --------------------------------------------------

true_s2_ids = set()
true_s3_ids = set()

for ids in sample_gt["matched_entity_ids"].fillna(""):
    
    if ids == "":
        continue

    for entity_id in ids.split(","):

        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            true_s2_ids.add(entity_id)

        elif entity_id.startswith("S3-"):
            true_s3_ids.add(entity_id)


# --------------------------------------------------
# 4. Take random negative examples
# --------------------------------------------------

random_s2 = source2.sample(n=20000, random_state=42)

random_s3 = source3.sample(n=20000, random_state=42)


# --------------------------------------------------
# 5. Add the actual matching records
# --------------------------------------------------

dev_s2 = pd.concat([
    random_s2,
    source2[source2["entity_id"].isin(true_s2_ids)]
]).drop_duplicates("entity_id")


dev_s3 = pd.concat([
    random_s3,
    source3[source3["entity_id"].isin(true_s3_ids)]
]).drop_duplicates("entity_id")


# --------------------------------------------------
# 6. Save development dataset
# --------------------------------------------------

# Because this script is already inside dev_data,
# save the files directly in the current folder.

dev_s2.to_csv("source2.tsv", sep="\t", index=False)
dev_s3.to_csv("source3.tsv", sep="\t", index=False)
sample_s1.to_csv("source1.tsv", sep="\t", index=False)
sample_gt.to_csv("ground_truth.tsv", sep="\t", index=False)


# --------------------------------------------------
# 7. Show result
# --------------------------------------------------

print("\nDevelopment dataset created successfully!")

print("Source 1:", len(sample_s1))
print("Source 2:", len(dev_s2))
print("Source 3:", len(dev_s3))
print("Ground truth:", len(sample_gt))

print("\nFiles created:")
print("source1.tsv")
print("source2.tsv")
print("source3.tsv")
print("ground_truth.tsv")