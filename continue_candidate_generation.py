import pandas as pd
import sqlite3
import re
import os
import time
from collections import defaultdict

DB_FILE = "candidate_index.db"
SOURCE1_FILE = "dev_data/source1.tsv"
OUTPUT_FILE = "dev_data/candidate_pairs_optimized.tsv"

TEST_ROWS = 1000

# More blocking routes = better recall
MAX_PER_BLOCK = 300

# Final candidates per source
FINAL_PER_SOURCE = 120


def clean_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower()
    x = re.sub(r"[^\w\s]", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def get_blocks(name, address):
    blocks = set()

    name = clean_text(name)
    address = clean_text(address)

    # Name word prefixes
    for token in name.split():
        if len(token) >= 4:
            blocks.add("N:" + token[:4])

    # Address word prefixes
    for token in address.split():
        if len(token) >= 4:
            blocks.add("A:" + token[:4])

    # First characters
    if name:
        blocks.add("NF:" + name[0])

    if address:
        blocks.add("AF:" + address[0])

    return blocks


print("Opening existing database...")
conn = sqlite3.connect(DB_FILE)

print("Loading development Source 1...")

s1 = pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    dtype=str
).head(TEST_ROWS)

s1 = s1.fillna("")

if os.path.exists(OUTPUT_FILE):
    os.remove(OUTPUT_FILE)

output = []

start = time.time()

print("Source 1 records:", len(s1))


for i, row in s1.iterrows():

    s1_id = row["entity_id"]
    country = clean_text(row["country"])

    blocks = get_blocks(
        row["business_name"],
        row["business_address"]
    )

    for source in ["source2", "source3"]:

        scores = defaultdict(int)

        for block in blocks:

            # Name prefix block
            if block.startswith("N:"):

                prefix = block[2:]

                rows = conn.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE source = ?
                    AND country = ?
                    AND business_name LIKE ?
                    LIMIT ?
                    """,
                    (
                        source,
                        country,
                        prefix + "%",
                        MAX_PER_BLOCK
                    )
                ).fetchall()

            # Address prefix block
            elif block.startswith("A:"):

                prefix = block[2:]

                rows = conn.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE source = ?
                    AND country = ?
                    AND business_address LIKE ?
                    LIMIT ?
                    """,
                    (
                        source,
                        country,
                        prefix + "%",
                        MAX_PER_BLOCK
                    )
                ).fetchall()

            # First name character
            elif block.startswith("NF:"):

                prefix = block[3:]

                rows = conn.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE source = ?
                    AND country = ?
                    AND business_name LIKE ?
                    LIMIT ?
                    """,
                    (
                        source,
                        country,
                        prefix + "%",
                        MAX_PER_BLOCK
                    )
                ).fetchall()

            # First address character
            else:

                prefix = block[3:]

                rows = conn.execute(
                    """
                    SELECT entity_id
                    FROM records
                    WHERE source = ?
                    AND country = ?
                    AND business_address LIKE ?
                    LIMIT ?
                    """,
                    (
                        source,
                        country,
                        prefix + "%",
                        MAX_PER_BLOCK
                    )
                ).fetchall()

            for (entity_id,) in rows:
                scores[entity_id] += 1

        # Keep candidates that appeared in multiple blocks first
        ranked = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:FINAL_PER_SOURCE]

        for candidate_id, score in ranked:

            output.append({
                "source1_entity_id": s1_id,
                "candidate_entity_id": candidate_id,
                "source": source
            })

    if (i + 1) % 100 == 0:

        print(
            f"Processed: {i+1:,}/{len(s1):,} | "
            f"Candidates: {len(output):,} | "
            f"Time: {(time.time()-start)/60:.1f} min"
        )


result = pd.DataFrame(output)

result.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False
)

conn.close()

print()
print("================================")
print("TEST COMPLETE")
print("================================")
print("Source 1 processed:", len(s1))
print("Candidates:", len(result))
print(
    "Candidates per Source-1:",
    len(result) / len(s1)
)
print(
    f"Time: {(time.time()-start)/60:.2f} minutes"
)
print("Output:", OUTPUT_FILE)