import pandas as pd
import sqlite3
import re
import os
import time


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "dataset/train"
OUTPUT_DIR = "output"

SOURCE1_FILE = os.path.join(DATA_DIR, "train_source1.tsv")
SOURCE2_FILE = os.path.join(DATA_DIR, "train_source2.tsv")
SOURCE3_FILE = os.path.join(DATA_DIR, "train_source3.tsv")

DB_FILE = "candidate_index.db"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Number of rows read at one time
CHUNK_SIZE = 5000


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).lower()

    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def get_tokens(text):
    """
    Return useful tokens.

    Very short tokens are ignored because they create
    too many irrelevant candidates.
    """

    return {
        token
        for token in text.split()
        if len(token) >= 3
    }


# ============================================================
# CHECK INPUT FILES
# ============================================================

for filename in [
    SOURCE1_FILE,
    SOURCE2_FILE,
    SOURCE3_FILE
]:

    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"File not found: {filename}"
        )


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# REMOVE OLD DATABASE
# ============================================================

if os.path.exists(DB_FILE):

    print("Removing old candidate index...")

    os.remove(DB_FILE)


# ============================================================
# REMOVE OLD OUTPUT
# ============================================================

if os.path.exists(OUTPUT_FILE):

    print("Removing old candidate output...")

    os.remove(OUTPUT_FILE)


# ============================================================
# CREATE SQLITE DATABASE
# ============================================================

print("\nCreating SQLite candidate index...")

conn = sqlite3.connect(DB_FILE)

cursor = conn.cursor()


# Store original records
cursor.execute("""
CREATE TABLE records (
    entity_id TEXT,
    source TEXT,
    country TEXT,
    business_name TEXT,
    business_address TEXT
)
""")


# Token inverted index
cursor.execute("""
CREATE TABLE tokens (
    token TEXT,
    entity_id TEXT,
    source TEXT,
    country TEXT,
    field TEXT
)
""")


# Important indexes
cursor.execute("""
CREATE INDEX idx_tokens_lookup
ON tokens(token, source, country, field)
""")


cursor.execute("""
CREATE INDEX idx_records_lookup
ON records(entity_id, source)
""")


conn.commit()


# ============================================================
# BUILD SOURCE 2 / SOURCE 3 INDEX
# ============================================================

def build_index(filename, source_name):

    print()
    print("=" * 60)
    print("BUILDING INDEX:", source_name)
    print("=" * 60)

    total_rows = 0
    total_tokens = 0

    start_time = time.time()

    for chunk in pd.read_csv(
        filename,
        sep="\t",
        chunksize=CHUNK_SIZE,
        dtype=str
    ):

        chunk = chunk.fillna("")

        records_to_insert = []
        tokens_to_insert = []

        for _, row in chunk.iterrows():

            entity_id = str(
                row["entity_id"]
            )

            country = clean_text(
                row["country"]
            )

            name = clean_text(
                row["business_name"]
            )

            address = clean_text(
                row["business_address"]
            )

            # ------------------------------------------------
            # Store record
            # ------------------------------------------------

            records_to_insert.append(
                (
                    entity_id,
                    source_name,
                    country,
                    name,
                    address
                )
            )

            # ------------------------------------------------
            # Name tokens
            # ------------------------------------------------

            for token in get_tokens(name):

                tokens_to_insert.append(
                    (
                        token,
                        entity_id,
                        source_name,
                        country,
                        "name"
                    )
                )

            # ------------------------------------------------
            # Address tokens
            # ------------------------------------------------

            for token in get_tokens(address):

                tokens_to_insert.append(
                    (
                        token,
                        entity_id,
                        source_name,
                        country,
                        "address"
                    )
                )

        # ----------------------------------------------------
        # Insert records
        # ----------------------------------------------------

        cursor.executemany(
            """
            INSERT INTO records
            VALUES (?, ?, ?, ?, ?)
            """,
            records_to_insert
        )

        # ----------------------------------------------------
        # Insert tokens
        # ----------------------------------------------------

        cursor.executemany(
            """
            INSERT INTO tokens
            VALUES (?, ?, ?, ?, ?)
            """,
            tokens_to_insert
        )

        conn.commit()

        total_rows += len(chunk)
        total_tokens += len(tokens_to_insert)

        elapsed = time.time() - start_time

        print(
            f"{source_name}: "
            f"{total_rows:,} records | "
            f"{total_tokens:,} tokens | "
            f"{elapsed/60:.1f} min"
        )

    print()
    print(
        f"{source_name} indexing complete: "
        f"{total_rows:,} records"
    )


# Build Source 2 index
build_index(
    SOURCE2_FILE,
    "source2"
)


# Build Source 3 index
build_index(
    SOURCE3_FILE,
    "source3"
)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

print()
print("=" * 60)
print("STARTING SOURCE 1 CANDIDATE GENERATION")
print("=" * 60)


source1_total = 0
candidate_total = 0

first_write = True

start_time = time.time()


for chunk in pd.read_csv(
    SOURCE1_FILE,
    sep="\t",
    chunksize=CHUNK_SIZE,
    dtype=str
):

    chunk = chunk.fillna("")

    output_rows = []

    for _, row in chunk.iterrows():

        source1_id = str(
            row["entity_id"]
        )

        country = clean_text(
            row["country"]
        )

        name = clean_text(
            row["business_name"]
        )

        address = clean_text(
            row["business_address"]
        )

        name_tokens = get_tokens(name)
        address_tokens = get_tokens(address)


        # ====================================================
        # PROCESS SOURCE 2 AND SOURCE 3
        # ====================================================

        for source_name in [
            "source2",
            "source3"
        ]:

            candidate_scores = {}


            # ------------------------------------------------
            # NAME TOKEN BLOCKING
            # ------------------------------------------------

            for token in name_tokens:

                results = cursor.execute(
                    """
                    SELECT entity_id
                    FROM tokens
                    WHERE token = ?
                    AND source = ?
                    AND country = ?
                    AND field = 'name'
                    """,
                    (
                        token,
                        source_name,
                        country
                    )
                ).fetchall()


                for result in results:

                    candidate_id = result[0]

                    candidate_scores[candidate_id] = (
                        candidate_scores.get(
                            candidate_id,
                            0
                        ) + 2
                    )


            # ------------------------------------------------
            # ADDRESS TOKEN BLOCKING
            # ------------------------------------------------

            for token in address_tokens:

                results = cursor.execute(
                    """
                    SELECT entity_id
                    FROM tokens
                    WHERE token = ?
                    AND source = ?
                    AND country = ?
                    AND field = 'address'
                    """,
                    (
                        token,
                        source_name,
                        country
                    )
                ).fetchall()


                for result in results:

                    candidate_id = result[0]

                    candidate_scores[candidate_id] = (
                        candidate_scores.get(
                            candidate_id,
                            0
                        ) + 1
                    )


            # ------------------------------------------------
            # KEEP TOP 50
            # ------------------------------------------------

            ranked_candidates = sorted(
                candidate_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )[:50]


            # ------------------------------------------------
            # SAVE CANDIDATES
            # ------------------------------------------------

            for candidate_id, token_score in ranked_candidates:

                output_rows.append(
                    {
                        "source1_entity_id": source1_id,
                        "candidate_entity_id": candidate_id,
                        "source": source_name
                    }
                )


    # ========================================================
    # WRITE THIS CHUNK TO DISK
    # ========================================================

    if output_rows:

        output_df = pd.DataFrame(
            output_rows
        ).drop_duplicates()


        output_df.to_csv(
            OUTPUT_FILE,
            sep="\t",
            index=False,
            mode="a",
            header=first_write
        )


        first_write = False

        candidate_total += len(
            output_df
        )


    source1_total += len(chunk)

    elapsed = time.time() - start_time

    print(
        f"Source 1 processed: "
        f"{source1_total:,} | "
        f"candidates written: "
        f"{candidate_total:,} | "
        f"time: {elapsed/60:.1f} min"
    )


# ============================================================
# FINISH
# ============================================================

conn.close()


print()
print("=" * 60)
print("CANDIDATE GENERATION COMPLETE")
print("=" * 60)

print(
    f"Source 1 processed: "
    f"{source1_total:,}"
)

print(
    f"Approx candidates written: "
    f"{candidate_total:,}"
)

print(
    f"Output file: "
    f"{OUTPUT_FILE}"
)

print()
print("DO NOT RUN THE MATCHING STAGE YET.")
print("Send me the final numbers first.")