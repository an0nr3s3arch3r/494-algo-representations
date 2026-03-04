import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
from openai import OpenAI
import json
import re
import time
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

notes_files = glob.glob("notes-*.tsv")
notes_list = [pd.read_csv(f, sep="\t", low_memory=False) for f in notes_files]
notes_df = pd.concat(notes_list, ignore_index=True)

print(notes_df.shape)
notes_df.head()

ratings_files = glob.glob("ratings-00000.tsv")
KEEP_COLS_RATINGS = ["noteId","raterParticipantId", "suggestion"]
chunks = []
all_rater_ids = set()

for f in ratings_files:
    print(f"Reading {f}...")
    cols = pd.read_csv(f, sep="\t", nrows=0).columns.tolist()
    rater_col = [c for c in cols if "articipantId" in c][0]  # matches both versions
    keep = [c for c in ["noteId", "suggestion", rater_col] if c in cols]
    df = pd.read_csv(f, sep="\t", usecols=keep, low_memory=False)
    df = df.rename(columns={rater_col: "raterParticipantId"})
    all_rater_ids.update(df["raterParticipantId"].dropna().unique())
    df = df[df["suggestion"].notna()]
    chunks.append(df)
    print(f"  -> {len(df):,} suggestions found")

sug_df = pd.concat(chunks, ignore_index=True)
print(f"Total unique raters: {len(all_rater_ids):,}")

collab_notes = notes_df[notes_df["isCollaborativeNote"] == 1].copy()
print(f"Collaborative notes: {len(collab_notes):,}")

# suggestions per note
sug_per_note = sug_df.groupby("noteId")["suggestion"].count().rename("n_suggestions")
print("\nSuggestions per note")
print(sug_per_note.describe().round(2))
print(f"Notes with exactly 1 suggestion: {(sug_per_note == 1).sum():,}")
print(f"Notes with 2+ suggestions: {(sug_per_note >= 2).sum():,}")
print(f"Max suggestions on a single note: {sug_per_note.max():,}")

#contributor analysis
sug_raters = sug_df["raterParticipantId"].nunique()
all_raters = len(all_rater_ids)
print(f"\nContributor Analysis")
print(f"Total unique raters (all ratings): {all_raters:,}")
print(f"Raters who submitted a suggestion: {sug_raters:,}")
print(f"% of all raters who suggested: {100 * sug_raters / all_raters:.2f}%")

sug_per_rater = sug_df.groupby("raterParticipantId")["suggestion"].count()
print(f"\nSuggestions per contributor")
print(sug_per_rater.describe().round(2))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# suggestions per note histogram
axes[0].hist(sug_per_note.values, bins=range(1, 14), color="#4C72B0", edgecolor="white", align="left")
axes[0].set_xlabel("Suggestions per note")
axes[0].set_ylabel("Number of notes")
axes[0].set_title("Suggestions per collaborative note")
axes[0].set_xticks(range(1, 13))

# suggestions per contributor (log scale since one person has 388)
axes[1].hist(sug_per_rater.values, bins=30, color="#55A868", edgecolor="white")
axes[1].set_yscale("log")
axes[1].set_xlabel("Suggestions per contributor")
axes[1].set_ylabel("Number of contributors (log scale)")
axes[1].set_title("Suggestions per contributor")

plt.tight_layout()
plt.savefig("fig_suggestions_distribution.png")
print("Saved: fig_suggestions_distribution.png")


SYSTEM_PROMPT = """You are an annotation assistant for a research project on Community Notes. 
A user has written a suggestion to improve an AI-drafted Community Note. 
Classify the suggestion along two axes:

Axis A — Task: What is the suggestion doing?
- Evaluation: judging or commenting on the note or the tweet without requesting a specific change
- Information Check: asking to verify or fact-check a claim
- Evidence: requesting sources, links, or supporting material
- Transformation: asking to rewrite, modify, or restructure the note
- Conversation: off-topic or directed at the tweet poster rather than the note

Axis B — Specificity: How actionable is the suggestion?
- Directive: concrete and actionable (specific wording, URL, or fact provided)
- Indirect: identifies a problem but does not prescribe an exact fix
- Generic: vague reaction with no specific target

Return ONLY a JSON object with keys: task, specificity. No explanation, no markdown."""

def label_suggestion(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Suggestion: {text}"}
            ],
            max_completion_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"Error: {e}")
        return {"task": "ERROR", "specificity": "ERROR"}

tasks, specificities = [], []
for i, row in sug_df.iterrows():
    labels = label_suggestion(row["suggestion"])
    tasks.append(labels.get("task", ""))
    specificities.append(labels.get("specificity", ""))
    if i % 100 == 0:
        print(f"Labelled {i}/{len(sug_df)}...")

sug_df["task"] = tasks
sug_df["specificity"] = specificities
print("Done labelling!")
print(sug_df[["suggestion", "task", "specificity"]].head(10))

sug_df.to_csv("sug_df_labelled.csv", index=False)
print("Saved: sug_df_labelled.csv")

# ── Step 5: Understanding source updates from suggestions ────────────────────
import re as re_module

# 5.1 % of suggestions with a link
def has_link(text):
    return bool(re_module.search(r'http[s]?://|www\.', str(text), re_module.IGNORECASE))

sug_df["has_link"] = sug_df["suggestion"].apply(has_link)
n_with_link = sug_df["has_link"].sum()
pct_with_link = 100 * n_with_link / len(sug_df)
print(f"\n=== Step 5.1 ===")
print(f"Suggestions with a link: {n_with_link:,} / {len(sug_df):,} ({pct_with_link:.1f}%)")

# 5.2 % of link suggestions that result in a new note version on same tweetId
# A "new version" = a different noteId on the same tweetId created AFTER the suggestion
note_tweet = notes_df[["noteId", "tweetId", "createdAtMillis"]].drop_duplicates("noteId")
sug_with_link = sug_df[sug_df["has_link"]].copy()
sug_with_link = sug_with_link.merge(note_tweet, on="noteId", how="left")

# for each suggestion, check if another note exists on same tweetId
all_notes_by_tweet = note_tweet.groupby("tweetId")["noteId"].apply(set).to_dict()