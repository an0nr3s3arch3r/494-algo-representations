import pandas as pd
import re
import glob


sug_df = pd.read_csv("sug_df_labelled.csv")
notes_files = glob.glob("notes-*.tsv")
notes_list = [pd.read_csv(f, sep="\t", usecols=["noteId", "tweetId", "createdAtMillis"], low_memory=False) for f in notes_files]
notes_df = pd.concat(notes_list, ignore_index=True)

# 5.1
def has_link(text):
    return bool(re.search(r'http[s]?://|www\.', str(text), re.IGNORECASE))

sug_df["has_link"] = sug_df["suggestion"].apply(has_link)
n_with_link = sug_df["has_link"].sum()
pct_with_link = 100 * n_with_link / len(sug_df)
print(f"Suggestions with a link: {n_with_link:,} / {len(sug_df):,} ({pct_with_link:.1f}%)")

# 5.2
note_tweet = notes_df[["noteId", "tweetId"]].drop_duplicates("noteId")
sug_with_link = sug_df[sug_df["has_link"]].copy()
sug_with_link = sug_with_link.merge(note_tweet, on="noteId", how="left")
all_notes_by_tweet = note_tweet.groupby("tweetId")["noteId"].apply(set).to_dict()

def has_new_version(row):
    tweet_notes = all_notes_by_tweet.get(row["tweetId"], set())
    return len(tweet_notes) > 1

sug_with_link["new_version"] = sug_with_link.apply(has_new_version, axis=1)
n_new_version = sug_with_link["new_version"].sum()
pct_new_version = 100 * n_new_version / max(len(sug_with_link), 1)
print(f"Link suggestions where a new note version exists on same tweetId: {n_new_version:,} / {len(sug_with_link):,} ({pct_new_version:.1f}%)")

#5.3 n 4
print("\n-- Suggestions WITH link where new version EXISTS --")
for s in sug_with_link[sug_with_link["new_version"]]["suggestion"].head(3).tolist():
    print(f"- {s}\n")

print("\n-- Suggestions WITH link where NO new version exists --")
for s in sug_with_link[~sug_with_link["new_version"]]["suggestion"].head(3).tolist():
    print(f"- {s}\n")