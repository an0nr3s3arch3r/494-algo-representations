import pandas as pd
import os

# Down to 10k
NUM_NOTES = 10000
#Load file
notes_df = pd.read_csv("data/notes-00000.tsv", sep="\t")
subset_df = notes_df.head(NUM_NOTES)
subset_df.to_csv("data/filtered_notes.tsv", index=False, sep="\t")  
note_ids_set = set(subset_df['noteId'])

ratings_per_note = {}
total_ratings = 0
filtered_ratings_df = []
rating_file_ext = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007',
                   '00008', '00009', '00010', '00011', '00012', '00013', '00014', '00015',
                   '00016', '00017', '00018', '00019'
                   ]
#Processing ratings for 20 files
for i in rating_file_ext:
        rating_file = f"data/ratings-{i}.tsv"
        print(f"Rating file # {i}")

        rating_df = pd.read_csv(rating_file, sep='\t')
        filtered = rating_df[rating_df['noteId'].isin(note_ids_set)]
        filtered_ratings_df.append(filtered)
        total_ratings += len(filtered)

        counts = filtered['noteId'].value_counts()
        for note_id, count in counts.items():
            if note_id in ratings_per_note:
                ratings_per_note[note_id] += count
            else:
                ratings_per_note[note_id] = count
        print(f"Found {len(filtered)} ratings")
    
# add filtered (df) to filtered_ratings (df)
all_filtered_ratings = pd.concat(filtered_ratings_df, ignore_index=True)
all_filtered_ratings.to_csv("data/filtered_ratings.tsv", index=False, sep="\t", header=True)
print(f"Saved filtered ratings to data/filtered_ratings.tsv")

numNotes = len(note_ids_set)
numRatings = total_ratings
ratingsSeries = pd.Series(ratings_per_note)
medianRatings = ratingsSeries.median()
meanRatings = ratingsSeries.mean()
maxRatings = ratingsSeries.max()
minRatings = ratingsSeries.min()
    
# Print summary
print(f"Total notes: {numNotes}")
print(f"Total ratings: {numRatings}")
print(f"Median Ratings per note: {medianRatings}")
print(f"Mean ratings per note: {meanRatings}")
print(f"Max ratings per note: {maxRatings}")
print(f"Min ratings per note: {minRatings}")

# Save statistics 
with open("data/dataset_statistics.txt", "w") as f:
    f.write(f"Total notes: {numNotes}\n")
    f.write(f"Total ratings: {numRatings}\n")
    f.write(f"Median ratings per note: {medianRatings}\n")
    f.write(f"Mean ratings per note: {meanRatings:.2f}\n")
    f.write(f"Max ratings per note: {maxRatings}\n")
    f.write(f"Min ratings per note: {minRatings}\n")

    





    