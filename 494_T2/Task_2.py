from collections import defaultdict

# Define constants
CUTOFF_MS = millis("2025-09-30") # convert date into milliseconds 
SOURCE_DEFAULT = "DEFAULT" # user chose (self selected)
SOURCE_POP = "POPULATION_SAMPLE" # user was nudged by system

def bucket_for(note_factor1): # categorize notes by ideology
    if note_factor1 < -0.2:
        return "low"
    if note_factor1 < 0.2:
        return "mid"
    return "high"

# Load tables
rater_factors = read("prescoringRaterModelOutput_1dim.tsv")
lean_by_rater = dict(zip(rater_factors["raterParticipantId"],
                         rater_factors["internalRaterFactor1"])) # for any rating, get rater ideology

note_factors = read("prescoringNoteModelOutput_1dim.tsv")
note_factor = dict(zip(note_factors["noteId"],
                       note_factors["internalNoteFactor1"])) # for any note, get rater ideology

status = read("noteStatusHistory-00000.tsv")

first_non_nmr_time = dict(zip(status["noteId"],
                              status["timestampMillisOfFirstNonNMRStatus"])) # used to make stage 1 and stage 2

flipped = {}
for _, row in status.iterrows():
    flipped[row["noteId"]] = (row["firstNonNMRStatus"] != row["latestNonNMRStatus"]) # store flip label per note

# Qualifying notes (>= 5 POPULATION_SAMPLED ratings)
pop_count = defaultdict(int)

for ratings_file in ratings_files:              # loop through ratings files
    ratings = read(ratings_file)
    for _, rating in ratings.iterrows():

        if rating["createdAtMillis"] < CUTOFF_MS:
            continue # skip old ratings

        if rating["ratingSourceBucketed"] == SOURCE_POP:
            pop_count[rating["noteId"]] += 1 # count ratings

qual_notes = {nid for nid, c in pop_count.items() if c >= 5} # keep notes that meet minimum sample size

# Collect leans by (bucket, stage, group, noteId) 
# leans_by_note[bucket][stage][group][noteId] -> list of leans
leans_by_note = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

for ratings_file in ratings_files:
    ratings = read(ratings_file)
    for _, rating in ratings.iterrows():

        rating_time = rating["createdAtMillis"] # get fields from rating row
        noteId = rating["noteId"]

        if rating_time < CUTOFF_MS:
            continue

        if noteId not in qual_notes:
            continue
  
        rater_id = rating["participantId"]     
        lean = lean_by_rater.get(rater_id) # get rater ideology
        if lean is None:
            continue

        nf = note_factor.get(noteId) # get note ideology
        if nf is None:
            continue
        bucket = bucket_for(nf)

        cut = first_non_nmr_time.get(noteId) # find note's first non-NMR timestamp
        if cut is None:
            stage = "stage2"
        else:
            stage = "stage1" if rating_time < cut else "stage2"

        source = rating["ratingSourceBucketed"]  # DEFAULT or POPULATION_SAMPLE

        if source == SOURCE_DEFAULT: 
            leans_by_note[bucket][stage]["DEFAULT"][noteId].append(lean) # store lean inside correct category

        if source == SOURCE_POP:
            leans_by_note[bucket][stage]["POP"][noteId].append(lean)

        if source in (SOURCE_DEFAULT, SOURCE_POP):
            leans_by_note[bucket][stage]["BOTH"][noteId].append(lean)
