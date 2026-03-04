# lets Python work with operating system settings (avoids OpenMP crash)
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch # helps run translator model
import pandas as pd # reads and stores data
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer # M2M100 model + tokenizer
from langdetect import detect # detects what langauge the text is in
from tqdm import tqdm # progress bar
import json # save cache to a file
from pathlib import Path # file checking 


MODEL_NAME = "facebook/m2m100_418M" # specify which translation model to use
INPUT_FILE = "notes-00000.tsv" # notes data
OUTPUT_FILE = "notes_with_english.parquet" # final translated table 
CACHE_FILE = "note_translations_en.json" # store translations to avoid redoing them again

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# tokenizer = convert text into tokens (numbers) for the model to understand
print("Loading model...")
tokenizer = M2M100Tokenizer.from_pretrained(MODEL_NAME) # load tokenizer
model = M2M100ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device) # load translation model and puts it on CPU/GPU
model.eval() # we are using the model, not training it

# saving data so program doesn't have to redo notes it has already translate in previous runs
cache_path = Path(CACHE_FILE) # create file object pointing to note_translations_en.json
if cache_path.exists(): # check if file already exists
    with open(cache_path, "r") as f:
        translation_cache = json.load(f)
else:
    translation_cache = {}

# mappings to help with consistency between langdetect and M2M100
LANG_MAP = {
    "zh-tw": "zh",
    "zh-cn": "zh",
    "zh-hk": "zh",
    "zh-sg": "zh",
    "jw": "jv",     
}

def detect_lang(text):
    try:
        lang = detect(text)
    except:
        lang = "en" # if detection fails, assume english

    lang = lang.lower() 
    return LANG_MAP.get(lang, lang) # apply mapping


def translate_to_english(text, src_lang):
    if src_lang == "en":
        return text

    tokenizer.src_lang = src_lang # tells model what language the text is in
    encoded = tokenizer( # turning text into format model can handle
        text,
        return_tensors="pt", # return numbers
        truncation=True, # if text is too long, cut it
        max_length=512
    ).to(device) # moves data to CPU/GPU

    generated = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.get_lang_id("en"), # force output language to be english
        max_length=512
    )

    return tokenizer.batch_decode(
        generated,
        skip_special_tokens=True
    )[0] # later code expects a string, not a list


print("Loading notes...")
df = pd.read_csv(INPUT_FILE, sep="\t", nrows=2000) # load first 2000 rows of data

translations = [] # empty list to store english translation for each note in order

print("Translating notes...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    note_id = str(row["noteId"]) # grab note ID from each row
    text = row["summary"] # grab summary from each row

    if note_id in translation_cache:
        en_text = translation_cache[note_id]
    else:
        lang = detect_lang(text) # detect language
        en_text = translate_to_english(text, lang) # translate to english 
        translation_cache[note_id] = en_text # store result in cache to avoid redoing later

    translations.append(en_text) # add english translation to the list
    if len(translations) % 100 == 0: # save cache every 100 translations 
        with open(CACHE_FILE, "w") as f:
            json.dump(translation_cache, f, ensure_ascii=False)

df["summary_en"] = translations # new column, summary_en that stores english translations
df.to_parquet(OUTPUT_FILE, index=False) # save table to parquet file and don't save row index as a column

with open(CACHE_FILE, "w") as f:
    json.dump(translation_cache, f, ensure_ascii=False, indent=2) # save cache again

print("Done!")
print(f"Saved: {OUTPUT_FILE}")
