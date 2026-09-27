import pandas as pd
import numpy as np
from tokenizers import Tokenizer

input_csv = r"D:\Projects\RapLyricGen\rap_lyrics_normalized.csv"
tokenizer_path = r"D:\Projects\RapLyricGen\rap_tokenizer.json"
output_bin = r"D:\Projects\RapLyricGen\train_data.bin"

print("Loading tokenizer...")
tokenizer = Tokenizer.from_file(tokenizer_path)

eos_id = tokenizer.token_to_id("[EOS]")
bos_id = tokenizer.token_to_id("[BOS]")

chunksize = 5000
total_tokens = 0

print("Compiling into a binary stream...")

with open(output_bin, "wb") as f:
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, engine= 'python', on_bad_lines='skip'):
        batch_ids = []
        for lyric in chunk.iloc[:, -1].dropna().astype(str):
            ids = tokenizer.encode(lyric).ids
            batch_ids.append(bos_id)
            batch_ids.extend(ids)
            batch_ids.append(eos_id)

        arr = np.array(batch_ids, dtype = np.uint16)
        f.write(arr.tobytes())

        total_tokens += len(arr)
        print("Total tokens processed: ", total_tokens)

print("Compilation complete! Total tokens: ", total_tokens)
print("File saved!!!!")