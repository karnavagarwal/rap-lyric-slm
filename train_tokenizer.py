from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
import pandas as pd

input_file = r"D:\Projects\RapLyricGen\rap_lyrics_normalized.csv"
output_tokenizer = r"D:\Projects\RapLyricGen\rap_tokenizer.json"

tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space = False)
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(
    vocab_size = 30000,
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[VERSE]", "[CHORUS]", "[INTRO]", "[OUTRO]"]
)

def get_training_corpus():
    for chunk in pd.read_csv(input_file, chunksize = 1000, engine = 'python', on_bad_lines = 'skip'):
        yield chunk.iloc[:,-1].dropna().astype(str).tolist()

print("Training tokenizer...")

tokenizer.train_from_iterator(get_training_corpus(), trainer = trainer)

tokenizer.save(output_tokenizer)