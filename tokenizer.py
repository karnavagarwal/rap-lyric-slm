from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDeocder

tokenizer = Tokenizer(BPE(unk_token = "[UNK]"))

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space = False)
tokenizer.decode = ByteLevelDeocder()

trainer = BpeTrainer(
    vocab_size = 100,
    special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]", "[VERSE]", "[CHORUS]", "[INTRO]", "[OUTRO]"]
)

fake_lyrics = ["[VERSE]\nYeah I am coding\n[CHORUS]\nI am building the brain"]
tokenizer.train_from_iterator(fake_lyrics, trainer = trainer)
encoded = tokenizer.encode(fake_lyrics[0])
print("Tokens: ", encoded.tokens)
print("IDs: ", encoded.ids)