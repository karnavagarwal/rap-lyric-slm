import torch
import torch.nn.functional as F
from model import RapModel
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = 'rap_model_checkpoint.pt'
tokenizer_path = 'rap_tokenizer.json'

tokenizer = Tokenizer.from_file(tokenizer_path)

model = RapModel().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

def generate_bars(prompt, temperature=1.0, penalty=1.15, max_new_tokens=300, top_k=40):
    encoded = tokenizer.encode(prompt).ids
    if not encoded:
        encoded = [0]
        
    idx = torch.tensor([encoded], dtype=torch.long).to(device)
    block_size = 1024 

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            recent_tokens = set(idx[0].tolist()[-80:])
            for t in recent_tokens:
                logits[0, t] -= penalty
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Decode and yield the text step-by-step
            current_text = tokenizer.decode(idx[0].tolist(), skip_special_tokens=False)
            formatted_text = current_text.replace('[EOS]', '\n').replace('[BOS]', '')
            
            yield formatted_text

if __name__ == "__main__":
    starting_lyric = "I stay polite"
    generated_text = generate_bars(starting_lyric, temperature=1.0, penalty=1.15)
    print(generated_text)