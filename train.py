import torch
import numpy as np
from torch.nn import functional as F
from model import RapModel

batch_size = 16
block_size = 1024
#for every forward pass, we have 16 * 1024 = 16384 tokens processed.
#700 million tokens / 16384 roughly gives us 42700 iterations for 1 epoch.
#Taking 3 epochs, we must have around 128000 iterations.
max_iters = 128000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#Using np.memmap to load the data on demand to the RAM rather than the whole file
data = np.memmap('train_data.bin', dtype = np.uint16, mode = 'r')

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    #scooping out the context and the target(x and y)
    #Casting to int64 to fulfill the embedding layer requirements
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

if __name__ == "__main__":
    print(f'Loading model to {device}...')

    model = RapModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    print('Starting training sanity check...')

    for iter in range(max_iters):
        X, Y = get_batch()
        logits = model(X)
        B, T, C = logits.shape
        logits_flat = logits.view(B*T, C)
        Y_flat = Y.view(B*T)
        loss = F.cross_entropy(logits_flat, Y_flat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter%10 == 0:
            print(f"Step {iter} | Loss: {loss.item():.4f}")

        if (iter > 0 and iter %4000 == 0):
            torch.save(model.state_dict(), 'rap_model_checkpoint.pt')
            print(f"-- Checkpoint saved at step {iter} --")

    print('Training complete!!!')
    torch.save(model.state_dict(), 'rap_model_checkpoint.pt')
    print("Final model has been saved!!")
