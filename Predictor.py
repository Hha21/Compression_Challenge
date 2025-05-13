import torch
import torch.nn as nn
import numpy as np 
import random
from collections import deque
from copy import deepcopy
import sys
import matplotlib.pyplot as plt
from torch.distributions import Normal  
from tqdm import tqdm

# GET .so 
sys.path.insert(1, "./build")
import neuralink

# print("Files:", reader.getNumFiles())
# print("Vocab size:", reader.getNumTokens())

# stream0 = reader.getTokenStream(0)
# print("First 20 token IDs:", stream0[:20])

class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, embed_dim = 64, hidden_dim = 128, num_layers = 1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)                   # (B, T) --> (B, T, E)
        out, _ = self.lstm(x)               # (B, T, H)
        out = out[:, -1, :]
        logits = self.fc(out)

        return logits

class MLPPredictor(nn.Module):
    def __init__(self, vocab_size,context_len = 8, embed_dim = 32, hidden_dim = 64):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim * context_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))

        return self.fc2(x)

class StreamingDataset(torch.utils.data.Dataset):
    def __init__(self, indices, context_len, reader):
        self.context_len = context_len
        self.data = []

        for idx in indices:
            stream = reader.getTokenStream(idx)
            for i in range(context_len, len(stream)):
                context = stream[i - context_len:i]
                target = stream[i]
                self.data.append((context, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, target = self.data[index]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train(num_epochs = 10, dict_size = 1023, context_len = 8, batch_size = 64):
    reader = neuralink.WavReader(dict_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(torch.cuda.is_available())       
    print(torch.cuda.get_device_name(0))   

    #model = LSTMPredictor(vocab_size = reader.getNumTokens()).to(device)
    model = MLPPredictor(vocab_size = reader.getNumTokens(), context_len = context_len).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr = 1e-3)

    # TRAIN/TEST SPLIT
    all_indices = list(range(reader.getNumFiles()))
    random.shuffle(all_indices)

    split_ratio = 0.2
    split_idx = int(split_ratio * len(all_indices))

    train_indices = all_indices[:split_idx]
    #test_indices = all_indices[split_idx:]

    dataset = StreamingDataset(train_indices, context_len, reader)  
    loader = torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True  
    )

    for epoch in range(num_epochs):
        print(f"Training... epoch num {epoch}")
        for xb, yb in tqdm(loader, desc=f"Epoch {epoch+1}"):
            xb = xb.to(device, non_blocking = True)
            yb = yb.to(device, non_blocking = True)
            logits = model(xb)
            loss = loss_fn(logits, yb)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

if __name__ == "__main__":

    train() 
