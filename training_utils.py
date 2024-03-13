import torch
import torch.nn as nn

from datasets import load_dataset

from data_process import get_seq_batch

def estimate_loss(data, model, criterion = nn.CrossEntropyLoss(),
                  batch_size=32, seq_len=4, moe=True, eval_iters=100):

    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        X, Y = get_seq_batch(data, batch_size=batch_size, seq_len=seq_len)
        logits = model(X)
        if moe:
            logits = logits[0]
        Y = Y.view(batch_size * seq_len)
        logits = logits.view(batch_size*seq_len, -1)
        loss = criterion(logits, Y)
        losses[k] = loss.item()
    out = losses.mean()
    model.train()

    return out