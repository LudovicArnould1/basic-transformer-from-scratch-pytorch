# python libraries
import sys


# Deep learning libraries
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.profiler


from datasets import load_dataset

# local libraries
from data_process import get_batch_from_raw, get_seq_batch, process_dataset
from model import Transformer, MoE_Transformer
from training_utils import estimate_loss


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# dataset parameters
seq_len = 4
dataset_len = 100
batch_size = 32
voc_size = 30000
# tokenizer parameters
tokenizer_params = {"vocab_size" : voc_size ,"min_frequency":2} 

# model parameters
emb_dim = 40
n_blocks = 2
n_head = 2
dropout_rate = 0.0

# training parameters
n_iter = 500
initial_lr = 1e-3
warmup_epochs = int(n_iter * 0.1) # set only if warmup is used


train_stream = load_dataset("wikitext", "wikitext-103-v1", split='train', streaming=True)
val_stream = load_dataset("wikitext", "wikitext-103-v1", split='validation', streaming=True)
 
# Process the dataset
train_path="train.txt"
train = process_dataset(train_stream, "new_train.txt", dataset_len, seq_len,
                          trained_tokenizer_path="bytelevel_bpe",  
                          truncation=True, padding=True,
                          tokenizer_params={})

val = process_dataset(val_stream, "new_val.txt", dataset_len, seq_len,
                      trained_tokenizer_path="bytelevel_bpe",  
                      truncation=True, padding=True, 
                      tokenizer_params={})






def train_model(model, n_iter, moe=False):
  for iter in range(1, n_iter+1):
    # zero grad
    optimizer.zero_grad()

    # get batch
    x,y = get_seq_batch(train, batch_size, seq_len)
    y = y.view(batch_size * seq_len)


    if not moe:
      preds = model(x)
      preds = preds.view(batch_size*seq_len, voc_size) # B*T,C
      loss = F.cross_entropy(preds, y)
    else:
      preds, load_loss, z_loss = model(x)
      data_loss = F.cross_entropy(preds.view(batch_size*seq_len, voc_size), y)
      loss = data_loss + load_loss + z_loss

    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if iter == n_iter - 1 or (iter % 10) == 0:
      val_loss = estimate_loss(val, model, moe=moe)
      print(f"step {iter}: train loss {loss:.4f}, val loss {val_loss:.4f}")


# The model
model_0 = Transformer(d=emb_dim, T=seq_len, n_head=n_head, d_ff=4*emb_dim, rotate=True,voc_size= voc_size,
              n_blocks=n_blocks, dropout_rate=dropout_rate, device=device).to(device)

model_1 = MoE_Transformer(d=emb_dim, T=seq_len, n_head=n_head, voc_size=voc_size,
                         d_ff=4*emb_dim, n_experts=4, topk=2, rotate=True,
                        dropout_rate=dropout_rate, device=device).to(device)

# Optimizer and scheduler settings

optimizer = torch.optim.Adam(model_1.parameters(), initial_lr)

#scheduler1 = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1,
#                                        total_iters = warmup_epochs)

#scheduler2 = optim.lr_scheduler.LinearLR(optimizer, start_factor=1.,
#                                          end_factor = 0.5,
#                                          total_iters = n_iter - warmup_epochs)

#scheduler = optim.lr_scheduler.SequentialLR(optimizer,
#                                            schedulers=[scheduler1, scheduler2],
#                                            milestones=[warmup_epochs])


train_model(model_1, 30, moe=True)


import torch
from experts import Gate
from experts import SparseMoE



x = torch.randn((32,10,60))
d = 60
n_experts = 4
d_ff = 120
k = 2




gate = Gate(60, 4, 2)
d = gate(x)


moe = SparseMoE(60, 120, 4, 2)
b = moe(x)

from model import MoEBlock
block = MoEBlock(60, 4, 120, False, 0.1, 4, 2)
blocks = torch.nn.Sequential(*[block for _ in range(4)])

for block in blocks:
    x, load_loss, z_loss = block(x)
  
print(x.shape, load_loss, z_loss)

import torch
torch.autograd.set_detect_anomaly(True)
moe = SparseMoE(60, 120, 4, 2)

optimizer = torch.optim.Adam(moe.parameters(), 1e-4)
optimizer.zero_grad()
x = torch.randn((32,10,60))
x, load_loss, z_loss = moe(x)
loss = load_loss + z_loss

loss.backward()

optimizer = torch.optim.Adam(gate.parameters(), 1e-4)
optimizer.zero_grad()
_, _, load_loss, z_loss = gate(x)

loss = load_loss + z_loss
loss.backward()