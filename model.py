import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from experts import SparseMoE


def rotation(x, theta):
   
   # x: (B, T, d)
  B,T,d = x.shape

  # Shape to [1,1, d]
  theta = rearrange(theta, 'd -> 1 1 d')

  idx = torch.arange(1,T+1, device=x.device).float().unsqueeze(-1)

  # shape [T, d]
  cos_itheta = torch.cos(idx * theta)
  sin_itheta = torch.sin(idx * theta)

  # first part with cosines
  res_cos = x * cos_itheta

  # second part with sinus has negative sign every other element
  swap_indices = torch.arange(d).view(-1, 2)[:, [1, 0]].flatten()
  x[..., ::2] = - x[..., ::2]
  x = x[..., swap_indices]

  res_sin = x * sin_itheta

  return res_cos + res_sin


# Designing a Transformer

class AttentionHead(nn.Module):
    # One attention head (vanilla)

  def __init__(self, d, d_H, dropout_rate=0.1):
    super().__init__()

    self.A_Q = nn.Linear(d, d_H, bias=False)
    self.A_K = nn.Linear(d, d_H, bias=False)
    self.A_V = nn.Linear(d, d_H, bias=False)
    self.dropout = nn.Dropout(dropout_rate)
    self.d = d

  def forward(self,x):
    
    B,T,d = x.shape

    Q = self.A_Q(x)
    K = self.A_K(x)
    V = self.A_V(x)

    w = Q @ K.transpose(-2,-1) * self.d**(-0.5)

    # Masking
    w = w.masked_fill(torch.tril(torch.ones(T, T, device=w.device), 
                             diagonal=0) == 0, float('-inf'))
    w = F.softmax(w, dim=-1) @ V

    return w


class RoPEAttentionHead(AttentionHead):
    # Attention Head with Rotary Positional Encoding (RoPE)

  def __init__(self, d, d_H, dropout_rate=0.1):
    super().__init__(d, d_H, dropout_rate)

    self.theta = torch.tensor([10**(-2*(i-1)/d) for i in range(d_H//2) for _ in range(2)])

  def forward(self,x):
    
    B,T,d = x.shape

    Q = self.A_Q(x)
    K = self.A_K(x)
    V = self.A_V(x)

    # Rotations
    Q = rotation(Q, self.theta)
    K = rotation(K, self.theta)

    w = Q @ K.transpose(-2,-1) * self.d**(-0.5)

    # Masking
    w = w.masked_fill(torch.tril(torch.ones(T, T, device=w.device), 
                             diagonal=0) == 0, float('-inf'))
    w = F.softmax(w, dim=-1) @ V

    return w



class MultiHeadAttention(nn.Module):
  # Multihead Attention module (vanilla)

  def __init__(self, d, n_head, rotate=False, dropout_rate=0.1):
    super().__init__()

    if rotate:
      self.heads = nn.ModuleList([RoPEAttentionHead(d, d_H = d//n_head, 
                                              dropout_rate=dropout_rate)
                                for _ in range(n_head)]
                              )
    else:
      self.heads = nn.ModuleList([AttentionHead(d, d_H = d//n_head, 
                                              dropout_rate=dropout_rate)
                                for _ in range(n_head)]
                              )
    self.linear = nn.Linear(d,d)
    self.dropout = nn.Dropout(dropout_rate)

  def forward(self, x):

    w_list = [head(x) for head in self.heads]
    w = torch.cat(w_list, dim=-1)

    w = self.dropout(self.linear(w))

    return w


class MLP(nn.Module):

  def __init__(self, d, d_ff, dropout_rate=0.1):
    super().__init__()

    self.lin_1 = nn.Linear(d, d_ff)
    self.lin_2 = nn.Linear(d_ff, d)
    self.dropout = nn.Dropout(dropout_rate)

    self.d_ff = d_ff

  def forward(self,x):

    x = F.relu(self.lin_1(x))
    x = self.lin_2(x)
    x = self.dropout(x)

    return x


class Block(nn.Module):

  def __init__(self, d, n_head, d_ff, rotate=False, dropout_rate=0.1):
    super().__init__()

    self.norm1 = nn.LayerNorm(d)
    self.norm2 = nn.LayerNorm(d)
    self.att = MultiHeadAttention(d, n_head, rotate=rotate, dropout_rate=dropout_rate)
    self.mlp = MLP(d, d_ff, dropout_rate)

  def forward(self, x):

    x = self.norm1(x)
    w = self.att(x)

    x = self.norm2(x+w)
    x = self.mlp(x) + x

    return x
  

class MoEBlock(nn.Module):
    """
    Similar to Switch-transformer MoE, but normalization at the beginning of the block 
    """
  
    def __init__(self, d, n_head, d_ff, rotate=False, dropout_rate=0.1, n_experts=4, topk=2):
      super().__init__()
  
      self.norm1 = nn.LayerNorm(d)
      self.norm2 = nn.LayerNorm(d)
      self.att = MultiHeadAttention(d, n_head, rotate=rotate, dropout_rate=dropout_rate)
      self.moe = SparseMoE(d, d_ff, n_experts, topk, dropout_rate)
      self.mlp = MLP(d, d_ff, dropout_rate)
  
    def forward(self, x):
  
      x = self.norm1(x)
      w = self.att(x)
  
      x = self.norm2(x+w)
      # The skip connection is included in the MoE layer
      x, load_loss, z_loss = self.moe(x)

      return x, load_loss, z_loss


class Transformer(nn.Module):

  def __init__(self, T, d, n_head, d_ff, voc_size, n_blocks=4, rotate=False, dropout_rate=0.1, device="cuda"):
    super().__init__()

    if rotate:
      self.emb = nn.Embedding(voc_size, d)
    else:
      self.emb = nn.Sequential(nn.Embedding(voc_size, d))
      self.pos_emb = nn.Embedding(T,d)
    
    self.blocks = nn.Sequential(*[Block(d, n_head, d_ff, rotate, dropout_rate) 
                                    for _ in range(n_blocks)])
    self.ln_f = nn.LayerNorm(d)

    self.linear_f = nn.Linear(d, voc_size)

    self.seq_len = T
    self.rotate = rotate
    
    self.device = device


    print("Model has {:.4f} M trainable parameters".format(count_parameters(self)/1e6))

  def forward(self, x):
    B,T = x.shape

    x = self.emb(x)

    if not self.rotate:
      pos_emb = self.pos_emb(torch.arange(T, device=self.device)) 
      x += pos_emb        # learned positional encoder at the beginning of the transformer

    x = self.blocks(x)
    x = self.ln_f(x)
    # Final linear
    x = self.linear_f(x)        # B,T, voc_size

    return x

  def generate(self, idx, max_new_tokens=200):
      # idx is (B, T) array of indices in the current context
      for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.seq_len:]
            # get the predictions
            logits = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, d)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, d)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
      return idx




def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class MoE_Transformer(Transformer):
    """
    Similar to switch-transformer, use positional encoding at the beginning of each block
    """
    def __init__(self, T, d, n_head, d_ff, voc_size, n_blocks=4, rotate=False, dropout_rate=0.1, 
                 device="cuda", n_experts=4, topk=2):
      
      super().__init__(T, d, n_head, d_ff, voc_size, n_blocks, rotate, dropout_rate, device)

      self.emb = nn.Embedding(voc_size, d)
      if not rotate:
        self.pos_emb = nn.Embedding(T, d)
      self.blocks = nn.Sequential(*[MoEBlock(d, n_head, d_ff, rotate=rotate, 
                                             dropout_rate=dropout_rate,
                                             n_experts=n_experts, topk=topk) for _ in range(n_blocks)])

      self.ln_f = nn.LayerNorm(d)
      self.linear_f = nn.Linear(d, voc_size)

      self.rotate = rotate
      self.device = device

      print("Model has {:.4f} M trainable parameters".format(count_parameters(self)/1e6))

    def forward(self, x):
      B,T = x.shape
      
      # Embeding
      x = self.emb(x)

      if not self.rotate:
        pos_emb = self.pos_emb(torch.arange(T, device=self.device)) 

      load_loss = 0
      z_loss = 0

      for block in self.blocks:
        if not self.rotate:
          x += pos_emb       # Add positional encoding (as in SwitchTrasnformer)
          x, load_loss, z_loss = block(x)

          load_loss += load_loss; z_loss += z_loss
        else:
          x, load_loss, z_loss = block(x)
          load_loss += load_loss; z_loss += z_loss

      x = self.ln_f(x)
      # Final linear
      x = self.linear_f(x)        # B,T, voc_size

      return x, load_loss, z_loss
  

class PositionalEncoding(nn.Module):  #@save
    """Positional encoding."""
    def __init__(self, d, max_len=1000):
        super().__init__()
        # Create a long enough P
        self.P = torch.zeros((1, max_len, d))
        x = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d, 2, dtype=torch.float32) / d)
        self.P[:, :, 0::2] = torch.sin(x)
        self.P[:, :, 1::2] = torch.cos(x)

    def forward(self, x):
        x = x + self.P[:, :x.shape[1], :]
        return x
