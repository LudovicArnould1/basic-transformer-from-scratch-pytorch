import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class Gate(nn.Module):
    """
    Gate network for SparseMoE.
    
    Args:
    - d (int): Input dimension.
    - n_experts (int): Number of experts.
    - k (int): Number of experts to select.
    """

    def __init__(self, d, n_experts=4, topk=2, alpha=1e-2, beta=0.1):
        super().__init__()

        self.n_experts = n_experts
        self.k = topk
        self.alpha = alpha
        self.beta = beta

        self.dispatch = nn.Linear(d, n_experts)

    def forward(self, x):
        """
        Args:
        - x (Tensor): Input tensor of shape [B, T, d]

        Returns:
        - Dispatch Tensor : Tensor of shape [B, T, E]
        - Combine Tensor : Tensor of shape [B, T, E]
        - Balance loss : Scalar
        - Z-router loss : Scalar
        """
        # Get dispatch scores
        dispatch_scores = self.dispatch(x)
        dispatch_scores = F.softmax(dispatch_scores, dim=-1)

        # Select top-k experts
        topk_values, topk_indices = dispatch_scores.topk(self.k, dim=-1)

        # Normalization
        combine_scores = topk_values / topk_values.sum(dim=-1, keepdim=True)

        if self.train():
             # Load balancing loss
            balance_loss = self.load_loss(dispatch_scores)
            # Z-routing loss
            z_routing_loss = self.z_router_loss(dispatch_scores)
        else:
            balance_loss, z_routing_loss = 0, 0

        return topk_indices, combine_scores, balance_loss, z_routing_loss

    
    def load_loss(self, dispatch_scores):
        """
        Compute the load balancing loss according to Fedus et al. 2022.
        Minimum when distribution is uniform.
    
        Args:
        - dispatch_scores (Tensor): Tensor of shape [B, T, E].
        
        Returns:
        - Tensor: Load balancing loss. 
        """
        # 'e' is the expert dimension
        f = reduce((dispatch_scores == dispatch_scores.max(dim=-1, keepdim=True).values).to(dtype=torch.float),
                     'b t e -> e', 'mean') 
        P = reduce(dispatch_scores, 'b t e -> e', 'mean') 

        # Overall load loss
        load_loss = self.alpha* (f*P).mean()

        return load_loss

    
    def z_router_loss(self, dispatch_scores):
        """
        Computes the Z-Router loss.

        Args:
        - x (torch.Tensor): A tensor of shape [B, N] representing the router logits for each expert (N) for each item in the batch (B).

        Returns:
        - torch.Tensor: The computed Z-Router loss.
        """
        # Inner sum and log
        log_sum_exp = torch.logsumexp(dispatch_scores, dim=-1) 

        # Squared
        squared_log_sum_exp = log_sum_exp ** 2

        # Mean times beta
        loss = self.beta * squared_log_sum_exp.mean()

        return loss


class SparseMoE(nn.Module):
    # Sparse Mixture of Experts
    # Implements a gating network and a sparse expert network

    def __init__(self, d, d_ff, n_experts, topk=1, dropout_rate=0.1):
        super().__init__()

        from model import MLP

        self.n_experts = n_experts
        self.d = d
        self.gate = Gate(d, n_experts, topk)
        self.experts = nn.ModuleList([MLP(d, d_ff, dropout_rate) for _ in range(n_experts)])
        
    def forward(self, x):
        """        
        Args:
        - x (Tensor): Input tensor of shape [batch_size, d_in].
        
        Returns:
        - Tensor: Output tensor of shape [batch_size, d].
        """
        # Get gating scores and losses
        # topk shape : [B, T, k]
        topk_idx, combine_scores, load_loss, z_loss = self.gate(x) 
        
        flat_idx = rearrange(topk_idx, 'b t k -> (b t) k')
        flat_scores = rearrange(combine_scores, 'b t k -> (b t) k')

        # flat scores to [B, T, E]
        # with 0s for the experts that are not selected and original score otherwise
        flat_scores = torch.nn.functional.one_hot(flat_idx, num_classes=self.n_experts).float() * flat_scores.unsqueeze(-1)
        flat_scores = reduce(flat_scores, '... k e -> ... e', 'sum')

        flat_x = rearrange(x, 'b t d -> (b t) d')

        # Storing results
        out = torch.zeros_like(flat_x)


        for expert_idx, expert in enumerate(self.experts):
            # Idx for current expert
            indices_current_expert = (flat_idx == expert_idx).any(dim=-1)
            
            # Processing x 
            out[indices_current_expert] = expert(flat_x[indices_current_expert])

            # Multiplying by the combine scores
            out[indices_current_expert] *= flat_scores[indices_current_expert, expert_idx].unsqueeze(-1)

        # Reshape
        out = rearrange(out, '(b t) d -> b t d', b=x.shape[0], t=x.shape[1])
        
        # Skip connection 
        x = x + out

        return x, load_loss, z_loss
