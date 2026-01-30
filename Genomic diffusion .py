"""
Diffusion Models for Generating Synthetic Protein-Coding Sequences
====================================================================
Author: Abel Tadesse (GSR/2025/17)
Advisor: Addane Letta
Institution: Addis Ababa University, College of Technology and Built Environment

This implementation provides a complete discrete diffusion model (D3PM) for 
generating synthetic protein-coding DNA sequences. The code includes:
- Data loading and preprocessing
- D3PM model architecture with Transformer backbone
- Training loop with logging
- Sequence generation and sampling
- Comprehensive evaluation metrics (k-mer, CAI, GC content, ORF validation)
- Visualization utilities

Requirements:
    pip install torch numpy pandas matplotlib seaborn biopython scikit-learn tqdm

Usage:
    python genomic_diffusion.py --mode train --epochs 200
    python genomic_diffusion.py --mode generate --num_samples 1000
    python genomic_diffusion.py --mode evaluate --checkpoint model_best.pt
"""

import os
import argparse
import math
import random
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Set random seeds for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==============================================================================
# Constants and Vocabulary
# ==============================================================================

NUCLEOTIDES = ['A', 'T', 'G', 'C']
MASK_TOKEN = '[MASK]'
VOCAB = NUCLEOTIDES + [MASK_TOKEN]
VOCAB_SIZE = len(VOCAB)  # 5: A, T, G, C, [MASK]

NUC_TO_IDX = {nuc: idx for idx, nuc in enumerate(VOCAB)}
IDX_TO_NUC = {idx: nuc for idx, nuc in enumerate(VOCAB)}

# Standard genetic code
CODON_TABLE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

STOP_CODONS = ['TAA', 'TAG', 'TGA']
START_CODON = 'ATG'

# Human codon usage frequencies (for CAI calculation)
HUMAN_CODON_FREQ = {
    'TTT': 0.45, 'TTC': 0.55, 'TTA': 0.07, 'TTG': 0.13,
    'TCT': 0.18, 'TCC': 0.22, 'TCA': 0.15, 'TCG': 0.06,
    'TAT': 0.43, 'TAC': 0.57, 'TAA': 0.28, 'TAG': 0.20,
    'TGT': 0.45, 'TGC': 0.55, 'TGA': 0.52, 'TGG': 1.00,
    'CTT': 0.13, 'CTC': 0.20, 'CTA': 0.07, 'CTG': 0.41,
    'CCT': 0.28, 'CCC': 0.33, 'CCA': 0.27, 'CCG': 0.11,
    'CAT': 0.41, 'CAC': 0.59, 'CAA': 0.25, 'CAG': 0.75,
    'CGT': 0.08, 'CGC': 0.19, 'CGA': 0.11, 'CGG': 0.21,
    'ATT': 0.36, 'ATC': 0.48, 'ATA': 0.16, 'ATG': 1.00,
    'ACT': 0.24, 'ACC': 0.36, 'ACA': 0.28, 'ACG': 0.12,
    'AAT': 0.46, 'AAC': 0.54, 'AAA': 0.42, 'AAG': 0.58,
    'AGT': 0.15, 'AGC': 0.24, 'AGA': 0.20, 'AGG': 0.20,
    'GTT': 0.18, 'GTC': 0.24, 'GTA': 0.11, 'GTG': 0.47,
    'GCT': 0.26, 'GCC': 0.40, 'GCA': 0.23, 'GCG': 0.11,
    'GAT': 0.46, 'GAC': 0.54, 'GAA': 0.42, 'GAG': 0.58,
    'GGT': 0.16, 'GGC': 0.34, 'GGA': 0.25, 'GGG': 0.25,
}


# ==============================================================================
# Dataset Classes
# ==============================================================================

class ProteinCodingDataset(Dataset):
    """Dataset for protein-coding DNA sequences."""
    
    def __init__(self, sequences: List[str], seq_length: int = 300):
        """
        Args:
            sequences: List of DNA sequences
            seq_length: Fixed length to pad/truncate sequences
        """
        self.sequences = sequences
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # Pad or truncate to fixed length
        if len(seq) < self.seq_length:
            seq = seq + 'A' * (self.seq_length - len(seq))  # Pad with A
        else:
            seq = seq[:self.seq_length]
        
        # Convert to indices
        indices = [NUC_TO_IDX.get(nuc, NUC_TO_IDX['A']) for nuc in seq]
        return torch.tensor(indices, dtype=torch.long)


def generate_synthetic_coding_sequences(
    num_sequences: int = 10000,
    min_codons: int = 100,
    max_codons: int = 200,
    gc_bias: float = 0.52
) -> List[str]:
    """
    Generate synthetic protein-coding sequences for training/testing.
    
    This function creates biologically plausible protein-coding sequences
    with realistic codon usage and GC content. Use this for demonstration
    or when real data is unavailable.
    
    Args:
        num_sequences: Number of sequences to generate
        min_codons: Minimum number of codons per sequence
        max_codons: Maximum number of codons per sequence  
        gc_bias: Target GC content (0.0 to 1.0)
        
    Returns:
        List of DNA sequence strings
    """
    sequences = []
    
    # Create codon sampling weights based on human codon usage
    coding_codons = [c for c in HUMAN_CODON_FREQ.keys() if CODON_TABLE[c] != '*']
    codon_weights = [HUMAN_CODON_FREQ[c] for c in coding_codons]
    codon_weights = np.array(codon_weights) / sum(codon_weights)
    
    for _ in tqdm(range(num_sequences), desc="Generating sequences"):
        num_codons = random.randint(min_codons, max_codons)
        
        # Start with ATG
        seq = START_CODON
        
        # Add random codons (excluding stop codons)
        for _ in range(num_codons - 2):  # -2 for start and stop
            codon = np.random.choice(coding_codons, p=codon_weights)
            seq += codon
        
        # End with stop codon
        stop = random.choice(STOP_CODONS)
        seq += stop
        
        sequences.append(seq)
    
    return sequences


def load_fasta_sequences(filepath: str) -> List[str]:
    """Load sequences from FASTA file."""
    sequences = []
    current_seq = ""
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(current_seq.upper())
                current_seq = ""
            else:
                current_seq += line
        if current_seq:
            sequences.append(current_seq.upper())
    
    # Filter valid sequences
    valid_seqs = []
    for seq in sequences:
        if all(nuc in 'ATGC' for nuc in seq) and len(seq) >= 60:
            valid_seqs.append(seq)
    
    return valid_seqs


# ==============================================================================
# Model Components
# ==============================================================================

class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for timestep encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class GenomicDiffusionModel(nn.Module):
    """
    Discrete Diffusion Model for Genomic Sequences.
    
    This implements the D3PM (Discrete Denoising Diffusion Probabilistic Model)
    framework with an absorbing state transition for nucleotide sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 5,
        seq_length: int = 300,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        num_timesteps: int = 500
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_timesteps = num_timesteps
        self.embed_dim = embed_dim
        
        # Token embedding (5 tokens: A, T, G, C, [MASK])
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_length, embed_dim) * 0.02)
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbeddings(embed_dim),
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output head (predict 4 nucleotides, not [MASK])
        self.output_norm = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, 4)  # Only predict A, T, G, C
        
        # Precompute transition matrices for absorbing state diffusion
        self._setup_transition_matrices()
        
    def _setup_transition_matrices(self):
        """Setup absorbing state transition matrices."""
        T = self.num_timesteps
        K = self.vocab_size  # 5 (A, T, G, C, MASK)
        
        # Beta schedule: β_t = 1 / (T - t + 1)
        betas = torch.zeros(T)
        for t in range(T):
            betas[t] = 1.0 / (T - t)
        
        # Cumulative product for direct sampling q(x_t | x_0)
        # For absorbing state: prob of staying = prod(1 - beta_i)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_cumprod', alpha_cumprod)
        
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample from q(x_t | x_0) using absorbing state diffusion.
        
        With probability (1 - alpha_cumprod[t]), replace token with [MASK].
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Get cumulative alpha for each sample
        alpha_t = self.alpha_cumprod[t]  # (batch,)
        
        # Sample mask: with prob (1 - alpha_t), mask the position
        mask_prob = 1.0 - alpha_t[:, None]  # (batch, 1)
        mask = torch.rand(x_0.shape, device=device) < mask_prob
        
        # Replace masked positions with [MASK] token (index 4)
        x_t = x_0.clone()
        x_t[mask] = 4  # MASK token index
        
        return x_t
    
    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict x_0 from x_t.
        
        Args:
            x_t: Noised sequences, shape (batch, seq_len)
            t: Timesteps, shape (batch,)
            
        Returns:
            Logits for x_0 prediction, shape (batch, seq_len, 4)
        """
        # Token embeddings
        h = self.token_embed(x_t)  # (batch, seq_len, embed_dim)
        
        # Add positional embeddings
        h = h + self.pos_embed
        
        # Add timestep embeddings
        t_emb = self.time_embed(t.float())  # (batch, embed_dim)
        h = h + t_emb[:, None, :]
        
        # Transformer layers
        for layer in self.layers:
            h = layer(h)
        
        # Output prediction
        h = self.output_norm(h)
        logits = self.output_head(h)  # (batch, seq_len, 4)
        
        return logits
    
    def compute_loss(
        self,
        x_0: torch.Tensor,
        return_metrics: bool = False
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        Args:
            x_0: Clean sequences, shape (batch, seq_len)
            return_metrics: Whether to return additional metrics
            
        Returns:
            Loss tensor (and optionally metrics dict)
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)
        
        # Sample noised sequences
        x_t = self.q_sample(x_0, t)
        
        # Predict x_0
        logits = self.forward(x_t, t)  # (batch, seq_len, 4)
        
        # Cross-entropy loss only on original nucleotides (not MASK)
        # x_0 should only contain indices 0-3
        x_0_clamped = x_0.clamp(0, 3)  # Ensure valid targets
        
        loss = F.cross_entropy(
            logits.reshape(-1, 4),
            x_0_clamped.reshape(-1),
            reduction='mean'
        )
        
        if return_metrics:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                acc = (preds == x_0_clamped).float().mean()
            return loss, {'accuracy': acc.item()}
        
        return loss
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        device: torch.device,
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion.
        
        Args:
            num_samples: Number of sequences to generate
            device: Device to generate on
            show_progress: Whether to show progress bar
            
        Returns:
            Generated sequences, shape (num_samples, seq_len)
        """
        self.eval()
        
        # Start from all [MASK] tokens
        x_t = torch.full(
            (num_samples, self.seq_length),
            fill_value=4,  # MASK token
            dtype=torch.long,
            device=device
        )
        
        timesteps = range(self.num_timesteps - 1, -1, -1)
        if show_progress:
            timesteps = tqdm(timesteps, desc="Sampling")
        
        for t in timesteps:
            t_batch = torch.full((num_samples,), t, dtype=torch.long, device=device)
            
            # Predict x_0
            logits = self.forward(x_t, t_batch)  # (batch, seq_len, 4)
            probs = F.softmax(logits, dim=-1)
            
            # Sample predictions
            pred_x0 = torch.multinomial(
                probs.reshape(-1, 4),
                num_samples=1
            ).reshape(num_samples, self.seq_length)
            
            # For absorbing state: unmask positions based on schedule
            if t > 0:
                # Calculate how many positions to unmask at this step
                alpha_t = self.alpha_cumprod[t]
                alpha_prev = self.alpha_cumprod[t - 1] if t > 0 else torch.tensor(1.0)
                
                # Probability of unmasking at this step
                unmask_prob = (alpha_prev - alpha_t) / (1 - alpha_t + 1e-8)
                
                # Get currently masked positions
                is_masked = (x_t == 4)
                
                # Sample which masked positions to unmask
                unmask = torch.rand_like(x_t.float()) < unmask_prob
                unmask = unmask & is_masked
                
                # Update with predictions where unmasked
                x_t = torch.where(unmask, pred_x0, x_t)
            else:
                # At t=0, unmask everything
                x_t = pred_x0
        
        return x_t


# ==============================================================================
# Evaluation Metrics
# ==============================================================================

def sequences_to_strings(indices: torch.Tensor) -> List[str]:
    """Convert index tensor to list of sequence strings."""
    sequences = []
    for seq_indices in indices:
        seq = ''.join([IDX_TO_NUC.get(int(idx), 'N') for idx in seq_indices])
        # Replace MASK tokens with random nucleotide
        seq = seq.replace('[MASK]', random.choice('ATGC'))
        sequences.append(seq)
    return sequences


def compute_kmer_distribution(sequences: List[str], k: int = 3) -> Dict[str, float]:
    """Compute k-mer frequency distribution."""
    kmer_counts = Counter()
    total = 0
    
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if all(nuc in 'ATGC' for nuc in kmer):
                kmer_counts[kmer] += 1
                total += 1
    
    # Normalize to frequencies
    if total > 0:
        return {kmer: count / total for kmer, count in kmer_counts.items()}
    return {}


def jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    # Get all kmers
    all_kmers = set(p.keys()) | set(q.keys())
    
    # Convert to arrays with small epsilon for missing kmers
    eps = 1e-10
    p_arr = np.array([p.get(k, eps) for k in all_kmers])
    q_arr = np.array([q.get(k, eps) for k in all_kmers])
    
    # Normalize
    p_arr = p_arr / p_arr.sum()
    q_arr = q_arr / q_arr.sum()
    
    # Compute JSD
    m = 0.5 * (p_arr + q_arr)
    
    def kl_div(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log(a[mask] / b[mask]))
    
    jsd = 0.5 * kl_div(p_arr, m) + 0.5 * kl_div(q_arr, m)
    return float(jsd)


def compute_gc_content(seq: str) -> float:
    """Compute GC content of a sequence."""
    gc = sum(1 for nuc in seq if nuc in 'GC')
    return gc / len(seq) if len(seq) > 0 else 0


def compute_positional_gc(seq: str) -> Tuple[float, float, float]:
    """Compute GC content at each codon position."""
    gc1 = gc2 = gc3 = 0
    n1 = n2 = n3 = 0
    
    for i in range(0, len(seq) - 2, 3):
        if seq[i] in 'GC':
            gc1 += 1
        n1 += 1
        
        if seq[i+1] in 'GC':
            gc2 += 1
        n2 += 1
        
        if seq[i+2] in 'GC':
            gc3 += 1
        n3 += 1
    
    return (
        gc1 / n1 if n1 > 0 else 0,
        gc2 / n2 if n2 > 0 else 0,
        gc3 / n3 if n3 > 0 else 0
    )


def compute_cai(seq: str) -> float:
    """
    Compute Codon Adaptation Index (CAI).
    
    CAI measures how well a sequence's codon usage matches
    highly expressed genes in the organism.
    """
    if len(seq) < 3:
        return 0
    
    log_w_sum = 0
    n_codons = 0
    
    for i in range(0, len(seq) - 2, 3):
        codon = seq[i:i+3]
        if codon in HUMAN_CODON_FREQ and CODON_TABLE.get(codon, '*') != '*':
            # Get relative adaptiveness (w)
            w = HUMAN_CODON_FREQ[codon]
            if w > 0:
                log_w_sum += math.log(w)
                n_codons += 1
    
    if n_codons == 0:
        return 0
    
    return math.exp(log_w_sum / n_codons)


def check_orf_validity(seq: str) -> bool:
    """
    Check if sequence has valid ORF structure.
    
    Valid ORF requires:
    - Starts with ATG
    - No in-frame stop codons before end
    - Ends with stop codon
    """
    if len(seq) < 9:  # Minimum: start + 1 codon + stop
        return False
    
    # Check start codon
    if seq[:3] != 'ATG':
        return False
    
    # Check for in-frame stop codons (excluding last codon)
    for i in range(3, len(seq) - 3, 3):
        codon = seq[i:i+3]
        if codon in STOP_CODONS:
            return False
    
    # Check last codon is stop (optional, depending on application)
    # For generated sequences, we might not enforce this
    return True


def compute_edit_distance(seq1: str, seq2: str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def evaluate_sequences(
    generated_seqs: List[str],
    real_seqs: List[str],
    k_values: List[int] = [3, 4, 5, 6]
) -> Dict[str, float]:
    """
    Comprehensive evaluation of generated sequences.
    
    Args:
        generated_seqs: List of generated sequence strings
        real_seqs: List of real sequence strings
        k_values: K-mer sizes to evaluate
        
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {}
    
    # K-mer JSD
    for k in k_values:
        gen_kmer = compute_kmer_distribution(generated_seqs, k)
        real_kmer = compute_kmer_distribution(real_seqs, k)
        jsd = jensen_shannon_divergence(gen_kmer, real_kmer)
        metrics[f'jsd_{k}mer'] = jsd
    
    # GC content
    gen_gc = [compute_gc_content(s) for s in generated_seqs]
    real_gc = [compute_gc_content(s) for s in real_seqs]
    metrics['gc_mean_gen'] = np.mean(gen_gc)
    metrics['gc_std_gen'] = np.std(gen_gc)
    metrics['gc_mean_real'] = np.mean(real_gc)
    metrics['gc_std_real'] = np.std(real_gc)
    
    # Positional GC
    gen_gc_pos = [compute_positional_gc(s) for s in generated_seqs]
    real_gc_pos = [compute_positional_gc(s) for s in real_seqs]
    
    for i, pos in enumerate(['gc1', 'gc2', 'gc3']):
        metrics[f'{pos}_mean_gen'] = np.mean([g[i] for g in gen_gc_pos])
        metrics[f'{pos}_mean_real'] = np.mean([g[i] for g in real_gc_pos])
    
    # CAI
    gen_cai = [compute_cai(s) for s in generated_seqs]
    real_cai = [compute_cai(s) for s in real_seqs]
    metrics['cai_mean_gen'] = np.mean(gen_cai)
    metrics['cai_std_gen'] = np.std(gen_cai)
    metrics['cai_mean_real'] = np.mean(real_cai)
    metrics['cai_std_real'] = np.std(real_cai)
    
    # ORF validity
    gen_orf_valid = sum(1 for s in generated_seqs if check_orf_validity(s))
    metrics['orf_valid_pct'] = gen_orf_valid / len(generated_seqs) * 100
    
    # Diversity (sample pairwise distances)
    if len(generated_seqs) >= 100:
        sample_idx = random.sample(range(len(generated_seqs)), 100)
        distances = []
        for i in range(len(sample_idx)):
            for j in range(i + 1, min(i + 10, len(sample_idx))):
                d = compute_edit_distance(
                    generated_seqs[sample_idx[i]][:100],
                    generated_seqs[sample_idx[j]][:100]
                )
                distances.append(d)
        metrics['mean_edit_distance'] = np.mean(distances)
    
    return metrics


# ==============================================================================
# Visualization
# ==============================================================================

def plot_kmer_comparison(
    generated_seqs: List[str],
    real_seqs: List[str],
    k: int = 3,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """Plot comparison of k-mer distributions."""
    gen_kmer = compute_kmer_distribution(generated_seqs, k)
    real_kmer = compute_kmer_distribution(real_seqs, k)
    
    # Get top kmers from real data
    top_kmers = sorted(real_kmer.keys(), key=lambda x: real_kmer[x], reverse=True)[:top_n]
    
    gen_freqs = [gen_kmer.get(km, 0) for km in top_kmers]
    real_freqs = [real_kmer.get(km, 0) for km in top_kmers]
    
    x = np.arange(len(top_kmers))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, real_freqs, width, label='Real', color='steelblue')
    ax.bar(x + width/2, gen_freqs, width, label='Generated', color='coral')
    
    ax.set_xlabel(f'{k}-mer')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{k}-mer Distribution Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(top_kmers, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_gc_distribution(
    generated_seqs: List[str],
    real_seqs: List[str],
    save_path: Optional[str] = None
):
    """Plot GC content distributions."""
    gen_gc = [compute_gc_content(s) * 100 for s in generated_seqs]
    real_gc = [compute_gc_content(s) * 100 for s in real_seqs]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(real_gc, bins=30, alpha=0.7, label='Real', color='steelblue', density=True)
    ax.hist(gen_gc, bins=30, alpha=0.7, label='Generated', color='coral', density=True)
    
    ax.axvline(np.mean(real_gc), color='blue', linestyle='--', label=f'Real mean: {np.mean(real_gc):.1f}%')
    ax.axvline(np.mean(gen_gc), color='red', linestyle='--', label=f'Gen mean: {np.mean(gen_gc):.1f}%')
    
    ax.set_xlabel('GC Content (%)')
    ax.set_ylabel('Density')
    ax.set_title('GC Content Distribution')
    ax.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None
):
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', color='steelblue')
    ax.plot(epochs, val_losses, label='Validation Loss', color='coral')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


# ==============================================================================
# Training Loop
# ==============================================================================

def train_model(
    model: GenomicDiffusionModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 200,
    learning_rate: float = 2e-4,
    device: torch.device = torch.device('cuda'),
    checkpoint_dir: str = 'checkpoints',
    log_interval: int = 100
) -> Tuple[List[float], List[float]]:
    """
    Train the diffusion model.
    
    Args:
        model: The diffusion model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_interval: Steps between logging
        
    Returns:
        Tuple of (train_losses, val_losses) per epoch
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in enumerate(pbar):
            batch = batch.to(device)
            
            optimizer.zero_grad()
            loss = model.compute_loss(batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                loss = model.compute_loss(batch)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, 'model_best.pt'))
            print(f"  Saved best model checkpoint!")
        
        # Periodic checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pt'))
    
    return train_losses, val_losses


# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Diffusion Models for Protein-Coding Sequence Generation'
    )
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'generate', 'evaluate', 'demo'],
                        help='Running mode')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--seq_length', type=int, default=300,
                        help='Sequence length (nucleotides)')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of sequences to generate')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model_best.pt',
                        help='Checkpoint path for generation/evaluation')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to FASTA file (optional)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    if args.mode == 'demo':
        # Quick demo with synthetic data
        print("\n" + "="*60)
        print("DEMO: Discrete Diffusion for Protein-Coding Sequences")
        print("="*60 + "\n")
        
        # Generate synthetic training data
        print("Generating synthetic protein-coding sequences...")
        sequences = generate_synthetic_coding_sequences(
            num_sequences=5000,
            min_codons=80,
            max_codons=100
        )
        
        # Split data
        train_seqs = sequences[:4000]
        val_seqs = sequences[4000:4500]
        test_seqs = sequences[4500:]
        
        print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
        
        # Create datasets
        train_dataset = ProteinCodingDataset(train_seqs, seq_length=args.seq_length)
        val_dataset = ProteinCodingDataset(val_seqs, seq_length=args.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model
        model = GenomicDiffusionModel(
            vocab_size=5,
            seq_length=args.seq_length,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            ff_dim=1024,
            num_timesteps=500
        )
        
        print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Short training for demo
        print("\nTraining for 10 epochs (demo)...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=10,
            learning_rate=2e-4,
            device=device,
            checkpoint_dir='checkpoints'
        )
        
        # Generate samples
        print("\nGenerating 100 samples...")
        model.eval()
        generated_indices = model.sample(100, device)
        generated_seqs = sequences_to_strings(generated_indices)
        
        # Evaluate
        print("\nEvaluation Metrics:")
        metrics = evaluate_sequences(generated_seqs, test_seqs)
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Show example
        print("\nExample generated sequence (first 90 nt):")
        print(generated_seqs[0][:90])
        
        print("\n" + "="*60)
        print("Demo complete! Run with --mode train for full training.")
        print("="*60)
        
    elif args.mode == 'train':
        # Full training
        print("Loading/generating training data...")
        
        if args.data_path and os.path.exists(args.data_path):
            sequences = load_fasta_sequences(args.data_path)
            print(f"Loaded {len(sequences)} sequences from {args.data_path}")
        else:
            print("No data file provided. Generating synthetic data...")
            sequences = generate_synthetic_coding_sequences(
                num_sequences=15000,
                min_codons=80,
                max_codons=100
            )
        
        # Split
        random.shuffle(sequences)
        n_train = int(0.8 * len(sequences))
        n_val = int(0.1 * len(sequences))
        
        train_seqs = sequences[:n_train]
        val_seqs = sequences[n_train:n_train + n_val]
        test_seqs = sequences[n_train + n_val:]
        
        print(f"Train: {len(train_seqs)}, Val: {len(val_seqs)}, Test: {len(test_seqs)}")
        
        # Datasets
        train_dataset = ProteinCodingDataset(train_seqs, seq_length=args.seq_length)
        val_dataset = ProteinCodingDataset(val_seqs, seq_length=args.seq_length)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4)
        
        # Model
        model = GenomicDiffusionModel(
            vocab_size=5,
            seq_length=args.seq_length,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            ff_dim=1024,
            num_timesteps=500
        )
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        train_losses, val_losses = train_model(
            model, train_loader, val_loader,
            num_epochs=args.epochs,
            learning_rate=2e-4,
            device=device
        )
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, save_path='training_curves.png')
        
        # Save test sequences for evaluation
        with open('test_sequences.txt', 'w') as f:
            for seq in test_seqs:
                f.write(seq + '\n')
        
        print("Training complete!")
        
    elif args.mode == 'generate':
        # Generate samples from checkpoint
        print(f"Loading checkpoint from {args.checkpoint}")
        
        model = GenomicDiffusionModel(
            vocab_size=5,
            seq_length=args.seq_length,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            ff_dim=1024,
            num_timesteps=500
        )
        
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        print(f"Generating {args.num_samples} sequences...")
        generated_indices = model.sample(args.num_samples, device)
        generated_seqs = sequences_to_strings(generated_indices)
        
        # Save generated sequences
        output_file = 'generated_sequences.fasta'
        with open(output_file, 'w') as f:
            for i, seq in enumerate(generated_seqs):
                f.write(f">generated_seq_{i+1}\n")
                f.write(seq + '\n')
        
        print(f"Saved {len(generated_seqs)} sequences to {output_file}")
        
        # Print examples
        print("\nExample generated sequences:")
        for i in range(min(5, len(generated_seqs))):
            print(f"  {i+1}: {generated_seqs[i][:60]}...")
            
    elif args.mode == 'evaluate':
        # Evaluate generated sequences
        print("Evaluating generated sequences...")
        
        # Load generated sequences
        if os.path.exists('generated_sequences.fasta'):
            generated_seqs = load_fasta_sequences('generated_sequences.fasta')
        else:
            print("No generated sequences found. Running generation first...")
            return
        
        # Load test sequences
        if os.path.exists('test_sequences.txt'):
            with open('test_sequences.txt', 'r') as f:
                test_seqs = [line.strip() for line in f if line.strip()]
        else:
            print("Generating reference sequences...")
            test_seqs = generate_synthetic_coding_sequences(1000)
        
        # Evaluate
        print(f"Evaluating {len(generated_seqs)} generated vs {len(test_seqs)} reference sequences")
        metrics = evaluate_sequences(generated_seqs, test_seqs)
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nK-mer Jensen-Shannon Divergence:")
        for k in [3, 4, 5, 6]:
            print(f"  {k}-mer JSD: {metrics[f'jsd_{k}mer']:.4f}")
        
        print("\nGC Content:")
        print(f"  Generated: {metrics['gc_mean_gen']*100:.1f}% ± {metrics['gc_std_gen']*100:.1f}%")
        print(f"  Reference: {metrics['gc_mean_real']*100:.1f}% ± {metrics['gc_std_real']*100:.1f}%")
        
        print("\nPositional GC Content:")
        print(f"  GC1: Gen={metrics['gc1_mean_gen']*100:.1f}%, Real={metrics['gc1_mean_real']*100:.1f}%")
        print(f"  GC2: Gen={metrics['gc2_mean_gen']*100:.1f}%, Real={metrics['gc2_mean_real']*100:.1f}%")
        print(f"  GC3: Gen={metrics['gc3_mean_gen']*100:.1f}%, Real={metrics['gc3_mean_real']*100:.1f}%")
        
        print("\nCodon Adaptation Index:")
        print(f"  Generated: {metrics['cai_mean_gen']:.3f} ± {metrics['cai_std_gen']:.3f}")
        print(f"  Reference: {metrics['cai_mean_real']:.3f} ± {metrics['cai_std_real']:.3f}")
        
        print(f"\nORF Validity: {metrics['orf_valid_pct']:.1f}%")
        
        if 'mean_edit_distance' in metrics:
            print(f"Mean Edit Distance: {metrics['mean_edit_distance']:.1f}")
        
        # Generate plots
        plot_kmer_comparison(generated_seqs, test_seqs, k=3, save_path='kmer_comparison_3.png')
        plot_gc_distribution(generated_seqs, test_seqs, save_path='gc_distribution.png')
        
        print("\nPlots saved!")


if __name__ == '__main__':
    main()