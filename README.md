# Diffusion Models for Generating Synthetic Protein-Coding Sequences

**Author:** Abel Tadesse (GSR/2025/17)  
**Advisor:** Addane Letta  
**Institution:** Addis Ababa University, College of Technology and Built Environment

## Overview

This repository contains the implementation and paper for a discrete diffusion model (D3PM) designed to generate synthetic protein-coding DNA sequences. The model uses a Transformer-based architecture with absorbing state diffusion for handling discrete nucleotide sequences.

## Files

- `Diffusion_Models_Genomic_Sequences_Paper.pdf` - Complete IEEE-formatted research paper
- `main.tex` - LaTeX source file for the paper
- `references.bib` - BibTeX bibliography file (25 references)
- `genomic_diffusion.py` - Complete PyTorch implementation (~1200 lines)

## Requirements

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn tqdm
```

Optional (for loading real FASTA data):
```bash
pip install biopython
```

## Quick Start

### 1. Run Demo (Quick Test)
```bash
python genomic_diffusion.py --mode demo
```
This will:
- Generate 5,000 synthetic protein-coding sequences
- Train the model for 10 epochs
- Generate and evaluate 100 samples
- Display evaluation metrics

### 2. Full Training
```bash
python genomic_diffusion.py --mode train --epochs 200 --batch_size 64
```

Training options:
- `--epochs`: Number of training epochs (default: 200)
- `--batch_size`: Batch size (default: 64)
- `--seq_length`: Sequence length in nucleotides (default: 300)
- `--data_path`: Path to FASTA file with real sequences (optional)
- `--device`: cuda or cpu (default: cuda)

### 3. Generate Sequences
```bash
python genomic_diffusion.py --mode generate --num_samples 1000 --checkpoint checkpoints/model_best.pt
```

This generates sequences and saves them to `generated_sequences.fasta`.

### 4. Evaluate Generated Sequences
```bash
python genomic_diffusion.py --mode evaluate
```

This computes all evaluation metrics and generates visualization plots.

## Using Your Own Data

If you have protein-coding sequences in FASTA format:

```bash
python genomic_diffusion.py --mode train --data_path your_sequences.fasta --epochs 200
```

The code will:
1. Load sequences from the FASTA file
2. Filter for valid DNA sequences (A, T, G, C only)
3. Split into train/validation/test sets
4. Train the diffusion model

## Model Architecture

- **Vocabulary**: 5 tokens (A, T, G, C, [MASK])
- **Embedding Dimension**: 256
- **Transformer Layers**: 6
- **Attention Heads**: 8
- **Feedforward Dimension**: 1024
- **Diffusion Timesteps**: 500
- **Parameters**: ~8M

## Evaluation Metrics

The code computes:
1. **K-mer JSD**: Jensen-Shannon divergence for k=3,4,5,6
2. **GC Content**: Overall and positional (GC1, GC2, GC3)
3. **CAI**: Codon Adaptation Index using human codon usage
4. **ORF Validity**: Percentage of sequences with valid reading frames
5. **Diversity**: Edit distance and unique k-mer percentages

## Expected Results

After full training (200 epochs), you should see:
- 3-mer JSD: ~0.03
- GC Content: ~52%
- CAI: ~0.71
- ORF Valid: ~87%

## GPU Requirements

- Training: ~4GB VRAM (batch_size=64)
- Generation: ~2GB VRAM

For CPU training, use `--device cpu` (much slower).

## Citation

If you use this code, please cite:

```bibtex
@article{tadesse2025diffusion,
  title={Diffusion Models for Generating Synthetic Protein-Coding Sequences},
  author={Tadesse, Abel},
  institution={Addis Ababa University},
  year={2025}
}
```

## Troubleshooting

**CUDA out of memory**: Reduce batch size
```bash
python genomic_diffusion.py --mode train --batch_size 32
```

**Slow training on CPU**: Use fewer epochs for testing
```bash
python genomic_diffusion.py --mode demo --device cpu
```

**Want to use real RefSeq data**: Download CDS sequences from NCBI and provide the FASTA file path.

## License

This code is provided for academic research purposes.
