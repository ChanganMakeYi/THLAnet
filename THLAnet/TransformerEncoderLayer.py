import torch
import torch.nn as nn
import numpy as np
import argparse
import logging
from tqdm import tqdm
import data_process
import os

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TransformerEncoderLayer(nn.Module):
    """Custom Transformer Encoder Layer with multi-head self-attention and feedforward network."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        """Forward pass through the layer."""
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.feedforward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src

class TransformerEncoder(nn.Module):
    """Transformer Encoder with multiple layers."""
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, src):
        """Forward pass through all layers."""
        for layer in self.layers:
            src = layer(src)
        return src

def validate_args(args):
    """Validate command-line arguments."""
    if not os.path.exists(args.file_path_posit):
        raise FileNotFoundError(f"Input file {args.file_path_posit} does not exist.")
    if args.num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    if args.nhead < 1:
        raise ValueError("nhead must be at least 1.")
    if args.d_model % args.nhead != 0:
        raise ValueError("d_model must be divisible by nhead.")
    if args.dropout < 0 or args.dropout > 1:
        raise ValueError("dropout must be between 0 and 1.")
    if args.batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

def process_data_in_batches(encoder, input_data, batch_size, device):
    """Process input data in batches through the TransformerEncoder."""
    logger.info("Starting batch processing...")
    encoder.eval()
    output_data = []
    input_data = input_data.to(device)

    for i in tqdm(range(0, input_data.size(0), batch_size), desc="Processing batches"):
        batch = input_data[i:i + batch_size].transpose(0, 1)  # Shape: [seq_len=80, batch, d_model=21]
        with torch.no_grad():
            output = encoder(batch)
        output_data.append(output.transpose(0, 1).cpu())  # Shape: [batch, seq_len=80, d_model=21]
        torch.cuda.empty_cache()

    return torch.cat(output_data, dim=0)

def main():
    """Main function to run the TransformerEncoder pipeline."""
    parser = argparse.ArgumentParser(description="Run TransformerEncoder on protein sequence data.")
    parser.add_argument('--file_path_posit', type=str, required=True,
                        help='Path to the input dataset file.')
    parser.add_argument('--output_path', type=str, default='transformer_output.pt',
                        help='Path to save the output tensor.')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers.')
    parser.add_argument('--d_model', type=int, default=21,
                        help='Dimension of the model (must match input feature dimension).')
    parser.add_argument('--nhead', type=int, default=3,
                        help='Number of attention heads.')
    parser.add_argument('--dim_feedforward', type=int, default=2048,
                        help='Dimension of the feedforward network.')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing.')

    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as e:
        logger.error(f"Argument validation failed: {e}")
        return

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and prepare data
    try:
        logger.info("Loading input data...")
        input_data = data_process.get_position_encoding_data(args.file_path_posit)
        logger.info(f"Input data shape: {input_data.shape}")
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        return

    # Validate input data shape
    if input_data.size(2) != args.d_model:
        logger.error(f"Input data feature dimension ({input_data.size(2)}) does not match d_model ({args.d_model})")
        return

    # Initialize model
    try:
        logger.info("Initializing TransformerEncoder...")
        encoder = TransformerEncoder(
            num_layers=args.num_layers,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        ).to(device)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # Process data
    try:
        output_data = process_data_in_batches(encoder, input_data, args.batch_size, device)
        logger.info(f"Output data shape: {output_data.shape}")
    except Exception as e:
        logger.error(f"Failed to process data: {e}")
        return

    # Save output
    try:
        torch.save(output_data, args.output_path)
        logger.info(f"Output saved to {args.output_path}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")

if __name__ == "__main__":
    main()