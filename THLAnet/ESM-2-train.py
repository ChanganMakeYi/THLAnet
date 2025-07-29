import esm
import pandas as pd
import torch
import argparse
from tqdm import tqdm
import os

# Set environment variable to reduce memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--file_path_train', type=str, help='Path to train_data set file path')
parser.add_argument('--output_file', type=str, help='Path to train_data set file path')
args = parser.parse_args()

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = pd.read_csv(args.file_path_train)
Antigen = list(dataset['Antigen'].dropna())
CDR3 = list(dataset['CDR3'].dropna())
HLA = list(dataset['HLA_seq'].dropna())

# Define default sequence lengths
TCR_LENGTH = 25
ANTIGEN_LENGTH = 15
HLA_LENGTH = 34


# Function to pad or truncate sequences
def adjust_sequence(seq, target_length, pad_char='X'):
    if len(seq) > target_length:
        return seq[:target_length]
    elif len(seq) < target_length:
        return seq + pad_char * (target_length - len(seq))
    return seq


# Preprocess sequences to fixed lengths
Antigen = [adjust_sequence(seq, ANTIGEN_LENGTH) for seq in Antigen]
CDR3 = [adjust_sequence(seq, TCR_LENGTH) for seq in CDR3]
HLA = [adjust_sequence(seq, HLA_LENGTH) for seq in HLA]

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # Disable dropout for deterministic results

# Use half-precision if supported
if torch.cuda.is_available():
    model = model.half()

# Prepare data lists
Antigen_list = [(f"protein{i}", seq) for i, seq in enumerate(Antigen)]
CDR3_list = [(f"protein{i}", seq) for i, seq in enumerate(CDR3)]
HLA_list = [(f"protein{i}", seq) for i, seq in enumerate(HLA)]

# Initialize result tensor on CPU to save GPU memory
pHLA_TCR_result = torch.zeros((len(Antigen_list), 80, 1280),
                              dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

# Process in chunks to reduce memory usage
batch_size = 5  # Further reduced to minimize memory usage
num_samples = len(Antigen_list)
num_batches = (num_samples + batch_size - 1) // batch_size

for i in tqdm(range(num_batches)):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_samples)

    # Convert only the current chunk to tokens
    batch_antigen = Antigen_list[start_index:end_index]
    batch_cdr3 = CDR3_list[start_index:end_index]
    batch_hla = HLA_list[start_index:end_index]

    batch_labels1, batch_strs1, batch_tokens1 = batch_converter(batch_antigen)
    batch_labels2, batch_strs2, batch_tokens2 = batch_converter(batch_cdr3)
    batch_labels3, batch_str3, batch_tokens3 = batch_converter(batch_hla)

    # Move batch data to GPU
    batch_data1 = batch_tokens1.to(device)
    batch_data2 = batch_tokens2.to(device)
    batch_data3 = batch_tokens3.to(device)

    with torch.no_grad():
        # Process each sequence type
        results1 = model(batch_data1, repr_layers=[33], return_contacts=True)
        results2 = model(batch_data2, repr_layers=[33], return_contacts=True)
        results3 = model(batch_data3, repr_layers=[33], return_contacts=True)

        # Concatenate representations and move to CPU
        batch_result = torch.cat(
            (results1["representations"][33], results2["representations"][33], results3["representations"][33]),
            dim=1
        ).cpu()

        # Store in result tensor
        pHLA_TCR_result[start_index:end_index] = batch_result

    # Clear memory
    del batch_data1, batch_data2, batch_data3, results1, results2, results3, batch_result
    torch.cuda.empty_cache()

# Save results
torch.save(pHLA_TCR_result, args.output_file)