import pandas as pd
import torch
import argparse
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument('--file_path_train', type=str, help='Path to train_data set file path')
parser.add_argument('--output_file', type=str, help='Path to train_data set file path')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset = pd.read_csv(args.file_path_train)
Antigen = list(dataset['Antigen'].dropna())
CDR3 = list(dataset['CDR3'].dropna())
HLA = list(dataset['HLA_seq'].dropna())

TCR_LENGTH = 25
ANTIGEN_LENGTH = 15
HLA_LENGTH = 34

def adjust_sequence(seq, target_length, pad_char='X'):
    if len(seq) > target_length:
        return seq[:target_length]
    elif len(seq) < target_length:
        return seq + pad_char * (target_length - len(seq))
    return seq

Antigen = [adjust_sequence(seq, ANTIGEN_LENGTH) for seq in Antigen]
CDR3 = [adjust_sequence(seq, TCR_LENGTH) for seq in CDR3]
HLA = [adjust_sequence(seq, HLA_LENGTH) for seq in HLA]

antigen_model_path = "./epitope_650_pretraining/final_model"
cdr3_model_path = "./cdr3_650_pretraining/final_model"
hla_model_path = "./hla_650_pretraining/final_model"

tokenizer = AutoTokenizer.from_pretrained(antigen_model_path)

antigen_model = AutoModelForMaskedLM.from_pretrained(antigen_model_path)
cdr3_model = AutoModelForMaskedLM.from_pretrained(cdr3_model_path)
hla_model = AutoModelForMaskedLM.from_pretrained(hla_model_path)

antigen_model.to(device)
cdr3_model.to(device)
hla_model.to(device)

if torch.cuda.is_available():
    antigen_model = antigen_model.half()
    cdr3_model = cdr3_model.half()
    hla_model = hla_model.half()

antigen_model.eval()
cdr3_model.eval()
hla_model.eval()

def get_representations(model, tokenizer, batch_list, device):
    batch_strs = [seq for _, seq in batch_list]
    inputs = tokenizer(batch_strs, return_tensors="pt", padding=True, truncation=True, add_special_tokens=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[33]
    reps = hidden[:, 1:-1, :]
    return reps

Antigen_list = [(f"protein{i}", seq) for i, seq in enumerate(Antigen)]
CDR3_list = [(f"protein{i}", seq) for i, seq in enumerate(CDR3)]
HLA_list = [(f"protein{i}", seq) for i, seq in enumerate(HLA)]

pHLA_TCR_result = torch.zeros((len(Antigen_list), 80, 1280),
                              dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

batch_size = 5
num_samples = len(Antigen_list)
num_batches = (num_samples + batch_size - 1) // batch_size

for i in tqdm(range(num_batches)):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, num_samples)

    batch_antigen = Antigen_list[start_index:end_index]
    batch_cdr3 = CDR3_list[start_index:end_index]
    batch_hla = HLA_list[start_index:end_index]

    results1 = get_representations(antigen_model, tokenizer, batch_antigen, device)
    results2 = get_representations(cdr3_model, tokenizer, batch_cdr3, device)
    results3 = get_representations(hla_model, tokenizer, batch_hla, device)

    batch_result = torch.cat((results1, results2, results3), dim=1).cpu()

    pad_length = 80 - batch_result.shape[1]
    if pad_length > 0:
        pad = torch.zeros((batch_result.shape[0], pad_length, batch_result.shape[2]), dtype=batch_result.dtype)
        batch_result = torch.cat((batch_result, pad), dim=1)

    pHLA_TCR_result[start_index:end_index] = batch_result

    del results1, results2, results3, batch_result
    torch.cuda.empty_cache()

torch.save(pHLA_TCR_result, args.output_file)