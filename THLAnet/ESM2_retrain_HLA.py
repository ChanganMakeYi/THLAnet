from transformers import AutoTokenizer, AutoModelForMaskedLM, TrainingArguments, Trainer
import torch
import pandas as pd
import os
import Data_precessor as dp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

esm_dir = "fabook/esm2-650-MHC"
tokenizer = AutoTokenizer.from_pretrained(esm_dir)
model = AutoModelForMaskedLM.from_pretrained(esm_dir)
model.to(device)
print(f"Loaded ESM-2 150M model from {esm_dir}")

csv_file = "pretrain_data/common_hla_sequence.csv"
df = pd.read_csv(csv_file)
df["HLA_sequence"] = df["HLA_sequence"].astype(str).fillna("")
sequences = df["HLA_sequence"].tolist()
print(f"Loaded {len(sequences)} epitope sequences")

dataset = dp.ProteinMLMDataset(sequences, tokenizer)

training_args = TrainingArguments(
    output_dir="./epitope_pretraining",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=True if torch.cuda.is_available() else False,
)

n_layers = len(model.esm.encoder.layer)
print(f"Total layers: {n_layers}")
for param in model.parameters():
    param.requires_grad = False
for i, layer in enumerate(model.esm.encoder.layer[:-4]):
    for param in layer.parameters():
        param.requires_grad = True

for name, param in model.named_parameters():
    if "encoder.layer" in name:
        print(f"{name}: requires_grad = {param.requires_grad}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./hla_650_pretraining/final_model")
tokenizer.save_pretrained("./hla_650_pretraining/final_model")