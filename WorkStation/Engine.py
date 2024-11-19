import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

from WorkStation.Dataloader import PreprocessedKoreanDataset
from WorkStation.ScratchGPT import ScratchGPT
from WorkStation.Train_GPT import train_step

device = "cuda" if torch.cuda.is_available() else "cpu"


data_dir = "C:/junha/Git_Clone/LLM_Classifier/data/training/training/korean"
batch_size = 16

# Create Dataset and DataLoader
dataset = PreprocessedKoreanDataset(data_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example of iterating over the DataLoader
for batch in dataloader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    key_padding_mask = batch['key_padding_mask']

print(input_ids.shape, attention_mask.shape, key_padding_mask.shape)

krbert_tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-Medium", do_lower_case=False)

batch_size = 16
vocab_size = krbert_tokenizer.vocab_size

gpt = ScratchGPT(vocab_size=vocab_size).to(device)
loss_fn = nn.CrossEntropyLoss(ignore_index=krbert_tokenizer.pad_token_id)
optimizer = torch.optim.Adam(gpt.parameters(), lr=1e-5,weight_decay=0.01)


epochs = 10

for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(
        model=gpt,
        dataloader=dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device
    )
    print(f"Epoch {epoch + 1} - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
