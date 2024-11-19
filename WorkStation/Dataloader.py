import os
import torch
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"


class PreprocessedKoreanDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = torch.load(file_path, weights_only=True)

        input_ids = data['input_ids']
        attention_mask = data['attention_mask']

        # Create key_padding_mask based on attention_mask
        # Key padding mask is 1 for padded tokens, 0 for non-padded tokens
        key_padding_mask = (attention_mask == 0).bool()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'key_padding_mask': key_padding_mask
        }



