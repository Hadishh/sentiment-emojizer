from torch.utils.data import Dataset
import torch
import json
class MaskedLMDataset(Dataset):
    def __init__(self, json_file, tokenizer):
        self.tokenizer = tokenizer
        self.lines = self.load_lines(json_file)
        self.ids = self.encode_lines(self.lines)
        
    def load_lines(self, file):
        lines = []
        with open(file) as f:
            data = json.load(f)
            for key in data.keys():
                line = ' '.join(data[key])
                lines.append(line)
        return lines
    
    def encode_lines(self, lines):
        batch_encoding = self.tokenizer(
            lines, add_special_tokens=True, truncation=True, max_length=128
        )
        return batch_encoding["input_ids"]

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return torch.tensor(self.ids[idx], dtype=torch.long)