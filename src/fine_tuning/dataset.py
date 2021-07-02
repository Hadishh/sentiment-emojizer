from torch.utils.data import Dataset
import torch
import json
import os
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


class ClassifierDataset(Dataset):
    def __init__(self, json_file, labels_dir, tokenizer):
        self.tokenizer = tokenizer
        self.lines, self.labels = self.load_lines(json_file, labels_dir)
        
    def load_lines(self, file, labels_dir):
        lines = []
        labels = []
        with open(file) as f:
            data = json.load(f)
            for key in data.keys():
                line = ' '.join(data[key])
                label_path = os.path.join(labels_dir, f"{key}.json")
                with open(label_path, 'r') as labels_f:
                    label = json.load(labels_f)
                    label_v = []
                    for key_ in label.keys():
                        label_v.append(label[key_])
                labels.append(label_v)
                lines.append(line)
        return lines, labels
    

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx): 
        text_ = self.lines[idx]
        encoding = self.tokenizer(text_)
        return dict(
            text=text_, 
            input_ids=encoding["input_ids"].flatten(), 
            attention_mask=encoding['attention_mask'].flatten(), 
            labels=torch.FloatTensor(self.labels[idx]))