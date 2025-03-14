
import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, financial_data, text_embeddings, target, seq_length=10):
        self.financial_data = financial_data
        self.text_embeddings = text_embeddings
        self.target = target
        self.seq_length = seq_length

    def __len__(self):
        return len(self.financial_data) - self.seq_length

    def __getitem__(self, index):
        x_financial = self.financial_data[index:index+self.seq_length]
        x_text = self.text_embeddings[index]
        y = self.target[index+self.seq_length]
        return torch.tensor(x_financial, dtype=torch.float32), torch.tensor(x_text, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
