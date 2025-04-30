import os
import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen2ForSequenceClassification
from peft import PeftModel

from transformers import AutoTokenizer
from transformers import Qwen2ForSequenceClassification
import torch

BASE_MODEL  = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
ADAPTER_DIR = "/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/dora_fine-tuning/data/sentiment_data/model_7b"
TEST_CSV    = "data/sentiment_data/test.csv"
OUT_CSV     = "data/sentiment_data/test_with_preds_7b.csv"

import os
os.environ["TRITON_CACHE_DIR"] = "/home/m/mehrad/brikiyou/scratch/triton_cache"
os.environ["TRITON_HOME"]      = "/home/m/mehrad/brikiyou/scratch/triton_home"
os.environ["CC"]  = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/gcc"
os.environ["CXX"] = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/g++"
os.environ["HF_HOME"]            = "/home/m/mehrad/brikiyou/scratch/huggingface_cache"
os.environ["HF_HUB_CACHE"]       = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "models")
os.environ["HF_DATASETS_CACHE"]  = os.path.join(os.environ["HF_HOME"], "datasets")

cache_dir = os.environ["HF_HOME"]


tokenizer = AutoTokenizer.from_pretrained(
    ADAPTER_DIR,
    use_fast=True,
    local_files_only=True
)

base_model = Qwen2ForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=3,
    trust_remote_code=True
)

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_DIR,
    local_files_only=True
)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
tokenizer.save_pretrained(cache_dir)
model.save_pretrained(cache_dir)

model = torch.nn.DataParallel(model, device_ids=[0,1,2,3]).cuda()

model.eval()

class TextDataset(Dataset):
    def __init__(self, texts): self.texts = texts
    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]

df     = pl.read_csv("data/sentiment_data/test.csv")
texts  = df["text"].to_list()
ds     = TextDataset(texts)
loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

all_preds = []
for batch_idx, batch_texts in enumerate(loader, 1):
    enc = tokenizer(
        batch_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    enc = {k: v.cuda() for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits

    preds = logits.argmax(dim=-1).cpu().tolist()
    all_preds.extend(preds)
    print(f"[Batch {batch_idx}/{len(loader)}]  processed {len(batch_texts)} examples")

id2label = {0: "negative", 1: "neutral", 2: "positive"}
mapped   = [id2label[p] for p in all_preds]
df = df.with_columns(pl.Series("predicted_sentiment", mapped))
df.write_csv(OUT_CSV)
print("Done.")
