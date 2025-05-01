# To use langgraph 

# Check how many GPUs are available
# See if MPS could be applied
# Deepseek r1 1,5 - requires 10 gb of VRAM
# quantization 4bit & awq ???
# RAG over 

# Case 1: Multiple GPUs
# Case 2: Single GPU + CPU 
import pandas as pd
import torch
import logging
from socialMediaScraper.WebScraper import WebScraper
import json
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, Qwen2ForSequenceClassification, DataCollatorWithPadding
from peft import PeftModel
import os 
import nvtx

logging.basicConfig(level=logging.INFO)
"""
################ - to match yours
os.environ["TRITON_CACHE_DIR"] = "/home/m/mehrad/brikiyou/scratch/triton_cache"
os.environ["TRITON_HOME"]      = "/home/m/mehrad/brikiyou/scratch/triton_home"
os.environ["CC"]  = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/gcc"
os.environ["CXX"] = "/scinet/balam/rocky9/software/2023a/opt/cuda-12.3.1/gcc/12.3.0/bin/g++"
os.environ["HF_HOME"]            = "/home/m/mehrad/brikiyou/scratch/huggingface_cache"
os.environ["HF_HUB_CACHE"]       = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "models")
os.environ["HF_DATASETS_CACHE"]  = os.path.join(os.environ["HF_HOME"], "datasets")
cache_dir = os.environ["HF_HOME"]
################
"""

class SentimentDataset(Dataset):
    """
    Dataset for JSON or CSV records.
    """
    def __init__(self, records, tokenizer, platform, max_length=512):
        self.records = records
        self.tokenizer = tokenizer
        self.platform = platform
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        if self.platform == "reddit":
            text = rec.get("title", "") + " " + rec.get("description", "")
        elif self.platform == "bluesky":
            text = rec.get("content", "")
        elif self.platform == "csv":
            text = rec.get("title", "") + " " + rec.get("summary", "")
        else:
            raise ValueError(f"Unknown platform: {self.platform}")

        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}


logger = logging.getLogger(__name__)
class Agent:
    def __init__(self, company=None, tools=None, sentiment_model="deepseek"):
        self.company = company
        self.sentiment_model = sentiment_model.lower()

        # Load the DeepSeek PEFT model
        self.base_model = Qwen2ForSequenceClassification.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            num_labels=3,
            trust_remote_code=True
        )
        self.adapter_dir = "/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/stock_predictor_sentiment_analysis/dora_fine-tuning/data/sentiment_data/model_1.5b"
        self.tokenizer  = AutoTokenizer.from_pretrained(
            self.adapter_dir,
            use_fast=True,
            local_files_only=True
        )
        self.model = PeftModel.from_pretrained(
            self.base_model,
            self.adapter_dir,
            local_files_only=True
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.eos_token_id

        self.tokenizer.save_pretrained(cache_dir)
        self.model.save_pretrained(cache_dir)

        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus == 0:
            logger.warning("No GPUs available. Using CPU.")
        elif self.num_gpus == 1:
            logger.warning("Single GPU detected.")
        else:
            logger.info(f"Using {self.num_gpus} GPUs")
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(self.num_gpus)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.eval()

        self.id2label = {0: "negative", 1: "neutral", 2: "positive"}


    @nvtx.annotate("retrieve_social_media_data", color="blue")
    def _retrieve_date(self):
        ws = WebScraper()
        ws(
           reddit_posts_limit=50,
           bluesky_posts_limit=50,
           reddit_comments_limit=10)
        return ws.date



    def analyze_json(self):
        """Process scraped JSON files."""
        date = self._retrieve_date()
        merged = {"reddit": [], "bluesky": []}
        for plat in merged:
            path = f"data/{plat}_{date}.json"
            if os.path.exists(path):
                merged[plat] = json.load(open(path, 'r', encoding='utf-8'))
        self._batch_infer(merged, ["reddit","bluesky"], suffix=f"_{date}")


    def analyze_csv(self, input_csv: str, output_csv: str = None):
        """Read CSV, batch sentiment, and save three deepseek columns."""
        df = pd.read_csv(input_csv)
        records = df.to_dict(orient='records')

        preds, confs, logits = self._batch_infer(
            {"csv": records}, ["csv"], suffix="", return_scores=True
        )

        df["deepseek_pred"]   = [self.id2label[i] for i in preds]
        df["deepseek_confidence"]  = confs
        df["deepseek_score"] = logits

        out = output_csv or input_csv.replace(".csv", f"_with_deepseek.csv")
        df.to_csv(out, index=False)
        logging.info(f"CSV written to {out}")


    def _batch_infer(self, data_dict, platforms, suffix, return_scores=False):
        all_preds, all_confs, all_logits = {}, {}, {}

        for plat in platforms:
            recs = data_dict[plat]
            ds = SentimentDataset(recs, self.tokenizer, plat)
            collator = DataCollatorWithPadding(self.tokenizer, padding="longest")
            loader = DataLoader(
                ds,
                batch_size=32,
                shuffle=False,
                num_workers=4,
                collate_fn=collator
            )

            preds, confs, logits = [], [], []
            num_batches = len(loader)

            with torch.no_grad():
                for batch_idx, batch in enumerate(loader, start=1):
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    out   = self.model(**batch)
                    logit = out.logits                            
                    prob  = torch.softmax(logit, dim=-1)
                    idx   = logit.argmax(dim=-1)              

                    preds.extend(idx.cpu().tolist())
                    confs.extend(prob.max(dim=-1).values.cpu().tolist())
                    logits.extend(logit.cpu().tolist())

                    bs = next(iter(batch.values())).size(0)
                    logging.info(
                        f"[Batch {batch_idx}/{num_batches}] processed {bs} examples"
                    )

            if plat != "csv":
                for rec, p, c, l in zip(recs, preds, confs, logits):
                    rec["deepseek_pred"]       = self.id2label[p]
                    rec["deepseek_confidence"] = c
                    rec["deepseek_score"]      = l

                path = f"data/{plat}{suffix}_with_deepseek.json"
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(recs, f, ensure_ascii=False, indent=2)
                logging.info(f"Wrote {path}")

            all_preds[plat]  = preds
            all_confs[plat]  = confs
            all_logits[plat] = logits

        if return_scores:
            return (
                all_preds.get("csv", []),
                all_confs.get("csv", []),
                all_logits.get("csv", [])
            )
        return None



if __name__ == "__main__":
    agent = Agent()
    #agent.analyze_json()
    # TO match your own path 
    #agent.analyze_csv(input_csv=input_csv, output_csv=output_csv)
    
