import polars as pl
import argparse
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--model', required=False, type=str,default="None")
model = parser.parse_args().model

if model == "None":
    model = input("Please enter the model name: ")
PATH = os.path.join(os.getcwd(), f"test_with_predictions_{model}_.csv")

try:
    df_test = pl.read_csv(PATH, has_header=True, null_values="error", separator="\t")
except:
    df_test = pl.read_csv(input(), has_header=True, null_values="error", separator=",")
print(df_test.shape)
df_test = df_test.drop_nulls(subset=["predicted_sentiment"])
print(df_test.shape)
try:
    labels = df_test["sentiment"].to_list()
except:
    labels = df_test["label"].to_list()
    
predicted = list(map(lambda x:x.lower(), df_test["predicted_sentiment"]))

report = classification_report(labels, predicted, output_dict=True, labels=["positive", "negative", "neutral"])
report_df = pd.DataFrame(report).transpose().round(2)

# Keep only class-level metrics
metrics_df = report_df.loc[["positive", "negative", "neutral"], ["precision", "recall", "f1-score"]]

plt.figure(figsize=(6, 4))
sns.heatmap(metrics_df, annot=True, cmap="YlGnBu", fmt=".2f", cbar=True)
plt.title(f"Classification Metrics - {model}")
plt.ylabel("Class")
plt.xlabel("Metric")
plt.tight_layout()
plt.savefig(f"classification_metrics_{model}.png")
plt.show()

cm = confusion_matrix(labels, predicted, labels=["positive", "negative", "neutral"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["positive", "negative", "neutral"],
            yticklabels=["positive", "negative", "neutral"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix - {model}")
plt.tight_layout()
plt.savefig(f"confusion_matrix_{model}.png")
plt.show()
