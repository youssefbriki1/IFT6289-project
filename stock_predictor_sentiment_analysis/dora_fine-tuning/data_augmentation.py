from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
import argparse
from deepseek_output_parser import ParaphraseOutputParser
import httpx


#!iconv -f latin1 -t utf8 Sentences_AllAgree.txt -o Sentences_AllAgree_utf8.txt

parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str)
parser.add_argument('--skip', required=False, type=int, default=0)
args = parser.parse_args()
path = args.path.split('/')[:-1]
path = '/'.join(path) + '/'
skip = args.skip	
# path = '/home/m/mehrad/brikiyou/scratch/ift6289/IFT6289-project/data/FinancialPhraseBank-v1.0/'
print(path)
file_name = args.path.split('/')[-1]

llm = OllamaLLM(model="deepseek-r1:32b", temperature=1.2)
parser = ParaphraseOutputParser()
prompt = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="""
You are a professional financial writer. Your task is to rewrite the following sentence 
in a different way, while keeping its original meaning and the same sentiment.

Sentiment: {sentiment}
Original sentence: "{text}"

Paraphrased sentence:
""",
)

chain = prompt | llm.with_retry(retry_if_exception_type=(ValueError, httpx.RemoteProtocolError), wait_exponential_jitter=False,stop_after_attempt=3) | parser

with open(f"{path}{file_name}", 'r') as f:
    lines = f.readlines()
    
for line in lines[skip:]:
    sentence,sentiment = line.split('@')
    sentiment = sentiment.strip()
    for count in range(3):
        augmented_sentence = chain.invoke({"text": sentence, "sentiment": sentiment})
        with open(f"{path}augmented_{file_name}", 'a') as f:
            f.write(f"{augmented_sentence}@{sentiment}\n")
