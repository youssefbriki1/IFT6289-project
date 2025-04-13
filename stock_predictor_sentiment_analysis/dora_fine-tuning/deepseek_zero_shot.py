from langchain_core.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
import polars as pl
import httpx
from deepseek_output_parser import TextClassificationOutputParser


prompt = PromptTemplate(
    template = """
            You are a professional financial sentiment analysis expert. Your task is to classify the sentiment of the following sentence as either positive, negative, or neutral. The output should contain only the sentiment label.
            Sentence: {text}
            Sentiment: 
               """,
    input_variables=["text"]    
)
parser = TextClassificationOutputParser()
llm = OllamaLLM(model="deepseek-r1:32b", temperature=1.2)
chain = prompt | llm.with_retry(retry_if_exception_type=(ValueError, httpx.RemoteProtocolError), wait_exponential_jitter=False,stop_after_attempt=3) | parser

df = pl.read_csv("data/sentiment_data/test.csv", separator="\t")
predictions = []

for text in df["text"]:
    try:
        sentiment = chain.invoke({"text": text})
        print(f"Processed text: {text}, Sentiment: {sentiment}")
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        sentiment = "error"
    predictions.append(sentiment)

df = df.with_columns(pl.Series("predicted_sentiment", predictions))

df.write_csv("data/sentiment_data/test_with_predictions.csv", separator="\t")
